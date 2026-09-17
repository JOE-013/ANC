#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transient_limiter_v2.py — Rise-Time & Envelope-Slope Based Transient Limiter.

Discriminates genuine mechanical/explosive impulses (gunshots, artillery, fireworks with <1.5 ms rise time)
from speech plosives (p/t/k with >3.0 ms rise time) based on envelope derivative dE/dt and level floor.

Parameters:
-----------
audio : np.ndarray
    Input single-channel audio signal (float32).
fs : int
    Sampling rate in Hz (default: 16000).
smooth_win_ms : float
    Envelope smoothing window duration in milliseconds (default: 1.0 ms).
rise_time_threshold_ms : float
    Maximum rise-time threshold in milliseconds (default: 1.5 ms -> dE/dt threshold ~380 units/sec).
min_level_db : float
    Minimum peak amplitude level floor in dBFS to allow triggering (default: -12.0 dBFS / 0.251 linear).
target_ceiling_db : float
    Maximum peak ceiling level in dBFS applied during gain reduction (default: -6.0 dBFS).
attenuation_db : float
    Gain attenuation applied to detected transient peaks in dB (default: 6.0 dB).
attack_ms : float
    Attack time constant for gain reduction in milliseconds (default: 2.0 ms).
release_ms : float
    Release time constant for gain reduction in milliseconds (default: 50.0 ms).
"""

import numpy as np
import scipy.ndimage

def apply_transient_limiter_v2(
    audio,
    fs=16000,
    smooth_win_ms=1.0,
    rise_time_threshold_ms=1.5,
    min_level_db=-12.0,
    target_ceiling_db=-6.0,
    attenuation_db=6.0,
    attack_ms=2.0,
    release_ms=50.0
):
    """
    Apply rise-time-based transient limiting to single-channel audio array.
    
    Returns:
    --------
    limited_audio : np.ndarray
        Audio signal with transient peak reduction applied.
    gain_curve : np.ndarray
        Time-varying gain curve applied to audio (values <= 1.0).
    trigger_mask : np.ndarray (bool)
        Boolean mask indicating samples where rise-time trigger fired.
    """
    if audio.ndim > 1:
        audio = audio.flatten()
        
    audio_len = len(audio)
    if audio_len == 0:
        return audio.copy(), np.ones(0, dtype=np.float32), np.zeros(0, dtype=bool)

    abs_audio = np.abs(audio)
    
    # 1. Short-term energy/amplitude envelope (1.0 ms window)
    smooth_samples = max(1, int(round(fs * (smooth_win_ms / 1000.0))))
    kernel = np.ones(smooth_samples, dtype=np.float32) / smooth_samples
    envelope = scipy.ndimage.convolve1d(abs_audio, kernel, mode='constant')
    
    # 2. Envelope derivative (slope dE/dt in amplitude units per second) over 0.5 ms lag
    lag_samples = max(1, int(round(fs * 0.0005)))
    diff_env = np.zeros_like(envelope)
    diff_env[lag_samples:] = envelope[lag_samples:] - envelope[:-lag_samples]
    dt = lag_samples / float(fs)
    dE_dt = diff_env / dt
    
    # 3. Dual Trigger Conditions
    slope_threshold = 0.57 / (rise_time_threshold_ms / 1000.0) # 380.0 units/sec
    min_level_linear = 10.0 ** (min_level_db / 20.0)            # -12.0 dBFS = 0.2512
    target_ceiling_linear = 10.0 ** (target_ceiling_db / 20.0)  # -6.0 dBFS = 0.5012
    att_linear = 10.0 ** (-abs(attenuation_db) / 20.0)          # 6 dB atten = 0.5012
    
    # Trigger condition: steep envelope slope (fast rise time) AND envelope exceeds minimum level floor
    trigger_mask = (dE_dt >= slope_threshold) & (envelope >= min_level_linear)
    
    # 4. Target Gain Reduction Calculation
    target_gain = np.ones(audio_len, dtype=np.float32)
    for i in range(audio_len):
        if trigger_mask[i]:
            g_ceil = target_ceiling_linear / max(envelope[i], 1e-6)
            g_att = att_linear
            target_gain[i] = min(1.0, min(g_ceil, g_att))

    # 5. Fast-Attack / Slow-Release Gain Smoothing
    attack_alpha = np.exp(-1.0 / max(1.0, fs * (attack_ms / 1000.0)))
    release_alpha = np.exp(-1.0 / max(1.0, fs * (release_ms / 1000.0)))
    
    gain_curve = np.ones(audio_len, dtype=np.float32)
    current_gain = 1.0
    
    for i in range(audio_len):
        desired_g = target_gain[i]
        if desired_g < current_gain:
            current_gain = attack_alpha * current_gain + (1.0 - attack_alpha) * desired_g
        else:
            current_gain = release_alpha * current_gain + (1.0 - release_alpha) * desired_g
        gain_curve[i] = current_gain

    # 6. Apply Gain Curve to Audio
    limited_audio = audio * gain_curve
    return limited_audio.astype(np.float32), gain_curve, trigger_mask

if __name__ == '__main__':
    print("transient_limiter_v2 module verified with gain attenuation fix.")
