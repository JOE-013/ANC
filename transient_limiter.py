#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transient_limiter.py - Energy-based transient limiter for audio post-processing.
Supports both absolute dBFS thresholding and adaptive relative transient detection.
In relative mode, transients are detected when short-window peak energy exceeds
the local (~100ms) RMS energy envelope by a specified margin (relative_margin_db).
"""

import numpy as np
import scipy.ndimage

def apply_transient_limiter(
    audio, 
    fs=16000, 
    threshold_db=-6.0, 
    attack_ms=2.0, 
    release_ms=50.0, 
    win_ms=3.0,
    use_relative_thresh=True,
    relative_margin_db=10.0,
    local_win_ms=100.0
):
    """
    Apply short-window energy transient limiting to mono audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Mono 1D float32/float64 audio array (16kHz).
    fs : int
        Sample rate in Hz (default: 16000).
    threshold_db : float
        Absolute ceiling threshold in dBFS (default: -6.0 dBFS).
    attack_ms : float
        Attack time constant in milliseconds (default: 2.0 ms).
    release_ms : float
        Release time constant in milliseconds (default: 50.0 ms).
    win_ms : float
        Short-window transient peak analysis duration in milliseconds (default: 3.0 ms).
    use_relative_thresh : bool
        If True, transients are detected when short peak exceeds local RMS by relative_margin_db (default: True).
    relative_margin_db : float
        Transient detection crest-factor margin in dB above local RMS (default: 10.0 dB).
    local_win_ms : float
        Local RMS energy window in milliseconds (default: 100.0 ms).
        
    Returns:
    --------
    limited_audio : np.ndarray
        Audio with transient peak reduction applied.
    gain_curve : np.ndarray
        Time-varying gain reduction curve (values <= 1.0).
    """
    if audio.ndim > 1:
        audio = audio.flatten()
        
    audio_len = len(audio)
    if audio_len == 0:
        return audio.copy(), np.ones(0, dtype=np.float32)

    # Convert window lengths to samples
    win_samples = max(1, int(round(fs * (win_ms / 1000.0))))
    local_win_samples = max(1, int(round(fs * (local_win_ms / 1000.0))))
    
    abs_audio = np.abs(audio)
    
    # 1. Short-window peak envelope (e.g. 3ms)
    peak_envelope = scipy.ndimage.maximum_filter1d(abs_audio, size=win_samples, mode='constant')
    
    # 2. Determine dynamic threshold curve
    abs_ceiling = 10.0 ** (threshold_db / 20.0)
    
    if use_relative_thresh:
        # Local RMS energy envelope over local_win_ms (~100ms)
        local_rms = np.sqrt(scipy.ndimage.uniform_filter1d(abs_audio ** 2, size=local_win_samples, mode='constant') + 1e-12)
        margin_linear = 10.0 ** (relative_margin_db / 20.0)
        
        # Transient threshold is local RMS + margin, capped at absolute ceiling
        thresh_curve = local_rms * margin_linear
        thresh_curve = np.minimum(thresh_curve, abs_ceiling)
    else:
        thresh_curve = np.full(audio_len, abs_ceiling, dtype=np.float32)

    # 3. Compute un-smoothed target gain curve
    over_mask = peak_envelope > thresh_curve
    g_target = np.ones(audio_len, dtype=np.float32)
    g_target[over_mask] = thresh_curve[over_mask] / (peak_envelope[over_mask] + 1e-12)

    # 4. Apply asymmetric gain smoothing (Fast Attack, Smooth Release)
    alpha_attack = np.exp(-1.0 / (fs * (attack_ms / 1000.0))) if attack_ms > 0 else 0.0
    alpha_release = np.exp(-1.0 / (fs * (release_ms / 1000.0))) if release_ms > 0 else 0.0

    g_smoothed = np.ones(audio_len, dtype=np.float32)
    current_gain = 1.0

    for n in range(audio_len):
        target = g_target[n]
        if target < current_gain:
            # Attack phase
            current_gain = alpha_attack * current_gain + (1.0 - alpha_attack) * target
        else:
            # Release phase
            current_gain = alpha_release * current_gain + (1.0 - alpha_release) * target
        g_smoothed[n] = current_gain

    # 5. Apply gain curve
    limited_audio = audio * g_smoothed
    return limited_audio.astype(np.float32), g_smoothed.astype(np.float32)

if __name__ == '__main__':
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sig = 0.1 * np.sin(2 * np.pi * 440 * t)
    # Add sudden transient spike
    sig[4000:4050] += 0.85
    
    lim_sig, gain = apply_transient_limiter(sig, fs=fs, use_relative_thresh=True, relative_margin_db=10.0)
    print(f"Relative limiter test complete. Max peak before: {np.max(np.abs(sig)):.4f}, after: {np.max(np.abs(lim_sig)):.4f}")
    print(f"Min gain applied: {np.min(gain):.4f}")
