#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_risetime_detector.py — Standalone validation of transient_limiter_v2 rise-time detection & gain attenuation.
Tests detector on raw impulsive audio (ESC-50 fireworks) vs clean validation speech.
Prints envelope peaks, sample trigger counts, discrete event counts, min gain values, and status.
"""

import os
import glob
import numpy as np
import librosa
from transient_limiter_v2 import apply_transient_limiter_v2

def count_discrete_events(mask, min_gap_samples=160):
    """
    Count contiguous trigger events separated by at least min_gap_samples (10ms at 16kHz).
    """
    if not np.any(mask):
        return 0
    trig_indices = np.where(mask)[0]
    gaps = np.diff(trig_indices)
    events = 1 + np.sum(gaps > min_gap_samples)
    return int(events)

def main():
    script_dir = r"d:\SIH 2026\DTLN"
    esc50_dir = os.path.join(script_dir, "datasets", "esc50", "audio")
    speech_dir = os.path.join(script_dir, "data_full", "val_speech")
    
    # 5 Impulsive files (ESC-50 Category 48 Fireworks)
    fireworks_files = sorted(glob.glob(os.path.join(esc50_dir, "*-48.wav")))[:5]
    # 5 Clean expressive speech files from validation set
    speech_files = sorted(glob.glob(os.path.join(speech_dir, "*.wav")))[:5]
    
    print("=== Transient Limiter V2 Standalone Validation Report ===")
    print("Parameters: smooth_win_ms=1.0ms, rise_time_threshold_ms=1.5ms, min_level_db=-12.0dBFS, target_ceiling_db=-6.0dBFS, attenuation_db=6.0dB\n")
    
    print(f"{'Category':<12} | {'Filename':<35} | {'Peak Env':<9} | {'Sample Trigs':<12} | {'Events':<7} | {'Min Gain':<9} | {'Status':<15}")
    print("-" * 115)
    
    # 1. Impulsive Files
    for fpath in fireworks_files:
        filename = os.path.basename(fpath)
        audio, fs = librosa.load(fpath, sr=16000, mono=True)
        limited, gain_curve, mask = apply_transient_limiter_v2(
            audio, fs=fs,
            smooth_win_ms=1.0,
            rise_time_threshold_ms=1.5,
            min_level_db=-12.0,
            target_ceiling_db=-6.0,
            attenuation_db=6.0,
            attack_ms=2.0,
            release_ms=50.0
        )
        
        peak_env = float(np.max(np.abs(audio)))
        num_triggers = int(np.sum(mask))
        num_events = count_discrete_events(mask)
        min_g = float(np.min(gain_curve))
        status = "FIRED (Correct)" if num_triggers > 0 else "NO TRIGGER"
        
        print(f"{'Impulsive':<12} | {filename:<35} | {peak_env:<9.3f} | {num_triggers:<12d} | {num_events:<7d} | {min_g:<9.3f} | {status:<15}")

    print("-" * 115)

    # 2. Clean Speech Files
    for fpath in speech_files:
        filename = os.path.basename(fpath)
        audio, fs = librosa.load(fpath, sr=16000, mono=True)
        limited, gain_curve, mask = apply_transient_limiter_v2(
            audio, fs=fs,
            smooth_win_ms=1.0,
            rise_time_threshold_ms=1.5,
            min_level_db=-12.0,
            target_ceiling_db=-6.0,
            attenuation_db=6.0,
            attack_ms=2.0,
            release_ms=50.0
        )
        
        peak_env = float(np.max(np.abs(audio)))
        num_triggers = int(np.sum(mask))
        num_events = count_discrete_events(mask)
        min_g = float(np.min(gain_curve))
        status = "SILENT (Correct)" if num_triggers == 0 else f"FALSE TRIGGER ({num_triggers})"
        
        print(f"{'Clean Speech':<12} | {filename:<35} | {peak_env:<9.3f} | {num_triggers:<12d} | {num_events:<7d} | {min_g:<9.3f} | {status:<15}")

    print("=" * 115)

if __name__ == '__main__':
    main()
