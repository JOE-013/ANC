#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_standalone_limiter.py - Standalone test script for transient_limiter.py.
Applies the limiter directly to raw noisy validation audio files (-5dB and 0dB SNR).
"""

import os
import glob
import re
import numpy as np
import soundfile as sf
import librosa
from transient_limiter import apply_transient_limiter

def main():
    print("=== Standalone Test: Energy Transient Limiter on Raw Noisy Validation Audio ===")
    
    val_mix_dir = r"d:\SIH 2026\DTLN\data_full\val_mix"
    out_dir = r"d:\SIH 2026\DTLN\data_full\standalone_limiter_test"
    os.makedirs(out_dir, exist_ok=True)
    
    noisy_files = sorted(glob.glob(os.path.join(val_mix_dir, "*.wav")))
    
    # Filter files from -5dB and 0dB SNR conditions
    harshest_files = []
    for f in noisy_files:
        fn = os.path.basename(f)
        snr_match = re.search(r'snr(-?\d+)dB', fn)
        if snr_match and float(snr_match.group(1)) in [-5.0, 0.0]:
            harshest_files.append(f)
            
    # Pick top 8 files for standalone testing
    test_sample = harshest_files[:8]
    print(f"Selected {len(test_sample)} harsh noisy clips (-5dB and 0dB SNR) for testing.\n")
    
    print("="*90)
    print(f"{'Filename':<38} | {'Peak Before':<12} | {'Peak After':<12} | {'Peak Atten (dB)':<16} | {'Min Gain':<10}")
    print("="*90)
    
    for path in test_sample:
        fn = os.path.basename(path)
        out_path = os.path.join(out_dir, fn)
        
        audio, fs = librosa.core.load(path, sr=16000, mono=True)
        
        lim_audio, gain = apply_transient_limiter(
            audio, 
            fs=fs, 
            threshold_db=-6.0, 
            attack_ms=2.0, 
            release_ms=50.0, 
            win_ms=3.0
        )
        
        sf.write(out_path, lim_audio, fs)
        
        peak_b = np.max(np.abs(audio))
        peak_a = np.max(np.abs(lim_audio))
        min_g = np.min(gain)
        atten_db = 20.0 * np.log10((peak_a + 1e-12) / (peak_b + 1e-12))
        
        print(f"{fn:<38} | {peak_b:<12.4f} | {peak_a:<12.4f} | {atten_db:<+16.2f} | {min_g:<10.4f}")
        
    print("="*90)
    print(f"Outputs written to {out_dir}\n")

if __name__ == '__main__':
    main()
