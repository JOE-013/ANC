#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_pesq_metrics.py - Standalone script to compute PESQ scores on existing enhanced WAV outputs.
Updates data_test/eval_results_smoke_test.csv and displays before/after PESQ scores by SNR level.
"""

import os
import glob
import csv
import numpy as np
import soundfile as sf
import librosa

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("ERROR: 'pesq' library is not installed yet. Please install C++ Build Tools and run 'pip install pesq'.")

def main():
    if not HAS_PESQ:
        print("Cannot compute PESQ: 'pesq' module unavailable.")
        return

    clean_dir = r"d:\SIH 2026\DTLN\data_test\train_speech"
    noisy_dir = r"d:\SIH 2026\DTLN\data_test\train_mix"
    enhanced_dir = r"d:\SIH 2026\DTLN\data_test\enhanced_smoke_test"
    csv_path = r"d:\SIH 2026\DTLN\data_test\eval_results_smoke_test.csv"

    # Read existing CSV
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} entries from {csv_path}. Computing PESQ scores (16kHz wideband)...")

    for r in rows:
        filename = r['filename']
        clean_p = os.path.join(clean_dir, filename)
        noisy_p = os.path.join(noisy_dir, filename)
        enhanced_p = os.path.join(enhanced_dir, filename)

        clean_sig, fs = librosa.core.load(clean_p, sr=16000, mono=True)
        noisy_sig, _ = librosa.core.load(noisy_p, sr=16000, mono=True)
        enhanced_sig, _ = librosa.core.load(enhanced_p, sr=16000, mono=True)

        min_len = min(len(clean_sig), len(noisy_sig), len(enhanced_sig))
        clean_sig = clean_sig[:min_len]
        noisy_sig = noisy_sig[:min_len]
        enhanced_sig = enhanced_sig[:min_len]

        pesq_b = pesq(16000, clean_sig, noisy_sig, 'wb')
        pesq_a = pesq(16000, clean_sig, enhanced_sig, 'wb')
        pesq_g = pesq_a - pesq_b

        r['pesq_before'] = f"{pesq_b:.4f}"
        r['pesq_after'] = f"{pesq_a:.4f}"
        r['pesq_gain'] = f"{pesq_g:.4f}"

    # Write updated CSV
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully updated {csv_path} with PESQ metrics.")

    # Summary Table
    snrs = sorted(list(set(float(r['input_snr_cond']) for r in rows)))
    print("\n" + "="*70)
    print(f"{'Input SNR':<12} | {'PESQ Before (Noisy)':<20} | {'PESQ After (Enhanced)':<22} | {'PESQ Gain':<10}")
    print("="*70)

    all_b, all_a = [], []
    for snr in snrs:
        sub = [r for r in rows if float(r['input_snr_cond']) == snr]
        pb = [float(r['pesq_before']) for r in sub]
        pa = [float(r['pesq_after']) for r in sub]
        all_b.extend(pb)
        all_a.extend(pa)
        print(f"{snr:<12.1f}dB | {np.mean(pb):<20.3f} | {np.mean(pa):<22.3f} | {np.mean(pa)-np.mean(pb):<+10.3f}")

    print("-" * 70)
    print(f"{'OVERALL':<12} | {np.mean(all_b):<20.3f} | {np.mean(all_a):<22.3f} | {np.mean(all_a)-np.mean(all_b):<+10.3f}")
    print("="*70)

if __name__ == '__main__':
    main()
