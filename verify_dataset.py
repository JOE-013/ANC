#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_dataset.py - Automated verification for generated DTLN training audio pairs.
Checks SNR accuracy, file pairing, mono 16kHz specs, and DTLN_model.py audio_generator compatibility.
"""

import os
import glob
import numpy as np
import soundfile as sf
from DTLN_model import audio_generator

def compute_rms(signal):
    return np.sqrt(np.mean(np.square(signal)) + 1e-12)

def main():
    noisy_dir = r"d:\SIH 2026\DTLN\data_test\train_mix"
    clean_dir = r"d:\SIH 2026\DTLN\data_test\train_speech"

    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))

    print("=== 1. File Pairing & Format Checks ===")
    print(f"Noisy files count: {len(noisy_files)}")
    print(f"Clean files count: {len(clean_files)}")

    assert len(noisy_files) == len(clean_files), "Mismatch in file counts between noisy and clean directories!"
    assert len(noisy_files) > 0, "No files found to verify!"

    mismatches = 0
    sample_rates = set()
    durations = []

    for n_path, c_path in zip(noisy_files, clean_files):
        n_name = os.path.basename(n_path)
        c_name = os.path.basename(c_path)
        if n_name != c_name:
            mismatches += 1

        n_data, n_fs = sf.read(n_path)
        c_data, c_fs = sf.read(c_path)

        sample_rates.add(n_fs)
        sample_rates.add(c_fs)
        durations.append(len(c_data) / c_fs)

        # Confirm mono
        assert n_data.ndim == 1, f"File {n_name} is not mono!"
        assert c_data.ndim == 1, f"File {c_name} is not mono!"

    print(f"Filename Mismatches: {mismatches}")
    print(f"Sample Rates Found: {sample_rates} (Expected: {{16000}})")
    print(f"Average Duration: {np.mean(durations):.2f}s (Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s)")

    print("\n=== 2. Measured SNR Spot-Check ===")
    snr_results = []
    for n_path, c_path in zip(noisy_files, clean_files):
        n_name = os.path.basename(n_path)
        n_data, _ = sf.read(n_path)
        c_data, _ = sf.read(c_path)

        # Extract target SNR from filename (e.g. pair_0000_snr-5dB_gunshot.wav)
        parts = n_name.split("_")
        target_snr = float(parts[2].replace("snr", "").replace("dB", ""))

        noise_part = n_data - c_data
        measured_snr = 20.0 * np.log10(compute_rms(c_data) / compute_rms(noise_part))
        diff = abs(measured_snr - target_snr)

        snr_results.append((n_name, target_snr, measured_snr, diff))

    print(f"{'Filename':<42} | {'Target SNR':<10} | {'Measured SNR':<12} | {'Diff (dB)':<10}")
    print("-" * 80)
    for n_name, target_snr, measured_snr, diff in snr_results[:10]:
        print(f"{n_name:<42} | {target_snr:<10.1f} | {measured_snr:<12.2f} | {diff:<10.3f}")

    max_diff = max(r[3] for r in snr_results)
    mean_diff = np.mean([r[3] for r in snr_results])
    print(f"\nSNR Accuracy: Mean Diff = {mean_diff:.3f} dB, Max Diff = {max_diff:.3f} dB")
    assert max_diff < 0.5, f"SNR difference too high! Max diff: {max_diff:.3f} dB"
    print("-> PASS: SNR accuracy spot-check within tolerances!")

    print("\n=== 3. DTLN_model.py audio_generator Compatibility Check ===")
    gen = audio_generator(noisy_dir, clean_dir, len_of_samples=240000, fs=16000, train_flag=True)
    print(f"audio_generator counted total_samples: {gen.total_samples}")

    dataset = gen.tf_data_set
    sample_batch = next(iter(dataset))
    in_chunk, tar_chunk = sample_batch

    print(f"Yielded batch shape: input = {in_chunk.shape}, target = {tar_chunk.shape}")
    print(f"Yielded batch dtypes: input = {in_chunk.dtype}, target = {tar_chunk.dtype}")

    assert in_chunk.shape == (240000,), f"Unexpected input chunk shape: {in_chunk.shape}"
    assert tar_chunk.shape == (240000,), f"Unexpected target chunk shape: {tar_chunk.shape}"
    print("-> PASS: DTLN_model.py audio_generator successfully loaded and yielded data chunks!")

    print("\n=== 4. Disk Usage Calculation ===")
    def get_dir_size_mb(path):
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)

    esc50_size = get_dir_size_mb(r"d:\SIH 2026\DTLN\datasets\esc50")
    libri_size = get_dir_size_mb(r"d:\SIH 2026\DTLN\datasets\librispeech_dev_clean")
    generated_size = get_dir_size_mb(r"d:\SIH 2026\DTLN\data_test")

    print(f"ESC-50 Dataset Size:             {esc50_size:.2f} MB")
    print(f"LibriSpeech dev-clean Size:       {libri_size:.2f} MB")
    print(f"Generated Test Dataset Size:      {generated_size:.2f} MB")
    print(f"Total Datasets Directory Size:   {esc50_size + libri_size + generated_size:.2f} MB")

if __name__ == '__main__':
    main()
