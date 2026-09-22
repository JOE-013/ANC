#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_full_dataset.py - Automated verification for generated full DTLN dataset (data_full).
"""

import os
import glob
import numpy as np
import soundfile as sf
from DTLN_model import audio_generator

def compute_rms(signal):
    return np.sqrt(np.mean(np.square(signal)) + 1e-12)

def verify_set(mix_dir, speech_dir, expected_count, label="Train"):
    print(f"\n--- Verifying {label} Set ---")
    noisy_files = sorted(glob.glob(os.path.join(mix_dir, "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(speech_dir, "*.wav")))

    print(f"{label} Noisy files count: {len(noisy_files)}")
    print(f"{label} Clean files count: {len(clean_files)}")

    assert len(noisy_files) == expected_count, f"{label} noisy count {len(noisy_files)} != expected {expected_count}"
    assert len(clean_files) == expected_count, f"{label} clean count {len(clean_files)} != expected {expected_count}"

    mismatches = 0
    sample_rates = set()
    durations = []
    snr_diffs = []

    for n_p, c_p in zip(noisy_files, clean_files):
        n_name = os.path.basename(n_p)
        c_name = os.path.basename(c_p)
        if n_name != c_name:
            mismatches += 1

        n_data, n_fs = sf.read(n_p)
        c_data, c_fs = sf.read(c_p)

        sample_rates.add(n_fs)
        sample_rates.add(c_fs)
        durations.append(len(c_data) / c_fs)

        assert n_data.ndim == 1, f"File {n_name} is not mono!"
        assert c_data.ndim == 1, f"File {c_name} is not mono!"

        parts = n_name.split("_")
        target_snr = float(parts[3].replace("snr", "").replace("dB", ""))
        noise_part = n_data - c_data
        measured_snr = 20.0 * np.log10(compute_rms(c_data) / compute_rms(noise_part))
        snr_diffs.append(abs(measured_snr - target_snr))

    print(f"Filename Mismatches: {mismatches}")
    print(f"Sample Rates: {sample_rates} (Expected: {{16000}})")
    print(f"Mean Duration: {np.mean(durations):.2f}s (Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s)")
    print(f"SNR Accuracy: Mean Diff = {np.mean(snr_diffs):.3f} dB, Max Diff = {np.max(snr_diffs):.3f} dB")
    assert np.max(snr_diffs) < 0.5, f"SNR difference too high! Max diff: {np.max(snr_diffs):.3f} dB"
    print(f"-> PASS: {label} set format and SNR spot-check verified!")

def main():
    print("=== Automated Verification for data_full ===")
    
    train_mix = r"d:\SIH 2026\DTLN\data_full\train_mix"
    train_speech = r"d:\SIH 2026\DTLN\data_full\train_speech"
    val_mix = r"d:\SIH 2026\DTLN\data_full\val_mix"
    val_speech = r"d:\SIH 2026\DTLN\data_full\val_speech"

    verify_set(train_mix, train_speech, 3570, label="Train")
    verify_set(val_mix, val_speech, 630, label="Validation")

    print("\n--- DTLN_model.py audio_generator Compatibility ---")
    gen_train = audio_generator(train_mix, train_speech, len_of_samples=240000, fs=16000, train_flag=True)
    print(f"Train audio_generator counted total_samples: {gen_train.total_samples}")

    gen_val = audio_generator(val_mix, val_speech, len_of_samples=240000, fs=16000, train_flag=False)
    print(f"Val audio_generator counted total_samples: {gen_val.total_samples}")

    in_c, tar_c = next(iter(gen_train.tf_data_set))
    print(f"Yielded train chunk shape: input = {in_c.shape}, target = {tar_c.shape}")

    def get_dir_size_mb(path):
        total = 0
        for r, d, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(r, f))
        return total / (1024 * 1024)

    train_size = get_dir_size_mb(r"d:\SIH 2026\DTLN\data_full\train_mix") + get_dir_size_mb(r"d:\SIH 2026\DTLN\data_full\train_speech")
    val_size = get_dir_size_mb(r"d:\SIH 2026\DTLN\data_full\val_mix") + get_dir_size_mb(r"d:\SIH 2026\DTLN\data_full\val_speech")

    print(f"\ndata_full/train Size: {train_size:.2f} MB")
    print(f"data_full/val Size:   {val_size:.2f} MB")
    print(f"Total data_full Size: {train_size + val_size:.2f} MB")

if __name__ == '__main__':
    main()
