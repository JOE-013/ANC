#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_full_dataset.py - Full training & validation dataset generator for DTLN.
Generates 800 disjoint paired audio files (680 train, 120 val) using LibriSpeech dev-clean,
all 50 ESC-50 categories, and MUSAN (noise & speech/babble subfolders).
"""

import os
import glob
import csv
import random
import numpy as np
import scipy.signal
import soundfile as sf

def compute_rms(audio):
    return np.sqrt(np.mean(np.square(audio)) + 1e-12)

def load_and_resample(file_path, target_fs=16000):
    data, fs = sf.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if fs != target_fs:
        num_target_samples = int(round(len(data) * target_fs / fs))
        data = scipy.signal.resample(data, num_target_samples)
    return data.astype(np.float32), target_fs

def ensure_duration(audio, min_samples):
    if len(audio) >= min_samples:
        return audio[:min_samples]
    repeats = int(np.ceil(min_samples / len(audio)))
    audio_extended = np.tile(audio, repeats)
    return audio_extended[:min_samples]

def mix_audio(speech, noise, target_snr_db):
    rms_speech = compute_rms(speech)
    rms_noise = compute_rms(noise)

    if rms_noise == 0 or rms_speech == 0:
        raise ValueError("Audio clip has zero RMS energy.")

    target_noise_rms = rms_speech / (10 ** (target_snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / rms_noise)
    noisy = speech + scaled_noise

    # Peak normalization to <= 0.99
    max_peak = np.max(np.abs(noisy))
    if max_peak > 0.99:
        norm_factor = 0.99 / max_peak
        noisy = noisy * norm_factor
        speech = speech * norm_factor
        scaled_noise = scaled_noise * norm_factor

    noise_part = noisy - speech
    measured_snr_db = 20.0 * np.log10(compute_rms(speech) / compute_rms(noise_part))
    return noisy.astype(np.float32), speech.astype(np.float32), measured_snr_db

def generate_pairs_set(speech_pool, noise_pool, out_noisy_dir, out_clean_dir, snrs, num_pairs, prefix="train", target_fs=16000, target_duration=15.0):
    os.makedirs(out_noisy_dir, exist_ok=True)
    os.makedirs(out_clean_dir, exist_ok=True)

    min_samples = int(target_fs * target_duration)
    results = []

    for i in range(num_pairs):
        sp_file = speech_pool[i % len(speech_pool)]
        ns_file, ns_cat = noise_pool[i % len(noise_pool)]
        snr = snrs[i % len(snrs)]

        # Load & process
        speech_audio, _ = load_and_resample(sp_file, target_fs=target_fs)
        speech_audio = ensure_duration(speech_audio, min_samples)

        noise_audio, _ = load_and_resample(ns_file, target_fs=target_fs)
        noise_audio = ensure_duration(noise_audio, len(speech_audio))

        noisy_audio, clean_audio, measured_snr = mix_audio(speech_audio, noise_audio, snr)

        filename = f"pair_{prefix}_{i:04d}_snr{int(snr)}dB_{ns_cat}.wav"
        out_noisy_path = os.path.join(out_noisy_dir, filename)
        out_clean_path = os.path.join(out_clean_dir, filename)

        sf.write(out_noisy_path, noisy_audio, target_fs)
        sf.write(out_clean_path, clean_audio, target_fs)

        results.append({
            'filename': filename,
            'target_snr': snr,
            'measured_snr': measured_snr,
            'category': ns_cat,
            'split': prefix
        })
    return results

def main():
    print("=== DTLN Full Dataset Generator (ESC-50 + MUSAN + LibriSpeech) ===")

    libri_dir = r"d:\SIH 2026\DTLN\datasets\librispeech_dev_clean"
    esc50_dir = r"d:\SIH 2026\DTLN\datasets\esc50\audio"
    musan_noise_dir = r"d:\SIH 2026\DTLN\datasets\musan\musan\noise"
    musan_speech_dir = r"d:\SIH 2026\DTLN\datasets\musan\musan\speech"

    # 1. Gather Clean Speech Files
    speech_files = glob.glob(os.path.join(libri_dir, "**", "*.flac"), recursive=True)
    print(f"Total LibriSpeech clean files found: {len(speech_files)}")
    assert len(speech_files) > 0, "No LibriSpeech clean FLAC files found!"

    random.seed(42)
    random.shuffle(speech_files)

    # Disjoint split: 85% train speech, 15% val speech
    split_idx = int(0.85 * len(speech_files))
    train_speech_pool = speech_files[:split_idx]
    val_speech_pool = speech_files[split_idx:]

    print(f"Train speech pool size: {len(train_speech_pool)} files (85%)")
    print(f"Val speech pool size:   {len(val_speech_pool)} files (15%)")

    # 2. Gather Noise Files
    noise_pool = []

    # ESC-50 noise files
    esc50_wavs = glob.glob(os.path.join(esc50_dir, "*.wav"))
    for w in esc50_wavs:
        noise_pool.append((w, "esc50"))

    # MUSAN noise files
    musan_noise_wavs = glob.glob(os.path.join(musan_noise_dir, "**", "*.wav"), recursive=True)
    for w in musan_noise_wavs:
        noise_pool.append((w, "musan_noise"))

    # MUSAN babble speech files
    musan_speech_wavs = glob.glob(os.path.join(musan_speech_dir, "**", "*.wav"), recursive=True)
    for w in musan_speech_wavs:
        noise_pool.append((w, "musan_babble"))

    print(f"Total Combined Noise Pool: {len(noise_pool)} files (ESC-50 + MUSAN noise/babble)")
    assert len(noise_pool) > 0, "No noise files found!"
    random.shuffle(noise_pool)

    # 3. Parameters
    snrs = [-5.0, 0.0, 5.0, 10.0, 15.0]
    num_train = 680
    num_val = 120

    print(f"\nGenerating {num_train} training pairs into data_full/train_* ...")
    train_res = generate_pairs_set(
        train_speech_pool, noise_pool,
        out_noisy_dir=r"d:\SIH 2026\DTLN\data_full\train_mix",
        out_clean_dir=r"d:\SIH 2026\DTLN\data_full\train_speech",
        snrs=snrs, num_pairs=num_train, prefix="train"
    )

    print(f"\nGenerating {num_val} validation pairs into data_full/val_* ...")
    val_res = generate_pairs_set(
        val_speech_pool, noise_pool,
        out_noisy_dir=r"d:\SIH 2026\DTLN\data_full\val_mix",
        out_clean_dir=r"d:\SIH 2026\DTLN\data_full\val_speech",
        snrs=snrs, num_pairs=num_val, prefix="val"
    )

    print(f"\nSuccessfully generated {len(train_res)} train pairs and {len(val_res)} val pairs (800 total pairs).")

if __name__ == '__main__':
    main()
