#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_babble_oversampled_dataset.py - Generates 2,100 new clean-speech / babble-noise pairs
specifically using MUSAN speech/babble noise sources at -5, 0, 5, and 10 dB SNR.
Appends new pairs to data_full/train_mix and data_full/train_speech.
"""

import os
import glob
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

    max_peak = np.max(np.abs(noisy))
    if max_peak > 0.99:
        norm_factor = 0.99 / max_peak
        noisy = noisy * norm_factor
        speech = speech * norm_factor
        scaled_noise = scaled_noise * norm_factor

    noise_part = noisy - speech
    measured_snr_db = 20.0 * np.log10(compute_rms(speech) / compute_rms(noise_part))
    return noisy.astype(np.float32), speech.astype(np.float32), measured_snr_db

def main():
    print("=== DTLN Babble Noise Oversampling Generator (MUSAN Babble) ===")

    libri_dir = r"d:\SIH 2026\DTLN\datasets\librispeech_dev_clean"
    musan_speech_dir = r"d:\SIH 2026\DTLN\datasets\musan\musan\speech"

    out_noisy_dir = r"d:\SIH 2026\DTLN\data_full\train_mix"
    out_clean_dir = r"d:\SIH 2026\DTLN\data_full\train_speech"

    # Gather clean speech files (85% train split)
    speech_files = glob.glob(os.path.join(libri_dir, "**", "*.flac"), recursive=True)
    random.seed(42)
    random.shuffle(speech_files)
    split_idx = int(0.85 * len(speech_files))
    train_speech_pool = speech_files[:split_idx]

    # Gather MUSAN babble speech files
    babble_pool = glob.glob(os.path.join(musan_speech_dir, "**", "*.wav"), recursive=True)
    print(f"Train speech pool size: {len(train_speech_pool)} files")
    print(f"MUSAN babble pool size: {len(babble_pool)} files")

    random.seed(8888) # Distinct seed for non-duplicate pairings
    random.shuffle(babble_pool)
    random.shuffle(train_speech_pool)

    # Targets: -5dB: 800, 0dB: 800, 5dB: 250, 10dB: 250
    targets = [
        (-5.0, 800),
        (0.0,  800),
        (5.0,  250),
        (10.0, 250)
    ]

    target_fs = 16000
    target_duration = 15.0
    min_samples = int(target_fs * target_duration)

    start_idx = 5610 # Resume index after existing 5,610 pairs
    added_count = 0

    for snr, num_pairs in targets:
        print(f"Generating {num_pairs} additional babble pairs for SNR = {snr} dB...")
        for i in range(num_pairs):
            curr_idx = start_idx + added_count
            sp_file = train_speech_pool[(curr_idx * 7 + 11) % len(train_speech_pool)]
            ns_file = babble_pool[(curr_idx * 13 + 17) % len(babble_pool)]

            speech_audio, _ = load_and_resample(sp_file, target_fs=target_fs)
            speech_audio = ensure_duration(speech_audio, min_samples)

            noise_audio, _ = load_and_resample(ns_file, target_fs=target_fs)
            noise_audio = ensure_duration(noise_audio, len(speech_audio))

            noisy_audio, clean_audio, _ = mix_audio(speech_audio, noise_audio, snr)

            filename = f"pair_train_{curr_idx:04d}_snr{int(snr)}dB_musan_babble.wav"
            out_noisy_path = os.path.join(out_noisy_dir, filename)
            out_clean_path = os.path.join(out_clean_dir, filename)

            sf.write(out_noisy_path, noisy_audio, target_fs)
            sf.write(out_clean_path, clean_audio, target_fs)

            added_count += 1

    print(f"\nSUCCESS: Added {added_count} new babble pairs to training set.")
    print(f"Total training pairs now: {len(os.listdir(out_noisy_dir))}")

if __name__ == '__main__':
    main()
