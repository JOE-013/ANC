#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix_pairs.py - Generator for DTLN speech noise suppression training pairs.
Mixes clean speech audio with noise clips at specified SNR levels.
Outputs matching clean and noisy WAV files (16kHz mono, >=15s duration).
"""

import argparse
import csv
import glob
import os
import random
import numpy as np
import scipy.signal
import soundfile as sf

def compute_rms(audio):
    """Compute Root Mean Square (RMS) energy of an audio signal."""
    return np.sqrt(np.mean(np.square(audio)) + 1e-12)

def load_and_resample(file_path, target_fs=16000):
    """Load audio file, convert to mono, and resample to target_fs."""
    data, fs = sf.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if fs != target_fs:
        num_target_samples = int(round(len(data) * target_fs / fs))
        data = scipy.signal.resample(data, num_target_samples)
    return data.astype(np.float32), target_fs

def ensure_duration(audio, min_samples):
    """Loop audio if it is shorter than min_samples."""
    if len(audio) >= min_samples:
        return audio[:min_samples]
    repeats = int(np.ceil(min_samples / len(audio)))
    audio_extended = np.tile(audio, repeats)
    return audio_extended[:min_samples]

def mix_audio(speech, noise, target_snr_db):
    """
    Mix speech and noise at target SNR in dB.
    Returns (noisy_speech, scaled_clean_speech, measured_snr_db).
    """
    rms_speech = compute_rms(speech)
    rms_noise = compute_rms(noise)

    if rms_noise == 0 or rms_speech == 0:
        raise ValueError("Audio clip has zero RMS energy.")

    # Calculate required noise RMS
    target_noise_rms = rms_speech / (10 ** (target_snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / rms_noise)

    noisy = speech + scaled_noise

    # Peak normalization if clipping occurs (keep max peak <= 0.99)
    max_peak = np.max(np.abs(noisy))
    if max_peak > 0.99:
        norm_factor = 0.99 / max_peak
        noisy = noisy * norm_factor
        speech = speech * norm_factor
        scaled_noise = scaled_noise * norm_factor

    # Measure actual resulting SNR
    noise_part = noisy - speech
    measured_snr_db = 20.0 * np.log10(compute_rms(speech) / compute_rms(noise_part))

    return noisy.astype(np.float32), speech.astype(np.float32), measured_snr_db


def main():
    parser = argparse.ArgumentParser(description="DTLN Training Data Pair Generator")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory with clean speech files (.flac or .wav)")
    parser.add_argument("--noise_dir", type=str, required=True, help="Directory with noise audio files (.wav)")
    parser.add_argument("--out_clean_dir", type=str, required=True, help="Output directory for clean speech pairs")
    parser.add_argument("--out_noisy_dir", type=str, required=True, help="Output directory for noisy mixed pairs")
    parser.add_argument("--esc50_csv", type=str, default=None, help="Path to ESC-50 csv metadata for filtering categories")
    parser.add_argument("--categories", nargs="+", default=["gunshot", "siren", "engine", "helicopter", "chainsaw", "dog"], help="Target ESC-50 noise categories")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-5.0, 5.0, 15.0], help="Target SNR levels in dB")
    parser.add_argument("--target_fs", type=int, default=16000, help="Target sample rate (default 16000)")
    parser.add_argument("--target_duration", type=float, default=15.0, help="Minimum clip duration in seconds (default 15.0)")
    parser.add_argument("--num_pairs", type=int, default=15, help="Number of distinct audio pairs per SNR level")

    args = parser.parse_args()

    os.makedirs(args.out_clean_dir, exist_ok=True)
    os.makedirs(args.out_noisy_dir, exist_ok=True)

    # 1. Gather clean speech files
    speech_files = glob.glob(os.path.join(args.clean_dir, "**", "*.flac"), recursive=True) + \
                   glob.glob(os.path.join(args.clean_dir, "**", "*.wav"), recursive=True)

    if not speech_files:
        raise FileNotFoundError(f"No .flac or .wav clean speech files found in {args.clean_dir}")

    random.seed(42)
    random.shuffle(speech_files)

    # 2. Gather noise files
    noise_files = []
    if args.esc50_csv and os.path.exists(args.esc50_csv):
        with open(args.esc50_csv, 'r') as f:
            reader = csv.DictReader(f)
            selected = [row for row in reader if row['category'] in args.categories]
            for s in selected:
                full_p = os.path.join(args.noise_dir, s['filename'])
                if os.path.exists(full_p):
                    noise_files.append((full_p, s['category']))
        print(f"Selected {len(noise_files)} noise files matching categories: {args.categories}")
    
    if not noise_files:
        raw_noise_wavs = glob.glob(os.path.join(args.noise_dir, "**", "*.wav"), recursive=True)
        noise_files = [(p, "general") for p in raw_noise_wavs]

    if not noise_files:
        raise FileNotFoundError(f"No noise .wav files found in {args.noise_dir}")

    random.shuffle(noise_files)

    min_samples = int(args.target_fs * args.target_duration)
    generated_count = 0
    results_summary = []

    print(f"\nGenerating dataset pairs across SNRs: {args.snrs} dB ...")

    pair_idx = 0
    for snr in args.snrs:
        for i in range(args.num_pairs):
            sp_file = speech_files[(pair_idx + i) % len(speech_files)]
            ns_file, ns_cat = noise_files[(pair_idx + i) % len(noise_files)]

            # Load clean speech & resample
            speech_audio, _ = load_and_resample(sp_file, target_fs=args.target_fs)
            speech_audio = ensure_duration(speech_audio, min_samples)

            # Load noise & resample
            noise_audio, _ = load_and_resample(ns_file, target_fs=args.target_fs)
            noise_audio = ensure_duration(noise_audio, len(speech_audio))

            # Mix
            noisy_audio, clean_audio, measured_snr = mix_audio(speech_audio, noise_audio, snr)

            # Identical filename in both folders for DTLN generator!
            filename = f"pair_{pair_idx+i:04d}_snr{int(snr)}dB_{ns_cat}.wav"
            out_noisy_path = os.path.join(args.out_noisy_dir, filename)
            out_clean_path = os.path.join(args.out_clean_dir, filename)

            sf.write(out_noisy_path, noisy_audio, args.target_fs)
            sf.write(out_clean_path, clean_audio, args.target_fs)

            generated_count += 1
            results_summary.append({
                'filename': filename,
                'target_snr': snr,
                'measured_snr': measured_snr,
                'category': ns_cat,
                'duration_sec': len(clean_audio) / args.target_fs
            })
        pair_idx += args.num_pairs

    print(f"Successfully generated {generated_count} pairs in '{args.out_noisy_dir}' and '{args.out_clean_dir}'.")
    return results_summary

if __name__ == '__main__':
    main()
