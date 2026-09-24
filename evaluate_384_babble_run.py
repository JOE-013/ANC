#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_384_babble_run.py -- Quantitative evaluation of the babble-retrained 384-unit DTLN TFLite model.
Evaluates enhanced files in data_full/enhanced_384_babble_run against data_full/val_speech and data_full/val_mix.
Outputs raw results CSV to data_full/eval_results_384_babble_run.csv.
Provides both SNR-level and Noise-Type (esc50 / musan_noise / musan_babble) breakdowns.
"""

import os
import glob
import csv
import re
import time
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

# ── Config ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NOISY_DIR   = os.path.join(SCRIPT_DIR, "data_full", "val_mix")
CLEAN_DIR   = os.path.join(SCRIPT_DIR, "data_full", "val_speech")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "data_full", "enhanced_384_babble_run")
CSV_OUT     = os.path.join(SCRIPT_DIR, "data_full", "eval_results_384_babble_run.csv")

def compute_si_snr(estimate, target):
    target   = target   - np.mean(target)
    estimate = estimate - np.mean(estimate)
    dot  = np.dot(estimate, target)
    t_e  = np.dot(target, target) + 1e-7
    s_t  = (dot / t_e) * target
    e_n  = estimate - s_t
    return float(10.0 * np.log10(np.sum(s_t**2) / (np.sum(e_n**2) + 1e-7)))

def detect_noise_category(fname):
    fname_lower = fname.lower()
    if "babble" in fname_lower:
        return "musan_babble"
    elif "esc50" in fname_lower:
        return "esc50"
    elif "noise" in fname_lower or "musan" in fname_lower:
        return "musan_noise"
    return "other"

def process_single_file(item):
    enh_path, clean_path, noisy_path = item
    fname = os.path.basename(enh_path)
    
    snr_match = re.search(r"snr(-?\d+)dB", fname)
    snr_cond = float(snr_match.group(1)) if snr_match else 0.0
    noise_cat = detect_noise_category(fname)

    clean_audio, fs = sf.read(clean_path)
    noisy_audio, _  = sf.read(noisy_path)
    enh_audio, _    = sf.read(enh_path)

    if clean_audio.ndim > 1: clean_audio = np.mean(clean_audio, axis=1)
    if noisy_audio.ndim > 1: noisy_audio = np.mean(noisy_audio, axis=1)
    if enh_audio.ndim > 1:   enh_audio   = np.mean(enh_audio, axis=1)

    # Delay alignment matching DTLN (384 samples lookahead/shift)
    pred_aligned  = enh_audio[384:] if len(enh_audio) > 384 else enh_audio
    clean_aligned = clean_audio[:len(pred_aligned)]
    noisy_aligned = noisy_audio[:len(pred_aligned)]

    n_samples = min(len(clean_aligned), len(pred_aligned), len(noisy_aligned))
    c = clean_aligned[:n_samples]
    t_clean = pred_aligned[:n_samples]
    t_noisy = noisy_aligned[:n_samples]

    si_snr_b = compute_si_snr(t_noisy, c)
    si_snr_a = compute_si_snr(t_clean, c)

    if HAS_STOI:
        try: stoi_b = float(stoi(c, t_noisy, fs, extended=False))
        except Exception: stoi_b = float("nan")
        try: stoi_a = float(stoi(c, t_clean, fs, extended=False))
        except Exception: stoi_a = float("nan")
    else:
        stoi_b, stoi_a = float("nan"), float("nan")

    if HAS_PESQ:
        try: pesq_b = float(pesq(fs, c, t_noisy, "wb"))
        except Exception: pesq_b = float("nan")
        try: pesq_a = float(pesq(fs, c, t_clean, "wb"))
        except Exception: pesq_a = float("nan")
    else:
        pesq_b, pesq_a = None, None

    audio_dur = len(noisy_audio) / fs

    return {
        "filename":            fname,
        "noise_type":          noise_cat,
        "input_snr_cond":      snr_cond,
        "duration_sec":        round(audio_dur, 3),
        "si_snr_before":       si_snr_b,
        "si_snr_after":        si_snr_a,
        "si_snr_gain":         si_snr_a - si_snr_b,
        "stoi_before":         stoi_b,
        "stoi_after":          stoi_a,
        "stoi_gain":           stoi_a - stoi_b,
        "pesq_before":         pesq_b if pesq_b is not None else "N/A",
        "pesq_after":          pesq_a if pesq_a is not None else "N/A",
        "pesq_gain":           (pesq_a - pesq_b) if (pesq_a is not None and pesq_b is not None) else "N/A",
    }

def main():
    print("=== Evaluating 384-Unit Babble Fine-Tuned DTLN Model ===")
    enh_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))
    print(f"Found {len(enh_files)} enhanced audio files in {OUTPUT_DIR}")

    tasks = []
    for enh_path in enh_files:
        fname = os.path.basename(enh_path)
        clean_path = os.path.join(CLEAN_DIR, fname)
        noisy_path = os.path.join(NOISY_DIR, fname)
        if os.path.exists(clean_path) and os.path.exists(noisy_path):
            tasks.append((enh_path, clean_path, noisy_path))

    print(f"Processing metrics across {len(tasks)} validation files ...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        results = list(executor.map(process_single_file, tasks))

    elapsed = time.time() - t0
    print(f"Evaluation completed in {elapsed:.2f} seconds!")

    results.sort(key=lambda x: x["filename"])

    fieldnames = [
        "filename", "noise_type", "input_snr_cond", "duration_sec",
        "si_snr_before", "si_snr_after", "si_snr_gain",
        "stoi_before", "stoi_after", "stoi_gain",
        "pesq_before", "pesq_after", "pesq_gain",
    ]
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"CSV file written to: {CSV_OUT}\n")

    def safe_mean(rows, key):
        vals = [r[key] for r in rows if r[key] != "N/A" and not np.isnan(r[key])]
        return np.mean(vals) if vals else float("nan")

    # 1. Noise-Type Breakdown
    print("="*105)
    print(" DTLN 384-UNIT BABBLE-RUN EVALUATION REPORT (BY NOISE TYPE)")
    print("="*105)
    hdr = (f"{'Noise Type':<15} | {'Count':<5} | {'SI-SNR Bef':<11} | {'SI-SNR Aft':<11} | "
           f"{'SI-SNR Gain':<11} | {'STOI Aft':<9} | {'PESQ Aft':<9}")
    print(hdr)
    print("-"*105)

    noise_types = ["esc50", "musan_noise", "musan_babble"]
    for nt in noise_types:
        sub = [r for r in results if r["noise_type"] == nt]
        if sub:
            print(f"{nt:<15} | {len(sub):<5} | "
                  f"{safe_mean(sub,'si_snr_before'):<11.2f} | {safe_mean(sub,'si_snr_after'):<11.2f} | "
                  f"{safe_mean(sub,'si_snr_gain'):<11.2f} | {safe_mean(sub,'stoi_after'):<9.4f} | "
                  f"{safe_mean(sub,'pesq_after'):<9.3f}")

    # 2. SNR Band Breakdown
    print("\n" + "="*105)
    print(" DTLN 384-UNIT BABBLE-RUN EVALUATION REPORT (BY INPUT SNR LEVEL)")
    print("="*105)
    print(hdr.replace("Noise Type", "SNR Level "))
    print("-"*105)

    target_snrs = sorted(list(set(r["input_snr_cond"] for r in results)))
    for snr_val in target_snrs:
        sub = [r for r in results if r["input_snr_cond"] == snr_val]
        print(f"{f'{snr_val:.1f} dB':<15} | {len(sub):<5} | "
              f"{safe_mean(sub,'si_snr_before'):<11.2f} | {safe_mean(sub,'si_snr_after'):<11.2f} | "
              f"{safe_mean(sub,'si_snr_gain'):<11.2f} | {safe_mean(sub,'stoi_after'):<9.4f} | "
              f"{safe_mean(sub,'pesq_after'):<9.3f}")

    print("-"*105)
    print(f"{'OVERALL':<15} | {len(results):<5} | "
          f"{safe_mean(results,'si_snr_before'):<11.2f} | {safe_mean(results,'si_snr_after'):<11.2f} | "
          f"{safe_mean(results,'si_snr_gain'):<11.2f} | {safe_mean(results,'stoi_after'):<9.4f} | "
          f"{safe_mean(results,'pesq_after'):<9.3f}")
    print("="*105)

if __name__ == "__main__":
    main()
