#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_matched120.py -- Re-evaluate 384-unit babble fine-tuned model on the EXACT 120 filenames
used in eval_results_384_run.csv.
Outputs results to data_full/eval_results_384_babble_run_matched120.csv.
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
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

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

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
NOISY_DIR    = os.path.join(SCRIPT_DIR, "data_full", "val_mix")
CLEAN_DIR    = os.path.join(SCRIPT_DIR, "data_full", "val_speech")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "data_full", "enhanced_384_babble_run_matched120")
CSV_BASELINE = os.path.join(SCRIPT_DIR, "data_full", "eval_results_384_run.csv")
CSV_OUT      = os.path.join(SCRIPT_DIR, "data_full", "eval_results_384_babble_run_matched120.csv")
MODEL1_PATH  = os.path.join(SCRIPT_DIR, "models_384_babble_run", "model_1.tflite")
MODEL2_PATH  = os.path.join(SCRIPT_DIR, "models_384_babble_run", "model_2.tflite")

NUM_UNITS   = 384
BLOCK_LEN   = 512
BLOCK_SHIFT = 128
STATE_SHAPE = (1, 1, NUM_UNITS, 2)


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


def process_file_item(item):
    noisy_path, clean_path, enh_path, model1_path, model2_path = item
    fname = os.path.basename(noisy_path)
    
    snr_match = re.search(r"snr(-?\d+)dB", fname)
    snr_cond = float(snr_match.group(1)) if snr_match else 0.0
    noise_cat = detect_noise_category(fname)

    # Load audio
    clean_audio, fs = sf.read(clean_path)
    noisy_audio, _  = sf.read(noisy_path)

    if clean_audio.ndim > 1: clean_audio = np.mean(clean_audio, axis=1)
    if noisy_audio.ndim > 1: noisy_audio = np.mean(noisy_audio, axis=1)

    # Initialize TFLite interpreter for this worker process
    interp1 = tflite.Interpreter(model_path=model1_path)
    interp1.allocate_tensors()
    interp2 = tflite.Interpreter(model_path=model2_path)
    interp2.allocate_tensors()

    in1  = interp1.get_input_details()
    out1 = interp1.get_output_details()
    in2  = interp2.get_input_details()
    out2 = interp2.get_output_details()

    for d in in1:
        sh = d["shape"]
        if len(sh) == 4 and sh[2] == NUM_UNITS: idx_s1_in = d["index"]
        elif len(sh) == 3 and sh[2] == 257: idx_mag_in = d["index"]

    for d in out1:
        sh = d["shape"]
        if len(sh) == 4 and sh[2] == NUM_UNITS: idx_s1_out = d["index"]
        elif len(sh) == 3 and sh[2] == 257: idx_mask_out = d["index"]

    for d in in2:
        sh = d["shape"]
        if len(sh) == 4 and sh[2] == NUM_UNITS: idx_s2_in = d["index"]
        elif len(sh) == 3 and sh[2] == BLOCK_LEN: idx_frame_in = d["index"]

    for d in out2:
        sh = d["shape"]
        if len(sh) == 4 and sh[2] == NUM_UNITS: idx_s2_out = d["index"]
        elif len(sh) == 3 and sh[2] == BLOCK_LEN: idx_decoded_out = d["index"]

    s1 = np.zeros(STATE_SHAPE, dtype=np.float32)
    s2 = np.zeros(STATE_SHAPE, dtype=np.float32)

    in_buf   = np.zeros(BLOCK_LEN, dtype=np.float32)
    out_buf  = np.zeros(BLOCK_LEN, dtype=np.float32)
    out_file = np.zeros(len(noisy_audio), dtype=np.float32)

    n_blocks = (len(noisy_audio) - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT
    block_latencies = []

    for idx in range(n_blocks):
        t_start = time.perf_counter()

        in_buf[:-BLOCK_SHIFT] = in_buf[BLOCK_SHIFT:]
        in_buf[-BLOCK_SHIFT:] = noisy_audio[idx * BLOCK_SHIFT: idx * BLOCK_SHIFT + BLOCK_SHIFT]

        fft      = np.fft.rfft(in_buf)
        in_mag   = np.abs(fft).reshape(1, 1, -1).astype(np.float32)
        in_phase = np.angle(fft)

        interp1.set_tensor(idx_s1_in,  s1)
        interp1.set_tensor(idx_mag_in, in_mag)
        interp1.invoke()
        mask = interp1.get_tensor(idx_mask_out)
        s1   = interp1.get_tensor(idx_s1_out)

        est_complex = in_mag * mask * np.exp(1j * in_phase)
        est_frame   = np.fft.irfft(est_complex).reshape(1, 1, -1).astype(np.float32)

        interp2.set_tensor(idx_s2_in,    s2)
        interp2.set_tensor(idx_frame_in, est_frame)
        interp2.invoke()
        out_block = interp2.get_tensor(idx_decoded_out)
        s2        = interp2.get_tensor(idx_s2_out)

        out_buf[:-BLOCK_SHIFT]  = out_buf[BLOCK_SHIFT:]
        out_buf[-BLOCK_SHIFT:]  = 0.0
        out_buf += np.squeeze(out_block)
        out_file[idx * BLOCK_SHIFT: idx * BLOCK_SHIFT + BLOCK_SHIFT] = out_buf[:BLOCK_SHIFT]

        block_latencies.append(time.perf_counter() - t_start)

    # Save output WAV
    sf.write(enh_path, out_file, fs)

    # Calculate metrics matching DTLN 384 alignment (384 samples lookahead/shift)
    pred_aligned  = out_file[384:]
    clean_aligned = clean_audio[:len(pred_aligned)]
    noisy_aligned = noisy_audio[:len(pred_aligned)]

    n_samples = min(len(clean_aligned), len(pred_aligned), len(noisy_aligned))
    c, t_clean, t_noisy = clean_aligned[:n_samples], pred_aligned[:n_samples], noisy_aligned[:n_samples]

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

    audio_dur  = len(noisy_audio) / fs
    total_inf  = sum(block_latencies)
    rtf        = total_inf / audio_dur
    avg_lat_ms = (np.mean(block_latencies) * 1000.0) if block_latencies else 0.0

    return {
        "filename":            fname,
        "noise_type":          noise_cat,
        "input_snr_cond":      snr_cond,
        "duration_sec":        round(audio_dur, 3),
        "num_blocks":          len(block_latencies),
        "pure_inf_time_sec":   round(total_inf,  4),
        "avg_block_latency_ms":round(avg_lat_ms, 3),
        "rtf":                 round(rtf,         4),
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
    print("=== Extracting filenames from val_mix folder ===")
    target_filenames = [os.path.basename(f) for f in glob.glob(os.path.join(NOISY_DIR, "*.wav"))]

    print(f"Extracted {len(target_filenames)} exact baseline filenames.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = []
    for fname in target_filenames:
        noisy_path = os.path.join(NOISY_DIR, fname)
        clean_path = os.path.join(CLEAN_DIR, fname)
        enh_path   = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(noisy_path), f"Missing noisy file: {noisy_path}"
        assert os.path.exists(clean_path), f"Missing clean file: {clean_path}"
        tasks.append((noisy_path, clean_path, enh_path, MODEL1_PATH, MODEL2_PATH))

    print("Executing evaluation on the exact 120 matched files across CPU cores ...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        results = list(executor.map(process_file_item, tasks))

    elapsed = time.time() - t0
    print(f"Evaluation complete in {elapsed:.2f} seconds!")

    results.sort(key=lambda x: x["filename"])

    fieldnames = [
        "filename", "noise_type", "input_snr_cond", "duration_sec", "num_blocks",
        "pure_inf_time_sec", "avg_block_latency_ms", "rtf",
        "si_snr_before", "si_snr_after", "si_snr_gain",
        "stoi_before", "stoi_after", "stoi_gain",
        "pesq_before", "pesq_after", "pesq_gain",
    ]
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"Matched 120 CSV written to: {CSV_OUT}\n")

    def safe_mean(rows, key):
        vals = [r[key] for r in rows if r[key] != "N/A" and not np.isnan(float(r[key]))]
        return np.mean([float(v) for v in vals]) if vals else float("nan")

    # 1. Noise-Type Breakdown
    print("="*115)
    print("DTLN 384-Unit (Babble Run) -- Matched 120 Files Breakdown by Noise Category")
    print("="*115)
    hdr = (f"{'Noise Type':<15} | {'Count':<5} | {'SI-SNR Bef':<12} | {'SI-SNR Aft':<12} | "
           f"{'SI-SNR Gain':<12} | {'STOI Aft':<10} | {'PESQ Aft':<9}")
    print(hdr)
    print("-"*115)

    noise_types = ["esc50", "musan_noise", "musan_babble"]
    for nt in noise_types:
        sub = [r for r in results if r["noise_type"] == nt]
        if sub:
            print(f"{nt:<15} | {len(sub):<5} | "
                  f"{safe_mean(sub,'si_snr_before'):<12.2f} | {safe_mean(sub,'si_snr_after'):<12.2f} | "
                  f"{safe_mean(sub,'si_snr_gain'):<12.2f} | {safe_mean(sub,'stoi_after'):<10.4f} | "
                  f"{safe_mean(sub,'pesq_after'):<9.3f}")

    # 2. SNR Band Breakdown
    print("\n" + "="*115)
    print("DTLN 384-Unit (Babble Run) -- Matched 120 Files Breakdown by Input SNR Level")
    print("="*115)
    print(hdr.replace("Noise Type", "SNR Level "))
    print("-"*115)

    target_snrs = sorted(list(set(r["input_snr_cond"] for r in results)))
    for snr_val in target_snrs:
        sub = [r for r in results if r["input_snr_cond"] == snr_val]
        print(f"{f'{snr_val:.1f} dB':<15} | {len(sub):<5} | "
              f"{safe_mean(sub,'si_snr_before'):<12.2f} | {safe_mean(sub,'si_snr_after'):<12.2f} | "
              f"{safe_mean(sub,'si_snr_gain'):<12.2f} | {safe_mean(sub,'stoi_after'):<10.4f} | "
              f"{safe_mean(sub,'pesq_after'):<9.3f}")

    print("-"*115)
    ov_lat = np.mean([r["avg_block_latency_ms"] for r in results])
    ov_rtf = np.mean([r["rtf"] for r in results])
    print(f"{'OVERALL':<15} | {len(results):<5} | "
          f"{safe_mean(results,'si_snr_before'):<12.2f} | {safe_mean(results,'si_snr_after'):<12.2f} | "
          f"{safe_mean(results,'si_snr_gain'):<12.2f} | {safe_mean(results,'stoi_after'):<10.4f} | "
          f"{safe_mean(results,'pesq_after'):<9.3f}")
    print("="*115)
    print(f"Average Block Latency: {ov_lat:.3f} ms | RTF: {ov_rtf:.4f}")


if __name__ == "__main__":
    main()
