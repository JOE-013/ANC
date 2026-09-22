#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_tflite_scaled_run.py -- Quantitative evaluation of the 256-unit DTLN TFLite model.
Processes all 630 files in data_full/val_mix (7 SNR bands, 90 files each).
Uses models_scaled_run/model_1.tflite and model_2.tflite.
Includes high-resolution per-block timing (isolated from I/O and metrics).
Results saved to: data_full/eval_results_tflite_scaled_run.csv
Does NOT overwrite any eval_results_tflite_full_run.csv from the 64-unit run.
"""

import os
import glob
import csv
import re
import time
import numpy as np
import soundfile as sf
import librosa
from pystoi import stoi

# Support both tflite_runtime (Pi) and tensorflow.lite (PC)
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

# ── Config ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NOISY_DIR   = os.path.join(SCRIPT_DIR, "data_full", "val_mix")
CLEAN_DIR   = os.path.join(SCRIPT_DIR, "data_full", "val_speech")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "data_full", "enhanced_tflite_scaled_run")
CSV_OUT     = os.path.join(SCRIPT_DIR, "data_full", "eval_results_tflite_scaled_run.csv")
MODEL1_PATH = os.path.join(SCRIPT_DIR, "models_scaled_run", "model_1.tflite")
MODEL2_PATH = os.path.join(SCRIPT_DIR, "models_scaled_run", "model_2.tflite")

NUM_UNITS   = 256
BLOCK_LEN   = 512
BLOCK_SHIFT = 128
STATE_SHAPE = (1, 1, NUM_UNITS, 2)


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_si_snr(estimate, target):
    target   = target   - np.mean(target)
    estimate = estimate - np.mean(estimate)
    dot  = np.dot(estimate, target)
    t_e  = np.dot(target, target) + 1e-7
    s_t  = (dot / t_e) * target
    e_n  = estimate - s_t
    return float(10.0 * np.log10(np.sum(s_t**2) / (np.sum(e_n**2) + 1e-7)))


def compute_metrics(clean, test, fs=16000):
    n = min(len(clean), len(test))
    c, t = clean[:n], test[:n]
    si_snr_val = compute_si_snr(t, c)
    try:
        stoi_val = float(stoi(c, t, fs, extended=False))
    except Exception:
        stoi_val = float("nan")
    if HAS_PESQ:
        try:
            pesq_val = float(pesq(fs, c, t, "wb"))
        except Exception:
            pesq_val = float("nan")
    else:
        pesq_val = None
    return si_snr_val, stoi_val, pesq_val


# ── Inference Engine ───────────────────────────────────────────────────────
class ScaledTFLiteEvaluator:
    """DTLN 256-unit TFLite inference engine with per-block latency instrumentation.
    
    Key difference from the 64-unit engine: all tensor shape probes use
    NUM_UNITS=256 (state tensors) and 512 (time-domain frame width).
    """

    def __init__(self, model1_path, model2_path):
        self.interp1 = tflite.Interpreter(model_path=model1_path)
        self.interp1.allocate_tensors()
        self.interp2 = tflite.Interpreter(model_path=model2_path)
        self.interp2.allocate_tensors()

        in1  = self.interp1.get_input_details()
        out1 = self.interp1.get_output_details()
        in2  = self.interp2.get_input_details()
        out2 = self.interp2.get_output_details()

        # Stage 1: state=(1,1,256,2), magnitude=(1,1,257)
        for d in in1:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s1_in  = d["index"]
            elif len(sh) == 3 and sh[2] == 257:
                self.idx_mag_in = d["index"]

        for d in out1:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s1_out   = d["index"]
            elif len(sh) == 3 and sh[2] == 257:
                self.idx_mask_out = d["index"]

        # Stage 2: state=(1,1,256,2), frame=(1,1,512)
        for d in in2:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s2_in    = d["index"]
            elif len(sh) == 3 and sh[2] == BLOCK_LEN:
                self.idx_frame_in = d["index"]

        for d in out2:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s2_out      = d["index"]
            elif len(sh) == 3 and sh[2] == BLOCK_LEN:
                self.idx_decoded_out = d["index"]

        # Verify all required indices found
        required = ["idx_s1_in", "idx_mag_in", "idx_s1_out", "idx_mask_out",
                    "idx_s2_in", "idx_frame_in", "idx_s2_out", "idx_decoded_out"]
        for attr in required:
            if not hasattr(self, attr):
                raise RuntimeError(
                    f"Tensor index for '{attr}' not found. "
                    f"Verify models_scaled_run/ was exported with numUnits={NUM_UNITS}."
                )

    def process_audio(self, audio_data):
        """
        Frame-by-frame TFLite inference with per-block timing.
        Returns:
            out_file        (np.ndarray) : enhanced audio
            block_latencies (list[float]): per-block wall-clock time in seconds
            total_inf_time  (float)      : sum of block latencies (seconds)
        """
        s1 = np.zeros(STATE_SHAPE, dtype=np.float32)
        s2 = np.zeros(STATE_SHAPE, dtype=np.float32)

        in_buf   = np.zeros(BLOCK_LEN, dtype=np.float32)
        out_buf  = np.zeros(BLOCK_LEN, dtype=np.float32)
        out_file = np.zeros(len(audio_data), dtype=np.float32)

        n_blocks = (len(audio_data) - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT
        block_latencies = []

        for idx in range(n_blocks):
            t_start = time.perf_counter()

            # Shift + fill input buffer
            in_buf[:-BLOCK_SHIFT] = in_buf[BLOCK_SHIFT:]
            in_buf[-BLOCK_SHIFT:] = audio_data[idx * BLOCK_SHIFT: idx * BLOCK_SHIFT + BLOCK_SHIFT]

            # STFT
            fft      = np.fft.rfft(in_buf)
            in_mag   = np.abs(fft).reshape(1, 1, -1).astype(np.float32)
            in_phase = np.angle(fft)

            # Stage 1
            self.interp1.set_tensor(self.idx_s1_in,  s1)
            self.interp1.set_tensor(self.idx_mag_in, in_mag)
            self.interp1.invoke()
            mask = self.interp1.get_tensor(self.idx_mask_out)
            s1   = self.interp1.get_tensor(self.idx_s1_out)

            # iFFT
            est_complex = in_mag * mask * np.exp(1j * in_phase)
            est_frame   = np.fft.irfft(est_complex).reshape(1, 1, -1).astype(np.float32)

            # Stage 2
            self.interp2.set_tensor(self.idx_s2_in,    s2)
            self.interp2.set_tensor(self.idx_frame_in, est_frame)
            self.interp2.invoke()
            out_block = self.interp2.get_tensor(self.idx_decoded_out)
            s2        = self.interp2.get_tensor(self.idx_s2_out)

            # Overlap-add
            out_buf[:-BLOCK_SHIFT]  = out_buf[BLOCK_SHIFT:]
            out_buf[-BLOCK_SHIFT:]  = 0.0
            out_buf += np.squeeze(out_block)
            out_file[idx * BLOCK_SHIFT: idx * BLOCK_SHIFT + BLOCK_SHIFT] = out_buf[:BLOCK_SHIFT]

            block_latencies.append(time.perf_counter() - t_start)

        return out_file, block_latencies, sum(block_latencies)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=== DTLN TFLite Scaled (256-unit) Validation Evaluation with Latency ===")
    print(f"  Model 1   : {MODEL1_PATH}")
    print(f"  Model 2   : {MODEL2_PATH}")
    print(f"  CSV out   : {CSV_OUT}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading TFLite interpreters ...")
    engine = ScaledTFLiteEvaluator(MODEL1_PATH, MODEL2_PATH)
    print("Interpreters ready.\n")

    noisy_files = sorted(glob.glob(os.path.join(NOISY_DIR, "*.wav")))
    print(f"Found {len(noisy_files)} validation files in {NOISY_DIR}\n")

    results = []
    all_latencies_ms = []

    print("Running frame-by-frame TFLite inference + metrics ...")
    for idx, noisy_path in enumerate(noisy_files):
        fname      = os.path.basename(noisy_path)
        clean_path = os.path.join(CLEAN_DIR, fname)
        enh_path   = os.path.join(OUTPUT_DIR, fname)

        snr_match  = re.search(r"snr(-?\d+)dB", fname)
        snr_cond   = float(snr_match.group(1)) if snr_match else 0.0

        # Audio loading (NOT included in timing)
        clean_audio, fs = librosa.core.load(clean_path, sr=16000, mono=True)
        noisy_audio, _  = librosa.core.load(noisy_path, sr=16000, mono=True)

        # TFLite inference + timing
        pred_speech, latencies, total_inf = engine.process_audio(noisy_audio)

        audio_dur    = len(noisy_audio) / fs
        rtf          = total_inf / audio_dur
        avg_lat_ms   = (np.mean(latencies) * 1000.0) if latencies else 0.0
        all_latencies_ms.extend([l * 1000.0 for l in latencies])

        # Save enhanced WAV
        sf.write(enh_path, pred_speech, fs)

        # Align (compensate 384-sample startup delay)
        pred_aligned  = pred_speech[384:]
        clean_aligned = clean_audio[:len(pred_aligned)]
        noisy_aligned = noisy_audio[:len(pred_aligned)]

        si_snr_b, stoi_b, pesq_b = compute_metrics(clean_aligned, noisy_aligned, fs)
        si_snr_a, stoi_a, pesq_a = compute_metrics(clean_aligned, pred_aligned,  fs)

        results.append({
            "filename":            fname,
            "input_snr_cond":      snr_cond,
            "duration_sec":        round(audio_dur, 3),
            "num_blocks":          len(latencies),
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
        })

        if (idx + 1) % 90 == 0 or (idx + 1) == len(noisy_files):
            print(f"  {idx+1}/{len(noisy_files)} processed | "
                  f"Avg block latency: {avg_lat_ms:.3f} ms | RTF: {rtf:.4f}")

    # Write CSV
    fieldnames = [
        "filename", "input_snr_cond", "duration_sec", "num_blocks",
        "pure_inf_time_sec", "avg_block_latency_ms", "rtf",
        "si_snr_before", "si_snr_after", "si_snr_gain",
        "stoi_before", "stoi_after", "stoi_gain",
        "pesq_before", "pesq_after", "pesq_gain",
    ]
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nResults written to: {CSV_OUT}")

    # Per-SNR breakdown
    snr_levels = sorted(set(r["input_snr_cond"] for r in results))

    def safe_mean(rows, key):
        vals = [r[key] for r in rows if r[key] != "N/A"]
        return np.nanmean(vals) if vals else float("nan")

    print("\n" + "="*115)
    print("DTLN TFLite Scaled (256-unit) -- Validation Breakdown by Input SNR Level")
    print("="*115)
    hdr = (f"{'SNR':<10} | {'Count':<5} | {'SI-SNR Bef':<12} | {'SI-SNR Aft':<12} | "
           f"{'STOI Bef':<10} | {'STOI Aft':<10} | "
           f"{'PESQ Bef':<9} | {'PESQ Aft':<9} | {'Block Lat':>10} | {'RTF':>8}")
    print(hdr)
    print("-"*115)

    for snr_val in snr_levels:
        sub = [r for r in results if r["input_snr_cond"] == snr_val]
        print(f"{snr_val:<10.1f}dB | {len(sub):<5} | "
              f"{safe_mean(sub,'si_snr_before'):<12.2f} | {safe_mean(sub,'si_snr_after'):<12.2f} | "
              f"{safe_mean(sub,'stoi_before'):<10.4f} | {safe_mean(sub,'stoi_after'):<10.4f} | "
              f"{safe_mean(sub,'pesq_before'):<9.3f} | {safe_mean(sub,'pesq_after'):<9.3f} | "
              f"{safe_mean(sub,'avg_block_latency_ms'):>8.3f} ms | "
              f"{safe_mean(sub,'rtf'):>8.4f}")

    print("-"*115)
    ov_lat = np.mean([r["avg_block_latency_ms"] for r in results])
    ov_rtf = np.mean([r["rtf"] for r in results])
    print(f"{'OVERALL':<10}   | {len(results):<5} | "
          f"{safe_mean(results,'si_snr_before'):<12.2f} | {safe_mean(results,'si_snr_after'):<12.2f} | "
          f"{safe_mean(results,'stoi_before'):<10.4f} | {safe_mean(results,'stoi_after'):<10.4f} | "
          f"{safe_mean(results,'pesq_before'):<9.3f} | {safe_mean(results,'pesq_after'):<9.3f} | "
          f"{ov_lat:>8.3f} ms | {ov_rtf:>8.4f}")
    print("="*115)

    print("\n=== Latency & Real-Time Performance Summary ===")
    print(f"  Audio block hop size (block_shift) : {BLOCK_SHIFT} samples = "
          f"{1000.0 * BLOCK_SHIFT / 16000:.1f} ms @ 16 kHz")
    print(f"  Average per-block latency          : {ov_lat:.3f} ms")
    print(f"  Latency budget headroom            : {8.0 - ov_lat:.3f} ms  (budget: 8.0 ms)")
    print(f"  Overall Real-Time Factor (RTF)     : {ov_rtf:.4f}x  "
          f"({'faster' if ov_rtf < 1.0 else 'SLOWER'} than real-time)")


if __name__ == "__main__":
    main()
