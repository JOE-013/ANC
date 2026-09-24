#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_check_tflite_384_babble.py -- Sanity check for the 384-unit babble fine-tuned DTLN TFLite submodels.
Checks model loading, tensor shapes, and runs inference on 10 validation files.
Uses models_384_babble_run/model_1.tflite and model_2.tflite.
"""

import os
import glob
import time
import numpy as np
import soundfile as sf

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter

# ── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL1_PATH  = os.path.join(SCRIPT_DIR, "models_384_babble_run", "model_1.tflite")
MODEL2_PATH  = os.path.join(SCRIPT_DIR, "models_384_babble_run", "model_2.tflite")
NOISY_DIR    = os.path.join(SCRIPT_DIR, "data_full", "val_mix")
CLEAN_DIR    = os.path.join(SCRIPT_DIR, "data_full", "val_speech")

NUM_UNITS   = 384
BLOCK_LEN   = 512
BLOCK_SHIFT = 128
STATE_SHAPE = (1, 1, NUM_UNITS, 2)
FRAME_SHAPE_1 = 257
FRAME_SHAPE_2 = BLOCK_LEN


def compute_si_snr(estimate, target):
    target   = target   - np.mean(target)
    estimate = estimate - np.mean(estimate)
    n = min(len(estimate), len(target))
    estimate, target = estimate[:n], target[:n]
    dot  = np.dot(estimate, target)
    t_e  = np.dot(target, target) + 1e-7
    s_t  = (dot / t_e) * target
    e_n  = estimate - s_t
    return float(10.0 * np.log10(np.sum(s_t**2) / (np.sum(e_n**2) + 1e-7)))


class Engine384Babble:
    def __init__(self, model1_path, model2_path):
        if not os.path.exists(model1_path):
            raise FileNotFoundError(f"Model 1 not found: {model1_path}")
        if not os.path.exists(model2_path):
            raise FileNotFoundError(f"Model 2 not found: {model2_path}")

        self.interp1 = Interpreter(model_path=model1_path)
        self.interp1.allocate_tensors()
        self.interp2 = Interpreter(model_path=model2_path)
        self.interp2.allocate_tensors()

        in1  = self.interp1.get_input_details()
        out1 = self.interp1.get_output_details()
        in2  = self.interp2.get_input_details()
        out2 = self.interp2.get_output_details()

        for d in in1:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s1_in  = d["index"]
            elif len(sh) == 3 and sh[2] == FRAME_SHAPE_1:
                self.idx_mag_in = d["index"]

        for d in out1:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s1_out  = d["index"]
            elif len(sh) == 3 and sh[2] == FRAME_SHAPE_1:
                self.idx_mask_out = d["index"]

        for d in in2:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s2_in   = d["index"]
            elif len(sh) == 3 and sh[2] == FRAME_SHAPE_2:
                self.idx_frame_in = d["index"]

        for d in out2:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s2_out    = d["index"]
            elif len(sh) == 3 and sh[2] == FRAME_SHAPE_2:
                self.idx_decoded_out = d["index"]

    def process_audio(self, audio_data):
        s1 = np.zeros(STATE_SHAPE, dtype=np.float32)
        s2 = np.zeros(STATE_SHAPE, dtype=np.float32)

        in_buf  = np.zeros(BLOCK_LEN, dtype=np.float32)
        out_buf = np.zeros(BLOCK_LEN, dtype=np.float32)
        out_file = np.zeros(len(audio_data), dtype=np.float32)

        n_blocks = (len(audio_data) - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT
        latencies = []

        for idx in range(n_blocks):
            t0 = time.perf_counter()

            in_buf[:-BLOCK_SHIFT] = in_buf[BLOCK_SHIFT:]
            in_buf[-BLOCK_SHIFT:] = audio_data[idx * BLOCK_SHIFT: idx * BLOCK_SHIFT + BLOCK_SHIFT]

            fft      = np.fft.rfft(in_buf)
            in_mag   = np.abs(fft).reshape(1, 1, -1).astype(np.float32)
            in_phase = np.angle(fft)

            self.interp1.set_tensor(self.idx_s1_in, s1)
            self.interp1.set_tensor(self.idx_mag_in, in_mag)
            self.interp1.invoke()
            mask = self.interp1.get_tensor(self.idx_mask_out)
            s1   = self.interp1.get_tensor(self.idx_s1_out)

            est_complex = in_mag * mask * np.exp(1j * in_phase)
            est_frame   = np.fft.irfft(est_complex).reshape(1, 1, -1).astype(np.float32)

            self.interp2.set_tensor(self.idx_s2_in, s2)
            self.interp2.set_tensor(self.idx_frame_in, est_frame)
            self.interp2.invoke()
            out_block = self.interp2.get_tensor(self.idx_decoded_out)
            s2        = self.interp2.get_tensor(self.idx_s2_out)

            out_buf[:-BLOCK_SHIFT]  = out_buf[BLOCK_SHIFT:]
            out_buf[-BLOCK_SHIFT:]  = 0.0
            out_buf += np.squeeze(out_block)

            start = idx * BLOCK_SHIFT
            end   = min(start + BLOCK_SHIFT, len(out_file))
            out_file[start:end] = out_buf[:end - start]

            latencies.append((time.perf_counter() - t0) * 1000.0)

        return out_file, np.mean(latencies)


def main():
    print("=== DTLN 384-Unit Babble Model TFLite Sanity Check ===")
    print(f"  Stage 1 Model : {MODEL1_PATH}")
    print(f"  Stage 2 Model : {MODEL2_PATH}\n")

    engine = Engine384Babble(MODEL1_PATH, MODEL2_PATH)
    print("TFLite Submodels Loaded & Allocated OK!\n")

    noisy_files = sorted(glob.glob(os.path.join(NOISY_DIR, "*.wav")))[:10]
    if not noisy_files:
        print("No validation files found for testing.")
        return

    print(f"{'Filename':<45} | {'SI-SNR Before':>14} | {'SI-SNR After':>14} | {'Latency/Block':>15}")
    print("-" * 100)

    snr_befores, snr_afters, block_lats = [], [], []

    for noisy_path in noisy_files:
        fname      = os.path.basename(noisy_path)
        clean_path = os.path.join(CLEAN_DIR, fname)

        noisy_audio, fs = sf.read(noisy_path)
        clean_audio, _  = sf.read(clean_path)

        if noisy_audio.ndim > 1: noisy_audio = np.mean(noisy_audio, axis=1)
        if clean_audio.ndim > 1: clean_audio = np.mean(clean_audio, axis=1)

        tflite_out, avg_lat_ms = engine.process_audio(noisy_audio)

        tflite_trimmed = tflite_out[384:]
        clean_trimmed  = clean_audio[:len(tflite_trimmed)]
        noisy_trimmed  = noisy_audio[:len(tflite_trimmed)]

        n = min(len(tflite_trimmed), len(clean_trimmed), len(noisy_trimmed))
        
        snr_b = compute_si_snr(noisy_trimmed[:n], clean_trimmed[:n])
        snr_a = compute_si_snr(tflite_trimmed[:n], clean_trimmed[:n])

        snr_befores.append(snr_b)
        snr_afters.append(snr_a)
        block_lats.append(avg_lat_ms)

        print(f"{fname:<45} | {snr_b:>12.2f} dB | {snr_a:>12.2f} dB | {avg_lat_ms:>12.3f} ms")

    print("-" * 100)
    print(f"{'AVERAGE':<45} | {np.mean(snr_befores):>12.2f} dB | {np.mean(snr_afters):>12.2f} dB | {np.mean(block_lats):>12.3f} ms")
    print(f"\nReal-Time Budget: 8.000 ms per block | Measured Average Latency: {np.mean(block_lats):.3f} ms")
    if np.mean(block_lats) < 8.0:
        print("STATUS: PASSED (Real-Time Headroom Confirmed!)")
    else:
        print("STATUS: WARNING (Exceeds 8.0ms budget)")


if __name__ == "__main__":
    main()
