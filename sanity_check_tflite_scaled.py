#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_check_tflite_scaled.py -- Sanity check for the 256-unit DTLN TFLite submodels.
Compares TFLite output against the .h5 reference model on 10 validation files.
Healthy aligned SI-SNR should be ~35-40+ dB (previous 64-unit result was 36.62 dB).
Flags clearly if the result is notably different.
Uses models_scaled_run/model_1.tflite and model_2.tflite.
"""

import os
import glob
import time
import numpy as np
import librosa
from DTLN_model import DTLN_model

# Fallback import for TFLite interpreter
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
MODEL1_PATH  = os.path.join(SCRIPT_DIR, "models_scaled_run", "model_1.tflite")
MODEL2_PATH  = os.path.join(SCRIPT_DIR, "models_scaled_run", "model_2.tflite")
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "models_full_run", "full_run_256.weights.h5")
NOISY_DIR    = os.path.join(SCRIPT_DIR, "data_full", "val_mix")
CLEAN_DIR    = os.path.join(SCRIPT_DIR, "data_full", "val_speech")

NUM_UNITS   = 256
NUM_LAYER   = 1
PREV_RESULT = 36.62   # Reference aligned SI-SNR from 64-unit model sanity check
WARN_DELTA  = 5.0     # dB drop below PREV_RESULT that triggers a warning

BLOCK_LEN   = 512
BLOCK_SHIFT = 128
STATE_SHAPE = (1, 1, NUM_UNITS, 2)   # (batch, seq, units, h+c)
FRAME_SHAPE_1 = 257                   # Stage 1: FFT magnitude bins
FRAME_SHAPE_2 = BLOCK_LEN            # Stage 2: time-domain frame width


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


class ScaledTFLiteEngine:
    """Dynamic-index TFLite engine for 256-unit DTLN. Detects tensor indices by shape."""

    def __init__(self, model1_path, model2_path):
        self.interp1 = Interpreter(model_path=model1_path)
        self.interp1.allocate_tensors()
        self.interp2 = Interpreter(model_path=model2_path)
        self.interp2.allocate_tensors()

        in1  = self.interp1.get_input_details()
        out1 = self.interp1.get_output_details()
        in2  = self.interp2.get_input_details()
        out2 = self.interp2.get_output_details()

        # Stage 1 inputs: state (1,1,256,2) and magnitude (1,1,257)
        for d in in1:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s1_in  = d["index"]
            elif len(sh) == 3 and sh[2] == FRAME_SHAPE_1:
                self.idx_mag_in = d["index"]

        # Stage 1 outputs: updated state and mask (1,1,257)
        for d in out1:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s1_out  = d["index"]
            elif len(sh) == 3 and sh[2] == FRAME_SHAPE_1:
                self.idx_mask_out = d["index"]

        # Stage 2 inputs: state (1,1,256,2) and estimated frame (1,1,512)
        for d in in2:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s2_in   = d["index"]
            elif len(sh) == 3 and sh[2] == FRAME_SHAPE_2:
                self.idx_frame_in = d["index"]

        # Stage 2 outputs: updated state and decoded frame (1,1,512)
        for d in out2:
            sh = d["shape"]
            if len(sh) == 4 and sh[2] == NUM_UNITS:
                self.idx_s2_out    = d["index"]
            elif len(sh) == 3 and sh[2] == FRAME_SHAPE_2:
                self.idx_decoded_out = d["index"]

        # Verify all indices were found
        required = ["idx_s1_in", "idx_mag_in", "idx_s1_out", "idx_mask_out",
                    "idx_s2_in", "idx_frame_in", "idx_s2_out", "idx_decoded_out"]
        for attr in required:
            if not hasattr(self, attr):
                raise RuntimeError(
                    f"Could not locate tensor index for '{attr}'. "
                    f"Check numUnits={NUM_UNITS} matches exported model."
                )

    def process_audio(self, audio_data):
        s1 = np.zeros(STATE_SHAPE, dtype=np.float32)
        s2 = np.zeros(STATE_SHAPE, dtype=np.float32)

        in_buf  = np.zeros(BLOCK_LEN, dtype=np.float32)
        out_buf = np.zeros(BLOCK_LEN, dtype=np.float32)
        out_file = np.zeros(len(audio_data), dtype=np.float32)

        n_blocks = (len(audio_data) - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT

        for idx in range(n_blocks):
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

        return out_file


def h5_inference(model_obj, audio_data):
    """Run the .h5 model using the same zero-pad trick as evaluate_256_run.py."""
    pad       = np.zeros(384, dtype=np.float32)
    padded    = np.concatenate([pad, audio_data, pad])
    pred      = model_obj.model.predict_on_batch(
                    np.expand_dims(padded, 0).astype(np.float32))
    return np.squeeze(pred)[384:384 + len(audio_data)]


def main():
    print("=== DTLN Scaled (256-unit) TFLite Sanity Check ===")
    print(f"  TFLite model 1 : {MODEL1_PATH}")
    print(f"  TFLite model 2 : {MODEL2_PATH}")
    print(f"  H5 reference   : {WEIGHTS_PATH}")
    print()

    # --- Load TFLite engine ---
    print("Loading TFLite models ...")
    engine = ScaledTFLiteEngine(MODEL1_PATH, MODEL2_PATH)
    print("TFLite models loaded OK.")

    # --- Load H5 reference model ---
    print("Loading H5 reference model for comparison ...")
    ref = DTLN_model()
    ref.numUnits = NUM_UNITS
    ref.numLayer = NUM_LAYER
    ref.build_DTLN_model()
    ref.model.load_weights(WEIGHTS_PATH)
    print("H5 model loaded OK.\n")

    # Use first 10 noisy validation files
    noisy_files = sorted(glob.glob(os.path.join(NOISY_DIR, "*.wav")))[:10]
    if not noisy_files:
        raise RuntimeError(f"No .wav files found in {NOISY_DIR}")
    print(f"Testing on {len(noisy_files)} files.\n")

    results_tflite_vs_h5 = []
    results_tflite_vs_clean = []

    print(f"{'Filename':<42} | {'TFLite vs H5 SI-SNR':>20} | {'TFLite vs Clean SI-SNR':>22} | {'Time (s)':>9}")
    print("-" * 100)

    for noisy_path in noisy_files:
        fname      = os.path.basename(noisy_path)
        clean_path = os.path.join(CLEAN_DIR, fname)

        noisy_audio, _ = librosa.load(noisy_path, sr=16000, mono=True)
        clean_audio, _ = librosa.load(clean_path, sr=16000, mono=True)

        # TFLite inference
        t0 = time.perf_counter()
        tflite_out = engine.process_audio(noisy_audio)
        elapsed = time.perf_counter() - t0

        # H5 reference inference
        h5_out = h5_inference(ref, noisy_audio)

        # Trim startup delay (3 blocks * 128 = 384 samples)
        tflite_trimmed = tflite_out[384:]
        h5_trimmed     = h5_out      # h5_inference already handles alignment

        n = min(len(tflite_trimmed), len(h5_trimmed), len(clean_audio))
        tflite_trimmed = tflite_trimmed[:n]
        h5_trimmed     = h5_trimmed[:n]
        clean_trimmed  = clean_audio[:n]

        snr_vs_h5    = compute_si_snr(tflite_trimmed, h5_trimmed)
        snr_vs_clean = compute_si_snr(tflite_trimmed, clean_trimmed)

        results_tflite_vs_h5.append(snr_vs_h5)
        results_tflite_vs_clean.append(snr_vs_clean)

        print(f"{fname:<42} | {snr_vs_h5:>18.2f} dB | {snr_vs_clean:>20.2f} dB | {elapsed:>9.3f}")

    print("-" * 100)
    avg_vs_h5    = np.mean(results_tflite_vs_h5)
    avg_vs_clean = np.mean(results_tflite_vs_clean)
    print(f"{'AVERAGE':<42} | {avg_vs_h5:>18.2f} dB | {avg_vs_clean:>20.2f} dB |")

    print()
    print("=== RESULT ===")
    print(f"  TFLite vs H5 aligned SI-SNR  : {avg_vs_h5:.2f} dB   (previous 64-unit model: {PREV_RESULT:.2f} dB)")
    print(f"  TFLite vs Clean SI-SNR       : {avg_vs_clean:.2f} dB")

    if avg_vs_h5 < (PREV_RESULT - WARN_DELTA):
        print(f"\n  *** WARNING: TFLite/H5 agreement ({avg_vs_h5:.2f} dB) is more than "
              f"{WARN_DELTA:.0f} dB below the 64-unit reference ({PREV_RESULT:.2f} dB). "
              "Verify that the correct weights file was exported. ***")
    elif avg_vs_h5 >= 35.0:
        print(f"\n  Agreement is HEALTHY (>= 35 dB) -- TFLite model faithfully reproduces the H5 output.")
    else:
        print(f"\n  NOTE: Agreement is {avg_vs_h5:.2f} dB (< 35 dB). Acceptable but worth inspecting.")


if __name__ == "__main__":
    main()
