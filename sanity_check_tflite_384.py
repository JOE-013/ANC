#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_check_tflite_384.py -- Sanity check for the 384-unit DTLN TFLite submodels.
Compares TFLite output against the .h5 reference model on 10 validation files.
Uses models_384_run/model_1.tflite, model_2.tflite, and 384_run.weights.h5.
"""

import os
import glob
import time
import numpy as np
import librosa
from DTLN_model import DTLN_model

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
MODEL1_PATH  = os.path.join(SCRIPT_DIR, "models_384_run", "model_1.tflite")
MODEL2_PATH  = os.path.join(SCRIPT_DIR, "models_384_run", "model_2.tflite")
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "models_384_run", "384_run.weights.h5")
NOISY_DIR    = os.path.join(SCRIPT_DIR, "data_full", "val_mix")
CLEAN_DIR    = os.path.join(SCRIPT_DIR, "data_full", "val_speech")

NUM_UNITS   = 384
NUM_LAYER   = 1
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


class Engine384:
    def __init__(self, model1_path, model2_path):
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
    pad    = np.zeros(384, dtype=np.float32)
    padded = np.concatenate([pad, audio_data, pad])
    pred   = model_obj.model.predict_on_batch(np.expand_dims(padded, 0).astype(np.float32))
    return np.squeeze(pred)[384:384 + len(audio_data)]


def main():
    print("=== DTLN 384-unit TFLite Sanity Check ===")
    print(f"  TFLite model 1 : {MODEL1_PATH}")
    print(f"  TFLite model 2 : {MODEL2_PATH}")
    print(f"  H5 reference   : {WEIGHTS_PATH}\n")

    engine = Engine384(MODEL1_PATH, MODEL2_PATH)

    ref = DTLN_model()
    ref.numUnits = NUM_UNITS
    ref.numLayer = NUM_LAYER
    ref.build_DTLN_model()
    ref.model.load_weights(WEIGHTS_PATH)
    print("Models loaded OK.\n")

    noisy_files = sorted(glob.glob(os.path.join(NOISY_DIR, "*.wav")))[:10]
    results_tflite_vs_h5 = []
    results_tflite_vs_clean = []

    print(f"{'Filename':<42} | {'TFLite vs H5 SI-SNR':>20} | {'TFLite vs Clean SI-SNR':>22} | {'Time (s)':>9}")
    print("-" * 100)

    for noisy_path in noisy_files:
        fname      = os.path.basename(noisy_path)
        clean_path = os.path.join(CLEAN_DIR, fname)

        noisy_audio, _ = librosa.load(noisy_path, sr=16000, mono=True)
        clean_audio, _ = librosa.load(clean_path, sr=16000, mono=True)

        t0 = time.perf_counter()
        tflite_out = engine.process_audio(noisy_audio)
        elapsed = time.perf_counter() - t0

        h5_out = h5_inference(ref, noisy_audio)

        tflite_trimmed = tflite_out[384:]
        h5_trimmed     = h5_out

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

    print(f"\nResult: TFLite/H5 agreement = {avg_vs_h5:.2f} dB | TFLite/Clean SI-SNR = {avg_vs_clean:.2f} dB")


if __name__ == "__main__":
    main()
