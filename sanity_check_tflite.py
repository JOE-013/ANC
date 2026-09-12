#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_check_tflite.py — Compare DTLN TFLite/LiteRT model predictions against clean reference speech.
Uses LiteRT (ai_edge_litert) with fallbacks for tflite_runtime and tensorflow.lite.
"""

import os
import glob
import time
import numpy as np
import librosa

# Fallback import for LiteRT / tflite-runtime / tensorflow.lite
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter

def compute_si_snr(estimate, target):
    """Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB."""
    target = target - np.mean(target)
    estimate = estimate - np.mean(estimate)

    n = min(len(estimate), len(target))
    estimate = estimate[:n]
    target = target[:n]

    dot_product = np.dot(estimate, target)
    target_energy = np.dot(target, target) + 1e-7
    s_target = (dot_product / target_energy) * target

    e_noise = estimate - s_target
    target_power = np.sum(s_target ** 2)
    noise_power = np.sum(e_noise ** 2) + 1e-7

    return float(10.0 * np.log10(target_power / noise_power))

class DTLN_TFLite_Inference:
    """DTLN TFLite/LiteRT inference engine with dynamic tensor index mapping."""
    def __init__(self, model1_path, model2_path):
        self.interpreter_1 = Interpreter(model_path=model1_path)
        self.interpreter_1.allocate_tensors()

        self.interpreter_2 = Interpreter(model_path=model2_path)
        self.interpreter_2.allocate_tensors()

        in1 = self.interpreter_1.get_input_details()
        out1 = self.interpreter_1.get_output_details()
        in2 = self.interpreter_2.get_input_details()
        out2 = self.interpreter_2.get_output_details()

        for d in in1:
            shape = d["shape"]
            if len(shape) == 4 and shape[2] == 64 and shape[3] == 2:
                self.idx_states_in_1 = d["index"]
            elif len(shape) == 3 and shape[2] == 257:
                self.idx_mag_in_1 = d["index"]

        for d in out1:
            shape = d["shape"]
            if len(shape) == 4 and shape[2] == 64 and shape[3] == 2:
                self.idx_states_out_1 = d["index"]
            elif len(shape) == 3 and shape[2] == 257:
                self.idx_mask_out_1 = d["index"]

        for d in in2:
            shape = d["shape"]
            if len(shape) == 4 and shape[2] == 64 and shape[3] == 2:
                self.idx_states_in_2 = d["index"]
            elif len(shape) == 3 and shape[2] == 512:
                self.idx_frame_in_2 = d["index"]

        for d in out2:
            shape = d["shape"]
            if len(shape) == 4 and shape[2] == 64 and shape[3] == 2:
                self.idx_states_out_2 = d["index"]
            elif len(shape) == 3 and shape[2] == 512:
                self.idx_decoded_out_2 = d["index"]

    def process_audio(self, audio_data, block_len=512, block_shift=128):
        states_1 = np.zeros((1, 1, 64, 2), dtype=np.float32)
        states_2 = np.zeros((1, 1, 64, 2), dtype=np.float32)

        in_buffer = np.zeros(block_len, dtype=np.float32)
        out_buffer = np.zeros(block_len, dtype=np.float32)
        out_file = np.zeros(len(audio_data), dtype=np.float32)

        num_blocks = (len(audio_data) - (block_len - block_shift)) // block_shift

        for idx in range(num_blocks):
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audio_data[
                idx * block_shift:idx * block_shift + block_shift
            ]

            in_block_fft = np.fft.rfft(in_buffer)
            in_mag = np.abs(in_block_fft)
            in_phase = np.angle(in_block_fft)

            in_mag_tensor = np.reshape(
                in_mag, (1, 1, -1)
            ).astype(np.float32)

            self.interpreter_1.set_tensor(
                self.idx_states_in_1, states_1
            )
            self.interpreter_1.set_tensor(
                self.idx_mag_in_1, in_mag_tensor
            )
            self.interpreter_1.invoke()

            out_mask = self.interpreter_1.get_tensor(
                self.idx_mask_out_1
            )
            states_1 = self.interpreter_1.get_tensor(
                self.idx_states_out_1
            )

            estimated_complex = (
                in_mag_tensor * out_mask *
                np.exp(1j * in_phase)
            )

            estimated_block = np.reshape(
                np.fft.irfft(estimated_complex),
                (1, 1, -1)
            ).astype(np.float32)

            self.interpreter_2.set_tensor(
                self.idx_states_in_2, states_2
            )
            self.interpreter_2.set_tensor(
                self.idx_frame_in_2, estimated_block
            )
            self.interpreter_2.invoke()

            out_block = self.interpreter_2.get_tensor(
                self.idx_decoded_out_2
            )

            states_2 = self.interpreter_2.get_tensor(
                self.idx_states_out_2
            )

            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = 0.0
            out_buffer += np.squeeze(out_block)

            start = idx * block_shift
            end = min(start + block_shift, len(out_file))
            out_file[start:end] = out_buffer[:end - start]

        return out_file

def main():
    print("=== DTLN TFLite/LiteRT Sanity Check ===")
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    noisy_dir = os.path.join(script_dir, "data_full", "val_mix")
    clean_dir = os.path.join(script_dir, "data_full", "val_speech")

    model1_path = os.path.join(script_dir, "models_full_run", "model_1.tflite")
    model2_path = os.path.join(script_dir, "models_full_run", "model_2.tflite")

    print("Loading LiteRT/TFLite models...")
    engine = DTLN_TFLite_Inference(model1_path, model2_path)
    print("Both models loaded successfully.")
    print()

    noisy_files = sorted(
        glob.glob(os.path.join(noisy_dir, "*.wav"))
    )[:10]

    print("Files found:", len(noisy_files))
    print()

    results = []

    print(f"{'Filename':<38} | {'SI-SNR (dB)':>11} | {'Time (s)':>10}")
    print("-" * 66)

    for noisy_path in noisy_files:
        filename = os.path.basename(noisy_path)
        clean_path = os.path.join(clean_dir, filename)

        noisy_audio, _ = librosa.load(
            noisy_path, sr=16000, mono=True
        )

        clean_audio, _ = librosa.load(
            clean_path, sr=16000, mono=True
        )

        start_time = time.perf_counter()

        enhanced_audio = engine.process_audio(noisy_audio)

        elapsed = time.perf_counter() - start_time

        enhanced_audio = enhanced_audio[384:]
        n = min(len(enhanced_audio), len(clean_audio))

        snr = compute_si_snr(
            enhanced_audio[:n],
            clean_audio[:n]
        )

        results.append(snr)

        print(
            f"{filename:<38} | "
            f"{snr:>11.2f} | "
            f"{elapsed:>10.3f}"
        )

    print("-" * 66)

    if results:
        print(
            f"{'AVERAGE':<38} | "
            f"{np.mean(results):>11.2f} |"
        )

        print()
        print("=== RESULT ===")
        print("TFLite/LiteRT inference completed successfully.")
        print(f"Average SI-SNR: {np.mean(results):.2f} dB")
        print(f"Files tested: {len(results)}")
    else:
        print("No WAV files found.")

if __name__ == "__main__":
    main()
