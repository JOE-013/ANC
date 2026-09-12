#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_tflite_full_run.py — Quantitative evaluation of DTLN TFLite model on validation dataset.
Processes all 120 files in data_full/val_mix using model_1.tflite and model_2.tflite.
Includes high-resolution per-block timing instrumentation (isolated from I/O and metrics).
Computes SI-SNR, STOI, PESQ, average per-block latency (ms), and Real-Time Factor (RTF).
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

# Support both tflite_runtime (lightweight Pi) and full tensorflow (PC)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

def compute_si_snr(estimate, target):
    """Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB."""
    target = target - np.mean(target)
    estimate = estimate - np.mean(estimate)
    
    dot_product = np.dot(estimate, target)
    target_energy = np.dot(target, target) + 1e-7
    s_target = (dot_product / target_energy) * target
    
    e_noise = estimate - s_target
    target_power = np.sum(np.square(s_target))
    noise_power = np.sum(np.square(e_noise)) + 1e-7
    
    return float(10.0 * np.log10(target_power / noise_power))

def compute_metrics(clean_sig, test_sig, fs=16000):
    """Compute SI-SNR, STOI, and PESQ (wideband) between clean_sig and test_sig."""
    min_len = min(len(clean_sig), len(test_sig))
    clean_sig = clean_sig[:min_len]
    test_sig = test_sig[:min_len]
    
    si_snr_val = compute_si_snr(test_sig, clean_sig)
    
    try:
        stoi_val = float(stoi(clean_sig, test_sig, fs, extended=False))
    except Exception:
        stoi_val = float('nan')

    if HAS_PESQ:
        try:
            pesq_val = float(pesq(fs, clean_sig, test_sig, 'wb'))
        except Exception:
            pesq_val = float('nan')
    else:
        pesq_val = None
        
    return si_snr_val, stoi_val, pesq_val

class DTLN_TFLite_Evaluator:
    """DTLN TFLite Inference Engine with timing instrumentation."""
    def __init__(self, model1_path, model2_path):
        self.interpreter_1 = tflite.Interpreter(model_path=model1_path)
        self.interpreter_1.allocate_tensors()
        self.interpreter_2 = tflite.Interpreter(model_path=model2_path)
        self.interpreter_2.allocate_tensors()
        
        self.in_details_1 = self.interpreter_1.get_input_details()
        self.out_details_1 = self.interpreter_1.get_output_details()
        self.in_details_2 = self.interpreter_2.get_input_details()
        self.out_details_2 = self.interpreter_2.get_output_details()
        
        # Stage 1 Tensor Mapping
        for detail in self.in_details_1:
            shape = detail['shape']
            if len(shape) == 4 and shape[2] == 64 and shape[3] == 2:
                self.idx_states_in_1 = detail['index']
            elif len(shape) == 3 and shape[2] == 257:
                self.idx_mag_in_1 = detail['index']
                
        for detail in self.out_details_1:
            shape = detail['shape']
            if len(shape) == 4 and shape[2] == 64 and shape[3] == 2:
                self.idx_states_out_1 = detail['index']
            elif len(shape) == 3 and shape[2] == 257:
                self.idx_mask_out_1 = detail['index']
                
        # Stage 2 Tensor Mapping
        for detail in self.in_details_2:
            shape = detail['shape']
            if len(shape) == 4 and shape[2] == 64 and shape[3] == 2:
                self.idx_states_in_2 = detail['index']
            elif len(shape) == 3 and shape[2] == 512:
                self.idx_frame_in_2 = detail['index']
                
        for detail in self.out_details_2:
            shape = detail['shape']
            if len(shape) == 4 and shape[2] == 64 and shape[3] == 2:
                self.idx_states_out_2 = detail['index']
            elif len(shape) == 3 and shape[2] == 512:
                self.idx_decoded_out_2 = detail['index']

    def process_audio(self, audio_data, block_len=512, block_shift=128):
        """
        Process audio array frame-by-frame with high-resolution timing.
        Returns:
            out_file (np.ndarray): Enhanced audio array
            block_latencies (list): Per-block processing latency in seconds
            pure_inference_time (float): Total processing time in seconds for this file
        """
        states_1 = np.zeros((1, 1, 64, 2), dtype=np.float32)
        states_2 = np.zeros((1, 1, 64, 2), dtype=np.float32)
        
        in_buffer = np.zeros(block_len, dtype=np.float32)
        out_buffer = np.zeros(block_len, dtype=np.float32)
        out_file = np.zeros(len(audio_data), dtype=np.float32)
        
        num_blocks = (len(audio_data) - (block_len - block_shift)) // block_shift
        block_latencies = []
        
        for idx in range(num_blocks):
            t_start = time.perf_counter()
            
            # Shift input buffer
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audio_data[idx * block_shift : idx * block_shift + block_shift]
            
            # STFT computation
            in_block_fft = np.fft.rfft(in_buffer)
            in_mag = np.abs(in_block_fft)
            in_phase = np.angle(in_block_fft)
            in_mag_tensor = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)
            
            # Stage 1 Inference
            self.interpreter_1.set_tensor(self.idx_states_in_1, states_1)
            self.interpreter_1.set_tensor(self.idx_mag_in_1, in_mag_tensor)
            self.interpreter_1.invoke()
            out_mask = self.interpreter_1.get_tensor(self.idx_mask_out_1)
            states_1 = self.interpreter_1.get_tensor(self.idx_states_out_1)
            
            # iFFT reconstruction
            estimated_complex = in_mag_tensor * out_mask * np.exp(1j * in_phase)
            estimated_block = np.reshape(np.fft.irfft(estimated_complex), (1, 1, -1)).astype(np.float32)
            
            # Stage 2 Inference
            self.interpreter_2.set_tensor(self.idx_states_in_2, states_2)
            self.interpreter_2.set_tensor(self.idx_frame_in_2, estimated_block)
            self.interpreter_2.invoke()
            out_block = self.interpreter_2.get_tensor(self.idx_decoded_out_2)
            states_2 = self.interpreter_2.get_tensor(self.idx_states_out_2)
            
            # Overlap-add reconstruction
            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = 0.0
            out_buffer += np.squeeze(out_block)
            out_file[idx * block_shift : idx * block_shift + block_shift] = out_buffer[:block_shift]
            
            t_end = time.perf_counter()
            block_latencies.append(t_end - t_start)
            
        pure_inference_time = sum(block_latencies)
        return out_file, block_latencies, pure_inference_time

def main():
    print("=== DTLN TFLite Full Validation Evaluation with Latency Instrumentation ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    noisy_dir = os.path.join(script_dir, "data_full", "val_mix")
    clean_dir = os.path.join(script_dir, "data_full", "val_speech")
    output_dir = os.path.join(script_dir, "data_full", "enhanced_tflite_full_run")
    csv_out_path = os.path.join(script_dir, "data_full", "eval_results_tflite_full_run.csv")
    
    model1_path = os.path.join(script_dir, "models_full_run", "model_1.tflite")
    model2_path = os.path.join(script_dir, "models_full_run", "model_2.tflite")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading TFLite interpreters from {model1_path} and {model2_path}...")
    engine = DTLN_TFLite_Evaluator(model1_path, model2_path)
    
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    print(f"Found {len(noisy_files)} validation audio files in {noisy_dir}\n")
    
    results = []
    all_block_latencies_ms = []
    
    print("Running TFLite frame-by-frame evaluation and measuring latency...")
    for idx, noisy_path in enumerate(noisy_files):
        filename = os.path.basename(noisy_path)
        clean_path = os.path.join(clean_dir, filename)
        enhanced_path = os.path.join(output_dir, filename)
        
        snr_match = re.search(r'snr(-?\d+)dB', filename)
        input_snr_cond = float(snr_match.group(1)) if snr_match else 0.0
        
        # Audio loading (NOT included in timing measurement)
        clean_audio, fs = librosa.core.load(clean_path, sr=16000, mono=True)
        noisy_audio, _ = librosa.core.load(noisy_path, sr=16000, mono=True)
        
        # Pure TFLite Inference & Processing
        pred_speech, latencies, total_inf_time = engine.process_audio(noisy_audio)
        
        audio_dur = len(noisy_audio) / fs
        rtf = total_inf_time / audio_dur
        avg_block_lat_ms = (np.mean(latencies) * 1000.0) if latencies else 0.0
        all_block_latencies_ms.extend([l * 1000.0 for l in latencies])
        
        # Save WAV output
        sf.write(enhanced_path, pred_speech, fs)
        
        # Compensate for 384-sample (24 ms / 3 blocks) overlap-add startup delay
        pred_aligned = pred_speech[384:]
        clean_aligned = clean_audio[:len(pred_aligned)]
        noisy_aligned = noisy_audio[:len(pred_aligned)]
        
        # Metrics computation (aligned)
        si_snr_b, stoi_b, pesq_b = compute_metrics(clean_aligned, noisy_aligned, fs)
        si_snr_a, stoi_a, pesq_a = compute_metrics(clean_aligned, pred_aligned, fs)
        
        row = {
            'filename': filename,
            'input_snr_cond': input_snr_cond,
            'duration_sec': round(audio_dur, 3),
            'num_blocks': len(latencies),
            'pure_inf_time_sec': round(total_inf_time, 4),
            'avg_block_latency_ms': round(avg_block_lat_ms, 3),
            'rtf': round(rtf, 4),
            'si_snr_before': si_snr_b,
            'si_snr_after': si_snr_a,
            'si_snr_gain': si_snr_a - si_snr_b,
            'stoi_before': stoi_b,
            'stoi_after': stoi_a,
            'stoi_gain': stoi_a - stoi_b,
            'pesq_before': pesq_b if pesq_b is not None else 'N/A',
            'pesq_after': pesq_a if pesq_a is not None else 'N/A',
            'pesq_gain': (pesq_a - pesq_b) if (pesq_a is not None and pesq_b is not None) else 'N/A'
        }
        results.append(row)
        
        if (idx + 1) % 30 == 0 or (idx + 1) == len(noisy_files):
            print(f"  Processed {idx + 1}/{len(noisy_files)} files | Avg Block Latency: {avg_block_lat_ms:.3f} ms | RTF: {rtf:.4f}")
            
    # Write CSV
    fieldnames = ['filename', 'input_snr_cond', 'duration_sec', 'num_blocks',
                  'pure_inf_time_sec', 'avg_block_latency_ms', 'rtf',
                  'si_snr_before', 'si_snr_after', 'si_snr_gain',
                  'stoi_before', 'stoi_after', 'stoi_gain',
                  'pesq_before', 'pesq_after', 'pesq_gain']
                  
    with open(csv_out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nFull TFLite evaluation results written to: {csv_out_path}")
    
    # Aggregated Summary by SNR Level
    snr_levels = sorted(list(set(r['input_snr_cond'] for r in results)))
    
    print("\n" + "="*112)
    print("DTLN TFLite Model Validation Performance Breakdown by Input SNR Level")
    print("="*112)
    print(f"{'Input SNR':<10} | {'SI-SNR Before':<14} | {'SI-SNR After':<14} | {'STOI Before':<12} | {'STOI After':<12} | {'PESQ Before':<11} | {'PESQ After':<11} | {'Block Latency':<13} | {'RTF':<8}")
    print("-"*112)
    
    for snr_val in snr_levels:
        sub = [r for r in results if r['input_snr_cond'] == snr_val]
        m_sisnr_b = np.mean([r['si_snr_before'] for r in sub])
        m_sisnr_a = np.mean([r['si_snr_after'] for r in sub])
        m_stoi_b = np.mean([r['stoi_before'] for r in sub])
        m_stoi_a = np.mean([r['stoi_after'] for r in sub])
        m_pesq_b = np.mean([r['pesq_before'] for r in sub if r['pesq_before'] != 'N/A'])
        m_pesq_a = np.mean([r['pesq_after'] for r in sub if r['pesq_after'] != 'N/A'])
        m_lat = np.mean([r['avg_block_latency_ms'] for r in sub])
        m_rtf = np.mean([r['rtf'] for r in sub])
        
        print(f"{snr_val:<10.1f}dB | {m_sisnr_b:<14.2f}dB | {m_sisnr_a:<14.2f}dB | {m_stoi_b:<12.4f} | {m_stoi_a:<12.4f} | {m_pesq_b:<11.3f} | {m_pesq_a:<11.3f} | {m_lat:<10.3f} ms | {m_rtf:<8.4f}")
        
    print("-" * 112)
    ov_sisnr_b = np.mean([r['si_snr_before'] for r in results])
    ov_sisnr_a = np.mean([r['si_snr_after'] for r in results])
    ov_stoi_b = np.mean([r['stoi_before'] for r in results])
    ov_stoi_a = np.mean([r['stoi_after'] for r in results])
    ov_pesq_b = np.mean([r['pesq_before'] for r in results if r['pesq_before'] != 'N/A'])
    ov_pesq_a = np.mean([r['pesq_after'] for r in results if r['pesq_after'] != 'N/A'])
    ov_lat = np.mean([r['avg_block_latency_ms'] for r in results])
    ov_rtf = np.mean([r['rtf'] for r in results])
    
    print(f"{'OVERALL':<10} | {ov_sisnr_b:<14.2f}dB | {ov_sisnr_a:<14.2f}dB | {ov_stoi_b:<12.4f} | {ov_stoi_a:<12.4f} | {ov_pesq_b:<11.3f} | {ov_pesq_a:<11.3f} | {ov_lat:<10.3f} ms | {ov_rtf:<8.4f}")
    print("="*112)
    
    print("\n=== Latency & Real-Time Performance Summary ===")
    print(f"  Audio Block Size (Hop Size): 8.000 ms (128 samples @ 16 kHz)")
    print(f"  Average Per-Block Latency:   {ov_lat:.3f} ms / block")
    print(f"  Latency Budget Margin:       {8.000 - ov_lat:.3f} ms headroom per block (Budget: 8.000 ms)")
    print(f"  Overall Real-Time Factor:    {ov_rtf:.4f}x ({'FASTER than real-time' if ov_rtf < 1.0 else 'SLOWER than real-time'})")

if __name__ == '__main__':
    main()
