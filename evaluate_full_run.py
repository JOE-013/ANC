#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_full_run.py - Quantitative evaluation script for the full-run DTLN model.
Runs inference on data_full/val_mix (120 held-out validation files) using models_full_run/full_run.weights.h5.
Computes SI-SNR, STOI, and PESQ before and after enhancement, broken down by SNR level (-5, 0, 5, 10, 15 dB).
Compares performance directly against the smoke-test model evaluation.
"""

import os
import glob
import csv
import re
import numpy as np
import soundfile as sf
import librosa
from DTLN_model import DTLN_model
from pystoi import stoi

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

def compute_si_snr(estimate, target):
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.
    """
    target = target - np.mean(target)
    estimate = estimate - np.mean(estimate)
    
    # Project estimate onto target
    dot_product = np.dot(estimate, target)
    target_energy = np.dot(target, target) + 1e-7
    s_target = (dot_product / target_energy) * target
    
    e_noise = estimate - s_target
    
    target_power = np.sum(np.square(s_target))
    noise_power = np.sum(np.square(e_noise)) + 1e-7
    
    si_snr = 10.0 * np.log10(target_power / noise_power)
    return float(si_snr)

def compute_metrics(clean_sig, test_sig, fs=16000):
    """
    Compute SI-SNR, STOI, and PESQ (wideband) between clean_sig and test_sig.
    """
    min_len = min(len(clean_sig), len(test_sig))
    clean_sig = clean_sig[:min_len]
    test_sig = test_sig[:min_len]
    
    # 1. SI-SNR
    si_snr_val = compute_si_snr(test_sig, clean_sig)
    
    # 2. STOI
    try:
        stoi_val = float(stoi(clean_sig, test_sig, fs, extended=False))
    except Exception:
        stoi_val = float('nan')

    # 3. PESQ (wideband 16kHz)
    if HAS_PESQ:
        try:
            pesq_val = float(pesq(fs, clean_sig, test_sig, 'wb'))
        except Exception:
            pesq_val = float('nan')
    else:
        pesq_val = None
        
    return si_snr_val, stoi_val, pesq_val

def main():
    print("=== DTLN Full-Run Model Quantitative Evaluation (Held-Out Validation Set) ===")
    
    noisy_dir = r"d:\SIH 2026\DTLN\data_full\val_mix"
    clean_dir = r"d:\SIH 2026\DTLN\data_full\val_speech"
    output_dir = r"d:\SIH 2026\DTLN\data_full\enhanced_full_run"
    csv_out_path = r"d:\SIH 2026\DTLN\data_full\eval_results_full_run.csv"
    smoke_csv_path = r"d:\SIH 2026\DTLN\data_test\eval_results_smoke_test.csv"
    
    weights_path = r"d:\SIH 2026\DTLN\models_full_run\full_run.weights.h5"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading full-run model weights (numUnits=64, numLayer=1)...")
    model_obj = DTLN_model()
    model_obj.numUnits = 64
    model_obj.numLayer = 1
    model_obj.build_DTLN_model()
    model_obj.model.load_weights(weights_path)
    print(f"Weights loaded successfully from {weights_path}")
    
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    print(f"Found {len(noisy_files)} validation audio files in {noisy_dir}")
    
    results = []
    
    print("\nRunning inference and computing PESQ / STOI / SI-SNR metrics...")
    for idx, noisy_path in enumerate(noisy_files):
        filename = os.path.basename(noisy_path)
        clean_path = os.path.join(clean_dir, filename)
        enhanced_path = os.path.join(output_dir, filename)
        
        # Parse SNR level using regex (e.g. pair_val_0000_snr-5dB_musan_noise.wav)
        snr_match = re.search(r'snr(-?\d+)dB', filename)
        input_snr_cond = float(snr_match.group(1)) if snr_match else 0.0
        
        # Read noisy and clean audio
        clean_audio, fs = librosa.core.load(clean_path, sr=16000, mono=True)
        noisy_audio, _ = librosa.core.load(noisy_path, sr=16000, mono=True)
        
        # Run inference using DTLN model
        len_orig = len(noisy_audio)
        zero_pad = np.zeros(384)
        in_padded = np.concatenate((zero_pad, noisy_audio, zero_pad), axis=0)
        
        pred = model_obj.model.predict_on_batch(np.expand_dims(in_padded, axis=0).astype(np.float32))
        pred_speech = np.squeeze(pred)[384:384+len_orig]
        
        # Save enhanced WAV output
        sf.write(enhanced_path, pred_speech, fs)
        
        # Compute BEFORE metrics (noisy vs clean)
        si_snr_b, stoi_b, pesq_b = compute_metrics(clean_audio, noisy_audio, fs)
        
        # Compute AFTER metrics (enhanced vs clean)
        si_snr_a, stoi_a, pesq_a = compute_metrics(clean_audio, pred_speech, fs)
        
        row = {
            'filename': filename,
            'input_snr_cond': input_snr_cond,
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
            print(f"  Processed {idx + 1}/{len(noisy_files)} validation files...")
            
    # Save CSV
    fieldnames = ['filename', 'input_snr_cond',
                  'si_snr_before', 'si_snr_after', 'si_snr_gain',
                  'stoi_before', 'stoi_after', 'stoi_gain',
                  'pesq_before', 'pesq_after', 'pesq_gain']
    
    with open(csv_out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nFull per-file evaluation saved to: {csv_out_path}")
    
    # 1. Aggregated Summary by Input SNR Level
    snr_levels = sorted(list(set(r['input_snr_cond'] for r in results)))
    
    print("\n" + "="*95)
    print("Full-Run Model Validation Performance Breakdown by Input SNR Level")
    print("="*95)
    print(f"{'Input SNR':<10} | {'SI-SNR Before':<14} | {'SI-SNR After':<14} | {'STOI Before':<12} | {'STOI After':<12} | {'PESQ Before':<11} | {'PESQ After':<11}")
    print("-"*95)
    
    for snr_val in snr_levels:
        sub = [r for r in results if r['input_snr_cond'] == snr_val]
        mean_sisnr_b = np.mean([r['si_snr_before'] for r in sub])
        mean_sisnr_a = np.mean([r['si_snr_after'] for r in sub])
        mean_stoi_b = np.mean([r['stoi_before'] for r in sub])
        mean_stoi_a = np.mean([r['stoi_after'] for r in sub])
        mean_pesq_b = np.mean([r['pesq_before'] for r in sub])
        mean_pesq_a = np.mean([r['pesq_after'] for r in sub])
            
        print(f"{snr_val:<10.1f}dB | {mean_sisnr_b:<14.2f}dB | {mean_sisnr_a:<14.2f}dB | {mean_stoi_b:<12.4f} | {mean_stoi_a:<12.4f} | {mean_pesq_b:<11.3f} | {mean_pesq_a:<11.3f}")
        
    print("-" * 95)
    overall_sisnr_b = np.mean([r['si_snr_before'] for r in results])
    overall_sisnr_a = np.mean([r['si_snr_after'] for r in results])
    overall_stoi_b = np.mean([r['stoi_before'] for r in results])
    overall_stoi_a = np.mean([r['stoi_after'] for r in results])
    overall_pesq_b = np.mean([r['pesq_before'] for r in results])
    overall_pesq_a = np.mean([r['pesq_after'] for r in results])
        
    print(f"{'OVERALL':<10} | {overall_sisnr_b:<14.2f}dB | {overall_sisnr_a:<14.2f}dB | {overall_stoi_b:<12.4f} | {overall_stoi_a:<12.4f} | {overall_pesq_b:<11.3f} | {overall_pesq_a:<11.3f}")
    print("="*95)

    # 2. Smoke Test vs Full Run Comparison
    smoke_rows = []
    if os.path.exists(smoke_csv_path):
        with open(smoke_csv_path, 'r', newline='') as f:
            smoke_rows = list(csv.DictReader(f))
            
    if smoke_rows:
        s_sisnr_b = np.mean([float(r['si_snr_before']) for r in smoke_rows])
        s_sisnr_a = np.mean([float(r['si_snr_after']) for r in smoke_rows])
        s_sisnr_g = s_sisnr_a - s_sisnr_b
        
        s_stoi_b = np.mean([float(r['stoi_before']) for r in smoke_rows])
        s_stoi_a = np.mean([float(r['stoi_after']) for r in smoke_rows])
        s_stoi_g = s_stoi_a - s_stoi_b
        
        s_pesq_b = np.mean([float(r['pesq_before']) for r in smoke_rows])
        s_pesq_a = np.mean([float(r['pesq_after']) for r in smoke_rows])
        s_pesq_g = s_pesq_a - s_pesq_b
        
        f_sisnr_g = overall_sisnr_a - overall_sisnr_b
        f_stoi_g = overall_stoi_a - overall_stoi_b
        f_pesq_g = overall_pesq_a - overall_pesq_b
        
        print("\n" + "="*85)
        print("Direct Comparison: Smoke-Test Model vs. Full-Run Model (Overall Averages)")
        print("="*85)
        print(f"{'Metric':<12} | {'Smoke Test Before':<18} | {'Smoke Test After (Gain)':<24} | {'Full Run Before':<16} | {'Full Run After (Gain)':<22}")
        print("-"*85)
        print(f"{'SI-SNR (dB)':<12} | {s_sisnr_b:<18.2f} | {s_sisnr_a:.2f} ({s_sisnr_g:+0.2f} dB)         | {overall_sisnr_b:<16.2f} | {overall_sisnr_a:.2f} ({f_sisnr_g:+0.2f} dB)")
        print(f"{'STOI':<12} | {s_stoi_b:<18.4f} | {s_stoi_a:.4f} ({s_stoi_g:+0.4f})           | {overall_stoi_b:<16.4f} | {overall_stoi_a:.4f} ({f_stoi_g:+0.4f})")
        print(f"{'PESQ (wb)':<12} | {s_pesq_b:<18.3f} | {s_pesq_a:.3f} ({s_pesq_g:+0.3f})           | {overall_pesq_b:<16.3f} | {overall_pesq_a:.3f} ({f_pesq_g:+0.3f})")
        print("="*85)

    # 3. Target Proximity Summary
    print("\n=== Target Proximity & Goal Assessment ===")
    print(f"Target 1: SI-SNR > 15.0 dB  --> Current Enhanced SI-SNR: {overall_sisnr_a:.2f} dB (Gap: {15.0 - overall_sisnr_a:.2f} dB)")
    print(f"Target 2: STOI > 0.8500     --> Current Enhanced STOI:   {overall_stoi_a:.4f} (Gap: {0.8500 - overall_stoi_a:.4f})")
    print(f"Target 3: PESQ > 2.500      --> Current Enhanced PESQ:   {overall_pesq_a:.3f} (Gap: {2.500 - overall_pesq_a:.3f})")

if __name__ == '__main__':
    main()
