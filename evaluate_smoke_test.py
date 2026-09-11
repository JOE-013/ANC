#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_smoke_test.py - Quantitative evaluation script for smoke-test DTLN model.
Runs inference on data_test/train_mix using models_smoke_test/smoke_test.weights.h5.
Computes SI-SNR, STOI, and PESQ before and after enhancement, broken down by SNR level.
"""

import os
import glob
import csv
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
    Compute SI-SNR, STOI, and PESQ (if available) between clean_sig and test_sig.
    """
    # Truncate to same length if slight mismatch
    min_len = min(len(clean_sig), len(test_sig))
    clean_sig = clean_sig[:min_len]
    test_sig = test_sig[:min_len]
    
    # 1. SI-SNR
    si_snr_val = compute_si_snr(test_sig, clean_sig)
    
    # 2. STOI
    try:
        stoi_val = float(stoi(clean_sig, test_sig, fs, extended=False))
    except Exception as e:
        stoi_val = float('nan')

    # 3. PESQ
    if HAS_PESQ:
        try:
            pesq_val = float(pesq(fs, clean_sig, test_sig, 'wb'))
        except Exception:
            pesq_val = float('nan')
    else:
        pesq_val = None
        
    return si_snr_val, stoi_val, pesq_val

def main():
    print("=== DTLN Smoke-Test Model Quantitative Evaluation ===")
    
    noisy_dir = r"d:\SIH 2026\DTLN\data_test\train_mix"
    clean_dir = r"d:\SIH 2026\DTLN\data_test\train_speech"
    output_dir = r"d:\SIH 2026\DTLN\data_test\enhanced_smoke_test"
    csv_out_path = r"d:\SIH 2026\DTLN\data_test\eval_results_smoke_test.csv"
    
    weights_path = r"d:\SIH 2026\DTLN\models_smoke_test\smoke_test.weights.h5"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading smoke-test model (numUnits=64, numLayer=1)...")
    model_obj = DTLN_model()
    model_obj.numUnits = 64
    model_obj.numLayer = 1
    model_obj.build_DTLN_model()
    model_obj.model.load_weights(weights_path)
    print(f"Weights loaded successfully from {weights_path}")
    
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    print(f"Found {len(noisy_files)} noisy test files in {noisy_dir}")
    
    results = []
    
    print("\nRunning inference and computing evaluation metrics...")
    for idx, noisy_path in enumerate(noisy_files):
        filename = os.path.basename(noisy_path)
        clean_path = os.path.join(clean_dir, filename)
        enhanced_path = os.path.join(output_dir, filename)
        
        # Parse target input SNR condition from filename (e.g. pair_0000_snr-5dB_siren.wav)
        parts = filename.split("_")
        input_snr_cond = float(parts[2].replace("snr", "").replace("dB", ""))
        category = parts[3].replace(".wav", "")
        
        # Read noisy and clean audio
        clean_audio, fs = librosa.core.load(clean_path, sr=16000, mono=True)
        noisy_audio, _ = librosa.core.load(noisy_path, sr=16000, mono=True)
        
        # Run inference using model
        len_orig = len(noisy_audio)
        zero_pad = np.zeros(384)
        in_padded = np.concatenate((zero_pad, noisy_audio, zero_pad), axis=0)
        
        pred = model_obj.model.predict_on_batch(np.expand_dims(in_padded, axis=0).astype(np.float32))
        pred_speech = np.squeeze(pred)[384:384+len_orig]
        
        # Save enhanced WAV output
        sf.write(enhanced_path, pred_speech, fs)
        
        # Compute BEFORE metrics (noisy vs clean)
        si_snr_before, stoi_before, pesq_before = compute_metrics(clean_audio, noisy_audio, fs)
        
        # Compute AFTER metrics (enhanced vs clean)
        si_snr_after, stoi_after, pesq_after = compute_metrics(clean_audio, pred_speech, fs)
        
        row = {
            'filename': filename,
            'input_snr_cond': input_snr_cond,
            'category': category,
            'si_snr_before': si_snr_before,
            'si_snr_after': si_snr_after,
            'si_snr_gain': si_snr_after - si_snr_before,
            'stoi_before': stoi_before,
            'stoi_after': stoi_after,
            'stoi_gain': stoi_after - stoi_before,
            'pesq_before': pesq_before if pesq_before is not None else 'N/A',
            'pesq_after': pesq_after if pesq_after is not None else 'N/A',
            'pesq_gain': (pesq_after - pesq_before) if (pesq_after is not None and pesq_before is not None) else 'N/A'
        }
        results.append(row)
        
        if (idx + 1) % 15 == 0 or (idx + 1) == len(noisy_files):
            print(f"  Processed {idx + 1}/{len(noisy_files)} files...")
            
    # Write CSV
    fieldnames = ['filename', 'input_snr_cond', 'category', 
                  'si_snr_before', 'si_snr_after', 'si_snr_gain',
                  'stoi_before', 'stoi_after', 'stoi_gain',
                  'pesq_before', 'pesq_after', 'pesq_gain']
    
    with open(csv_out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nFull per-file evaluation saved to: {csv_out_path}")
    
    # Aggregated Summary by Input SNR Level
    snr_levels = sorted(list(set(r['input_snr_cond'] for r in results)))
    
    print("\n" + "="*85)
    print(f"{'Input SNR':<10} | {'SI-SNR Before':<14} | {'SI-SNR After':<14} | {'STOI Before':<12} | {'STOI After':<12} | {'PESQ After':<10}")
    print("="*85)
    
    for snr_val in snr_levels:
        sub = [r for r in results if r['input_snr_cond'] == snr_val]
        mean_sisnr_b = np.mean([r['si_snr_before'] for r in sub])
        mean_sisnr_a = np.mean([r['si_snr_after'] for r in sub])
        mean_stoi_b = np.mean([r['stoi_before'] for r in sub])
        mean_stoi_a = np.mean([r['stoi_after'] for r in sub])
        
        pesq_str = "N/A"
        if HAS_PESQ and isinstance(sub[0]['pesq_after'], float):
            pesq_str = f"{np.mean([r['pesq_after'] for r in sub]):.3f}"
            
        print(f"{snr_val:<10.1f}dB | {mean_sisnr_b:<14.2f}dB | {mean_sisnr_a:<14.2f}dB | {mean_stoi_b:<12.4f} | {mean_stoi_a:<12.4f} | {pesq_str:<10}")
        
    print("-" * 85)
    overall_sisnr_b = np.mean([r['si_snr_before'] for r in results])
    overall_sisnr_a = np.mean([r['si_snr_after'] for r in results])
    overall_stoi_b = np.mean([r['stoi_before'] for r in results])
    overall_stoi_a = np.mean([r['stoi_after'] for r in results])
    
    overall_pesq_str = "N/A"
    if HAS_PESQ and isinstance(results[0]['pesq_after'], float):
        overall_pesq_str = f"{np.mean([r['pesq_after'] for r in results]):.3f}"
        
    print(f"{'OVERALL':<10} | {overall_sisnr_b:<14.2f}dB | {overall_sisnr_a:<14.2f}dB | {overall_stoi_b:<12.4f} | {overall_stoi_a:<12.4f} | {overall_pesq_str:<10}")
    print("="*85)

if __name__ == '__main__':
    main()
