#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_full_run_with_limiter.py - Evaluation script for DTLN + Transient Limiter chained pipeline.
Applies transient_limiter.py to data_full/enhanced_full_run/ (120 validation files),
saves chained outputs to data_full/enhanced_full_run_with_limiter/, computes PESQ/STOI/SI-SNR,
saves eval_results_full_run_with_limiter.csv, and produces a 3-way comparative analysis:
Noisy vs. DTLN-only vs. DTLN + Limiter.
"""

import os
import glob
import csv
import re
import numpy as np
import soundfile as sf
import librosa
from transient_limiter import apply_transient_limiter
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
    print("=== DTLN + Transient Limiter Chained Evaluation (Held-Out Validation Set) ===")
    
    dtln_enhanced_dir = r"d:\SIH 2026\DTLN\data_full\enhanced_full_run"
    clean_dir = r"d:\SIH 2026\DTLN\data_full\val_speech"
    noisy_dir = r"d:\SIH 2026\DTLN\data_full\val_mix"
    
    chained_output_dir = r"d:\SIH 2026\DTLN\data_full\enhanced_full_run_with_limiter"
    csv_out_path = r"d:\SIH 2026\DTLN\data_full\eval_results_full_run_with_limiter.csv"
    dtln_only_csv_path = r"d:\SIH 2026\DTLN\data_full\eval_results_full_run.csv"
    
    os.makedirs(chained_output_dir, exist_ok=True)
    
    dtln_files = sorted(glob.glob(os.path.join(dtln_enhanced_dir, "*.wav")))
    print(f"Found {len(dtln_files)} DTLN-enhanced audio files in {dtln_enhanced_dir}")
    
    # Parameters for the transient limiter
    limiter_params = {
        'threshold_db': -6.0,
        'attack_ms': 2.0,
        'release_ms': 50.0,
        'win_ms': 3.0
    }
    
    print(f"Limiter Parameters: {limiter_params}\n")
    print("Applying transient limiter to DTLN outputs & computing evaluation metrics...")
    
    results = []
    
    for idx, dtln_path in enumerate(dtln_files):
        filename = os.path.basename(dtln_path)
        clean_path = os.path.join(clean_dir, filename)
        noisy_path = os.path.join(noisy_dir, filename)
        chained_path = os.path.join(chained_output_dir, filename)
        
        snr_match = re.search(r'snr(-?\d+)dB', filename)
        input_snr_cond = float(snr_match.group(1)) if snr_match else 0.0
        
        # Load audio signals
        clean_sig, fs = librosa.core.load(clean_path, sr=16000, mono=True)
        noisy_sig, _ = librosa.core.load(noisy_path, sr=16000, mono=True)
        dtln_sig, _ = librosa.core.load(dtln_path, sr=16000, mono=True)
        
        # Apply transient limiter to DTLN output
        lim_sig, gain_curve = apply_transient_limiter(
            dtln_sig,
            fs=fs,
            threshold_db=limiter_params['threshold_db'],
            attack_ms=limiter_params['attack_ms'],
            release_ms=limiter_params['release_ms'],
            win_ms=limiter_params['win_ms']
        )
        
        # Save chained output WAV
        sf.write(chained_path, lim_sig, fs)
        
        # Compute metrics for chained output (DTLN + Limiter vs Clean)
        si_snr_chained, stoi_chained, pesq_chained = compute_metrics(clean_sig, lim_sig, fs)
        
        # Compute metrics for Noisy vs Clean
        si_snr_noisy, stoi_noisy, pesq_noisy = compute_metrics(clean_sig, noisy_sig, fs)
        
        # Compute metrics for DTLN-only vs Clean
        si_snr_dtln, stoi_dtln, pesq_dtln = compute_metrics(clean_sig, dtln_sig, fs)
        
        row = {
            'filename': filename,
            'input_snr_cond': input_snr_cond,
            'si_snr_noisy': si_snr_noisy,
            'si_snr_dtln': si_snr_dtln,
            'si_snr_chained': si_snr_chained,
            'stoi_noisy': stoi_noisy,
            'stoi_dtln': stoi_dtln,
            'stoi_chained': stoi_chained,
            'pesq_noisy': pesq_noisy if pesq_noisy is not None else 'N/A',
            'pesq_dtln': pesq_dtln if pesq_dtln is not None else 'N/A',
            'pesq_chained': pesq_chained if pesq_chained is not None else 'N/A',
            'min_limiter_gain': np.min(gain_curve)
        }
        results.append(row)
        
        if (idx + 1) % 30 == 0 or (idx + 1) == len(dtln_files):
            print(f"  Processed {idx + 1}/{len(dtln_files)} files...")
            
    # Save CSV
    fieldnames = ['filename', 'input_snr_cond',
                  'si_snr_noisy', 'si_snr_dtln', 'si_snr_chained',
                  'stoi_noisy', 'stoi_dtln', 'stoi_chained',
                  'pesq_noisy', 'pesq_dtln', 'pesq_chained',
                  'min_limiter_gain']
    
    with open(csv_out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nFull per-file chained evaluation saved to: {csv_out_path}")
    
    # 3-Way Comparative Summary Table by Input SNR Level
    snr_levels = sorted(list(set(r['input_snr_cond'] for r in results)))
    
    print("\n" + "="*115)
    print("3-WAY COMPARISON BY SNR LEVEL: Noisy vs. DTLN Only vs. DTLN + Limiter")
    print("="*115)
    print(f"{'Input SNR':<10} | {'SI-SNR (dB) [Noisy / DTLN / Chained]':<38} | {'STOI [Noisy / DTLN / Chained]':<32} | {'PESQ [Noisy / DTLN / Chained]':<30}")
    print("-"*115)
    
    for snr_val in snr_levels:
        sub = [r for r in results if r['input_snr_cond'] == snr_val]
        
        sisnr_n = np.mean([r['si_snr_noisy'] for r in sub])
        sisnr_d = np.mean([r['si_snr_dtln'] for r in sub])
        sisnr_c = np.mean([r['si_snr_chained'] for r in sub])
        
        stoi_n = np.mean([r['stoi_noisy'] for r in sub])
        stoi_d = np.mean([r['stoi_dtln'] for r in sub])
        stoi_c = np.mean([r['stoi_chained'] for r in sub])
        
        pesq_n = np.mean([r['pesq_noisy'] for r in sub])
        pesq_d = np.mean([r['pesq_dtln'] for r in sub])
        pesq_c = np.mean([r['pesq_chained'] for r in sub])
        
        sisnr_str = f"{sisnr_n:>5.2f} / {sisnr_d:>5.2f} / {sisnr_c:>5.2f}"
        stoi_str  = f"{stoi_n:.4f} / {stoi_d:.4f} / {stoi_c:.4f}"
        pesq_str  = f"{pesq_n:.3f} / {pesq_d:.3f} / {pesq_c:.3f}"
        
        print(f"{snr_val:<10.1f}dB | {sisnr_str:<38} | {stoi_str:<32} | {pesq_str:<30}")
        
    print("-" * 115)
    o_sisnr_n = np.mean([r['si_snr_noisy'] for r in results])
    o_sisnr_d = np.mean([r['si_snr_dtln'] for r in results])
    o_sisnr_c = np.mean([r['si_snr_chained'] for r in results])
    
    o_stoi_n = np.mean([r['stoi_noisy'] for r in results])
    o_stoi_d = np.mean([r['stoi_dtln'] for r in results])
    o_stoi_c = np.mean([r['stoi_chained'] for r in results])
    
    o_pesq_n = np.mean([r['pesq_noisy'] for r in results])
    o_pesq_d = np.mean([r['pesq_dtln'] for r in results])
    o_pesq_c = np.mean([r['pesq_chained'] for r in results])
    
    o_sisnr_str = f"{o_sisnr_n:>5.2f} / {o_sisnr_d:>5.2f} / {o_sisnr_c:>5.2f}"
    o_stoi_str  = f"{o_stoi_n:.4f} / {o_stoi_d:.4f} / {o_stoi_c:.4f}"
    o_pesq_str  = f"{o_pesq_n:.3f} / {o_pesq_d:.3f} / {o_pesq_c:.3f}"
    
    print(f"{'OVERALL':<10} | {o_sisnr_str:<38} | {o_stoi_str:<32} | {o_pesq_str:<30}")
    print("="*115)
    
    # SNR Specific Impact Analysis
    print("\n=== Impact Analysis: Chained Limiter vs. DTLN Only ===")
    for snr_val in snr_levels:
        sub = [r for r in results if r['input_snr_cond'] == snr_val]
        dsisnr = np.mean([r['si_snr_chained'] for r in sub]) - np.mean([r['si_snr_dtln'] for r in sub])
        dstoi  = np.mean([r['stoi_chained'] for r in sub]) - np.mean([r['stoi_dtln'] for r in sub])
        dpesq  = np.mean([r['pesq_chained'] for r in sub]) - np.mean([r['pesq_dtln'] for r in sub])
        avg_min_g = np.mean([r['min_limiter_gain'] for r in sub])
        print(f"SNR {snr_val:+.0f} dB: SI-SNR {dsisnr:+0.2f} dB | STOI {dstoi:+0.4f} | PESQ {dpesq:+0.3f} | Avg Min Gain: {avg_min_g:.4f}")

if __name__ == '__main__':
    main()
