#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_full_run_with_limiter_thresh3.py - Evaluation script for DTLN + Transient Limiter at threshold_db = -3.0 dBFS.
Runs on all 120 validation files, saves outputs to data_full/enhanced_full_run_with_limiter_thresh3/,
writes data_full/eval_results_full_run_with_limiter_thresh3.csv, and prints 3-way comparative metrics
against DTLN-only and the previous -6.0 dBFS limiter.
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
    print("=== DTLN + Transient Limiter Evaluation (threshold_db = -3.0 dBFS) ===")
    
    dtln_enhanced_dir = r"d:\SIH 2026\DTLN\data_full\enhanced_full_run"
    clean_dir = r"d:\SIH 2026\DTLN\data_full\val_speech"
    noisy_dir = r"d:\SIH 2026\DTLN\data_full\val_mix"
    
    chained_output_dir = r"d:\SIH 2026\DTLN\data_full\enhanced_full_run_with_limiter_thresh3"
    csv_out_path = r"d:\SIH 2026\DTLN\data_full\eval_results_full_run_with_limiter_thresh3.csv"
    csv_thresh6_path = r"d:\SIH 2026\DTLN\data_full\eval_results_full_run_with_limiter.csv"
    
    os.makedirs(chained_output_dir, exist_ok=True)
    
    dtln_files = sorted(glob.glob(os.path.join(dtln_enhanced_dir, "*.wav")))
    print(f"Found {len(dtln_files)} DTLN-enhanced audio files in {dtln_enhanced_dir}")
    
    limiter_params = {
        'threshold_db': -3.0,
        'attack_ms': 2.0,
        'release_ms': 50.0,
        'win_ms': 3.0
    }
    
    print(f"Limiter Parameters: {limiter_params}\n")
    print("Applying transient limiter (threshold_db = -3.0) to DTLN outputs & computing metrics...")
    
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
        
        # Apply transient limiter with threshold_db = -3.0
        lim_sig, gain_curve = apply_transient_limiter(
            dtln_sig,
            fs=fs,
            threshold_db=limiter_params['threshold_db'],
            attack_ms=limiter_params['attack_ms'],
            release_ms=limiter_params['release_ms'],
            win_ms=limiter_params['win_ms']
        )
        
        sf.write(chained_path, lim_sig, fs)
        
        # Compute metrics
        si_snr_chained, stoi_chained, pesq_chained = compute_metrics(clean_sig, lim_sig, fs)
        si_snr_noisy, stoi_noisy, pesq_noisy = compute_metrics(clean_sig, noisy_sig, fs)
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
        
    print(f"\nFull per-file chained evaluation (-3.0 dBFS) saved to: {csv_out_path}")
    
    # 3-Way Comparative Summary Table by Input SNR Level (-3.0 dBFS)
    snr_levels = sorted(list(set(r['input_snr_cond'] for r in results)))
    
    print("\n" + "="*115)
    print("3-WAY COMPARISON BY SNR LEVEL (Threshold = -3.0 dBFS): Noisy vs. DTLN Only vs. DTLN + Limiter (-3dB)")
    print("="*115)
    print(f"{'Input SNR':<10} | {'SI-SNR (dB) [Noisy / DTLN / Limiter-3dB]':<40} | {'STOI [Noisy / DTLN / Limiter-3dB]':<32} | {'PESQ [Noisy / DTLN / Limiter-3dB]':<30}")
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
        
        print(f"{snr_val:<10.1f}dB | {sisnr_str:<40} | {stoi_str:<32} | {pesq_str:<30}")
        
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
    
    print(f"{'OVERALL':<10} | {o_sisnr_str:<40} | {o_stoi_str:<32} | {o_pesq_str:<30}")
    print("="*115)

    # 4-Way Comparative Impact Analysis Table: DTLN-only vs -6.0 dBFS vs -3.0 dBFS
    rows_thresh6 = []
    if os.path.exists(csv_thresh6_path):
        with open(csv_thresh6_path, 'r', newline='') as f:
            rows_thresh6 = list(csv.DictReader(f))
            
    print("\n" + "="*115)
    print("DIRECT COMPARISON OF LIMITER THRESHOLDS (-6.0 dBFS vs. -3.0 dBFS vs. DTLN Only)")
    print("="*115)
    print(f"{'Input SNR':<10} | {'SI-SNR: DTLN / -6dB / -3dB':<32} | {'STOI: DTLN / -6dB / -3dB':<30} | {'PESQ: DTLN / -6dB / -3dB':<28}")
    print("-"*115)
    
    for snr_val in snr_levels:
        sub_3 = [r for r in results if r['input_snr_cond'] == snr_val]
        sub_6 = [r for r in rows_thresh6 if float(r['input_snr_cond']) == snr_val] if rows_thresh6 else []
        
        sd = np.mean([r['si_snr_dtln'] for r in sub_3])
        s6 = np.mean([float(r['si_snr_chained']) for r in sub_6]) if sub_6 else 0.0
        s3 = np.mean([r['si_snr_chained'] for r in sub_3])
        
        td = np.mean([r['stoi_dtln'] for r in sub_3])
        t6 = np.mean([float(r['stoi_chained']) for r in sub_6]) if sub_6 else 0.0
        t3 = np.mean([r['stoi_chained'] for r in sub_3])
        
        pd = np.mean([r['pesq_dtln'] for r in sub_3])
        p6 = np.mean([float(r['pesq_chained']) for r in sub_6]) if sub_6 else 0.0
        p3 = np.mean([r['pesq_chained'] for r in sub_3])
        
        print(f"{snr_val:<10.1f}dB | {sd:>5.2f} / {s6:>5.2f} / {s3:>5.2f}          | {td:.4f} / {t6:.4f} / {t3:.4f}        | {pd:.3f} / {p6:.3f} / {p3:.3f}")
        
    print("-" * 115)
    o_sd = np.mean([r['si_snr_dtln'] for r in results])
    o_s6 = np.mean([float(r['si_snr_chained']) for r in rows_thresh6]) if rows_thresh6 else 0.0
    o_s3 = np.mean([r['si_snr_chained'] for r in results])
    
    o_td = np.mean([r['stoi_dtln'] for r in results])
    o_t6 = np.mean([float(r['stoi_chained']) for r in rows_thresh6]) if rows_thresh6 else 0.0
    o_t3 = np.mean([r['stoi_chained'] for r in results])
    
    o_pd = np.mean([r['pesq_dtln'] for r in results])
    o_p6 = np.mean([float(r['pesq_chained']) for r in rows_thresh6]) if rows_thresh6 else 0.0
    o_p3 = np.mean([r['pesq_chained'] for r in results])
    
    print(f"{'OVERALL':<10} | {o_sd:>5.2f} / {o_s6:>5.2f} / {o_s3:>5.2f}          | {o_td:.4f} / {o_t6:.4f} / {o_t3:.4f}        | {o_pd:.3f} / {o_p6:.3f} / {o_p3:.3f}")
    print("="*115)

if __name__ == '__main__':
    main()
