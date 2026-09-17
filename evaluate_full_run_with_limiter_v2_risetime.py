#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_full_run_with_limiter_v2_risetime.py — Full 120-file evaluation of DTLN + Rise-Time Transient Limiter (v2).
Applies transient_limiter_v2.py (with gain attenuation fix) to all 120 DTLN validation outputs.
Saves data_full/eval_results_full_run_with_limiter_v2_risetime.csv and outputs 5-way comparative metrics across all limiter variants.
"""

import os
import glob
import csv
import re
import numpy as np
import soundfile as sf
import librosa
from transient_limiter_v2 import apply_transient_limiter_v2
from pystoi import stoi

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

def count_discrete_events(mask, min_gap_samples=160):
    if not np.any(mask):
        return 0
    trig_indices = np.where(mask)[0]
    gaps = np.diff(trig_indices)
    return int(1 + np.sum(gaps > min_gap_samples))

def compute_si_snr(estimate, target):
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

def main():
    print("=== DTLN + Rise-Time Transient Limiter (v2 Fixed) Evaluation (Held-Out 120-File Validation Set) ===")
    
    dtln_enhanced_dir = r"d:\SIH 2026\DTLN\data_full\enhanced_full_run"
    clean_dir = r"d:\SIH 2026\DTLN\data_full\val_speech"
    noisy_dir = r"d:\SIH 2026\DTLN\data_full\val_mix"
    
    chained_output_dir = r"d:\SIH 2026\DTLN\data_full\enhanced_full_run_with_limiter_v2_risetime"
    csv_out_path = r"d:\SIH 2026\DTLN\data_full\eval_results_full_run_with_limiter_v2_risetime.csv"
    
    csv_thresh6_path = r"d:\SIH 2026\DTLN\data_full\eval_results_full_run_with_limiter.csv"
    csv_thresh3_path = r"d:\SIH 2026\DTLN\data_full\eval_results_full_run_with_limiter_thresh3.csv"
    csv_relative_path = r"d:\SIH 2026\DTLN\data_full\eval_results_full_run_with_limiter_relative.csv"
    
    os.makedirs(chained_output_dir, exist_ok=True)
    
    dtln_files = sorted(glob.glob(os.path.join(dtln_enhanced_dir, "*.wav")))
    print(f"Found {len(dtln_files)} DTLN-enhanced audio files in {dtln_enhanced_dir}")
    
    limiter_params = {
        'smooth_win_ms': 1.0,
        'rise_time_threshold_ms': 1.5,
        'min_level_db': -12.0,
        'target_ceiling_db': -6.0,
        'attenuation_db': 6.0,
        'attack_ms': 2.0,
        'release_ms': 50.0
    }
    
    print(f"Rise-Time Limiter V2 Parameters: {limiter_params}\n")
    print("Applying fixed rise-time transient limiter to DTLN outputs & computing metrics...")
    
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
        
        # Apply rise-time transient limiter v2
        lim_sig, gain_curve, mask = apply_transient_limiter_v2(
            dtln_sig,
            fs=fs,
            smooth_win_ms=limiter_params['smooth_win_ms'],
            rise_time_threshold_ms=limiter_params['rise_time_threshold_ms'],
            min_level_db=limiter_params['min_level_db'],
            target_ceiling_db=limiter_params['target_ceiling_db'],
            attenuation_db=limiter_params['attenuation_db'],
            attack_ms=limiter_params['attack_ms'],
            release_ms=limiter_params['release_ms']
        )
        
        sf.write(chained_path, lim_sig, fs)
        
        # Compute metrics
        si_snr_chained, stoi_chained, pesq_chained = compute_metrics(clean_sig, lim_sig, fs)
        si_snr_noisy, stoi_noisy, pesq_noisy = compute_metrics(clean_sig, noisy_sig, fs)
        si_snr_dtln, stoi_dtln, pesq_dtln = compute_metrics(clean_sig, dtln_sig, fs)
        
        min_g = float(np.min(gain_curve))
        num_trig_samples = int(np.sum(mask))
        num_events = count_discrete_events(mask)
        triggered = bool(num_trig_samples > 0)
        
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
            'min_limiter_gain': min_g,
            'num_trig_samples': num_trig_samples,
            'num_events': num_events,
            'triggered': triggered
        }
        results.append(row)
        
        if (idx + 1) % 30 == 0 or (idx + 1) == len(dtln_files):
            print(f"  Processed {idx + 1}/{len(dtln_files)} files...")
            
    # Save CSV
    fieldnames = ['filename', 'input_snr_cond',
                  'si_snr_noisy', 'si_snr_dtln', 'si_snr_chained',
                  'stoi_noisy', 'stoi_dtln', 'stoi_chained',
                  'pesq_noisy', 'pesq_dtln', 'pesq_chained',
                  'min_limiter_gain', 'num_trig_samples', 'num_events', 'triggered']
    
    with open(csv_out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nFull per-file evaluation saved to: {csv_out_path}")
    
    # 3-File Verification Demonstration
    print("\n" + "="*115)
    print("EXPLICIT VERIFICATION DEMONSTRATION (First 3 Triggered Files)")
    print("="*115)
    trig_files = [r for r in results if r['triggered'] and r['min_limiter_gain'] < 0.99][:3]
    for tf in trig_files:
        fn = tf['filename']
        mg = tf['min_limiter_gain']
        snr_d = tf['si_snr_dtln']
        snr_c = tf['si_snr_chained']
        stoi_d = tf['stoi_dtln']
        stoi_c = tf['stoi_chained']
        pesq_d = tf['pesq_dtln']
        pesq_c = tf['pesq_chained']
        print(f"File: {fn}")
        print(f"  Min Limiter Gain: {mg:.4f} (Meaningfully < 1.0)")
        print(f"  SI-SNR: DTLN = {snr_d:.4f} dB -> Chained = {snr_c:.4f} dB (Diff = {snr_c - snr_d:+.4f} dB)")
        print(f"  STOI:   DTLN = {stoi_d:.4f}    -> Chained = {stoi_c:.4f}    (Diff = {stoi_c - stoi_d:+.4f})")
        print(f"  PESQ:   DTLN = {pesq_d:.4f}    -> Chained = {pesq_c:.4f}    (Diff = {pesq_c - pesq_d:+.4f})\n")
    print("="*115)

    # Trigger Statistics
    snr_levels = sorted(list(set(r['input_snr_cond'] for r in results)))
    print("\n" + "="*95)
    print("Rise-Time Limiter V2 Triggering Statistics by SNR Level")
    print("="*95)
    print(f"{'Input SNR':<10} | {'Triggered Files / Total':<25} | {'Avg Events / File':<20} | {'Avg Min Gain':<15}")
    print("-"*95)
    for snr_val in snr_levels:
        sub = [r for r in results if r['input_snr_cond'] == snr_val]
        trig_cnt = sum(1 for r in sub if r['triggered'])
        avg_ev = np.mean([r['num_events'] for r in sub])
        avg_g = np.mean([r['min_limiter_gain'] for r in sub])
        print(f"{snr_val:<10.1f}dB | {trig_cnt:>2}/{len(sub):<2}                     | {avg_ev:<20.2f} | {avg_g:<15.4f}")
    print("="*95)
    
    # Load previous evaluation results for 5-Way Comparison
    rows_thresh6 = list(csv.DictReader(open(csv_thresh6_path))) if os.path.exists(csv_thresh6_path) else []
    rows_thresh3 = list(csv.DictReader(open(csv_thresh3_path))) if os.path.exists(csv_thresh3_path) else []
    rows_relative = list(csv.DictReader(open(csv_relative_path))) if os.path.exists(csv_relative_path) else []
    
    print("\n" + "="*145)
    print("5-WAY COMPREHENSIVE COMPARISON: DTLN Only vs. Static -6dB vs. Static -3dB vs. Relative vs. Rise-Time (v2)")
    print("="*145)
    print(f"{'Input SNR':<10} | {'SI-SNR (dB): DTLN / -6dB / -3dB / Rel / RiseTime':<45} | {'STOI: DTLN / -6dB / -3dB / Rel / RiseTime':<42} | {'PESQ: DTLN / -6dB / -3dB / Rel / RiseTime':<40}")
    print("-"*145)
    
    for snr_val in snr_levels:
        sub_v2 = [r for r in results if r['input_snr_cond'] == snr_val]
        sub_t6 = [r for r in rows_thresh6 if float(r['input_snr_cond']) == snr_val] if rows_thresh6 else []
        sub_t3 = [r for r in rows_thresh3 if float(r['input_snr_cond']) == snr_val] if rows_thresh3 else []
        sub_rel = [r for r in rows_relative if float(r['input_snr_cond']) == snr_val] if rows_relative else []
        
        sd = np.mean([r['si_snr_dtln'] for r in sub_v2])
        s6 = np.mean([float(r['si_snr_chained']) for r in sub_t6]) if sub_t6 else 0.0
        s3 = np.mean([float(r['si_snr_chained']) for r in sub_t3]) if sub_t3 else 0.0
        sr = np.mean([float(r['si_snr_chained']) for r in sub_rel]) if sub_rel else 0.0
        sv2 = np.mean([r['si_snr_chained'] for r in sub_v2])
        
        td = np.mean([r['stoi_dtln'] for r in sub_v2])
        t6 = np.mean([float(r['stoi_chained']) for r in sub_t6]) if sub_t6 else 0.0
        t3 = np.mean([float(r['stoi_chained']) for r in sub_t3]) if sub_t3 else 0.0
        tr = np.mean([float(r['stoi_chained']) for r in sub_rel]) if sub_rel else 0.0
        tv2 = np.mean([r['stoi_chained'] for r in sub_v2])
        
        pd = np.mean([r['pesq_dtln'] for r in sub_v2])
        p6 = np.mean([float(r['pesq_chained']) for r in sub_t6]) if sub_t6 else 0.0
        p3 = np.mean([float(r['pesq_chained']) for r in sub_t3]) if sub_t3 else 0.0
        pr = np.mean([float(r['pesq_chained']) for r in sub_rel]) if sub_rel else 0.0
        pv2 = np.mean([r['pesq_chained'] for r in sub_v2])
        
        sisnr_str = f"{sd:>5.2f} / {s6:>5.2f} / {s3:>5.2f} / {sr:>5.2f} / {sv2:>5.2f}"
        stoi_str  = f"{td:.4f} / {t6:.4f} / {t3:.4f} / {tr:.4f} / {tv2:.4f}"
        pesq_str  = f"{pd:.3f} / {p6:.3f} / {p3:.3f} / {pr:.3f} / {pv2:.3f}"
        
        print(f"{snr_val:<10.1f}dB | {sisnr_str:<45} | {stoi_str:<42} | {pesq_str:<40}")
        
    print("-" * 145)
    o_sd = np.mean([r['si_snr_dtln'] for r in results])
    o_s6 = np.mean([float(r['si_snr_chained']) for r in rows_thresh6]) if rows_thresh6 else 0.0
    o_s3 = np.mean([float(r['si_snr_chained']) for r in rows_thresh3]) if rows_thresh3 else 0.0
    o_sr = np.mean([float(r['si_snr_chained']) for r in rows_relative]) if rows_relative else 0.0
    o_sv2 = np.mean([r['si_snr_chained'] for r in results])
    
    o_td = np.mean([r['stoi_dtln'] for r in results])
    o_t6 = np.mean([float(r['stoi_chained']) for r in rows_thresh6]) if rows_thresh6 else 0.0
    o_t3 = np.mean([float(r['stoi_chained']) for r in rows_thresh3]) if rows_thresh3 else 0.0
    o_tr = np.mean([float(r['stoi_chained']) for r in rows_relative]) if rows_relative else 0.0
    o_tv2 = np.mean([r['stoi_chained'] for r in results])
    
    o_pd = np.mean([r['pesq_dtln'] for r in results])
    o_p6 = np.mean([float(r['pesq_chained']) for r in rows_thresh6]) if rows_thresh6 else 0.0
    o_p3 = np.mean([float(r['pesq_chained']) for r in rows_thresh3]) if rows_thresh3 else 0.0
    o_pr = np.mean([float(r['pesq_chained']) for r in rows_relative]) if rows_relative else 0.0
    o_pv2 = np.mean([r['pesq_chained'] for r in results])
    
    o_sisnr_str = f"{o_sd:>5.2f} / {o_s6:>5.2f} / {o_s3:>5.2f} / {o_sr:>5.2f} / {o_sv2:>5.2f}"
    o_stoi_str  = f"{o_td:.4f} / {o_t6:.4f} / {o_t3:.4f} / {o_tr:.4f} / {o_tv2:.4f}"
    o_pesq_str  = f"{o_pd:.3f} / {o_p6:.3f} / {o_p3:.3f} / {o_pr:.3f} / {o_pv2:.3f}"
    
    print(f"{'OVERALL':<10} | {o_sisnr_str:<45} | {o_stoi_str:<42} | {o_pesq_str:<40}")
    print("="*145)

if __name__ == '__main__':
    main()
