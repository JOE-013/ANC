"""
Day 1 main script: builds the noisy/clean paired dataset.

Directory expectations (adjust paths below to match your machine):
    raw_data/LibriSpeech/dev-clean/**/*.flac   <- clean speech
    raw_data/musan/noise/**/*.wav              <- generic noise
    raw_data/defence_noise/*.wav               <- your manually sourced gunshot/rotor/siren clips

Output:
    dataset/noisy/*.wav
    dataset/clean/*.wav
    dataset/manifest.csv   <- log of every pair generated (clean_file, noise_file, snr_db)
"""

import os
import glob
import random
import csv
from mix_utils import load_audio, mix_at_snr, save_pair, SAMPLE_RATE

# ---- CONFIG ----
CLEAN_ROOT = "raw_data/LibriSpeech/dev-clean"
NOISE_ROOTS = ["raw_data/musan/noise", "raw_data/defence_noise"]
OUT_NOISY = "dataset/noisy"
OUT_CLEAN = "dataset/clean"
MANIFEST_PATH = "dataset/manifest.csv"

N_PAIRS = 400          # "a few hundred pairs" per the roadmap
SNR_RANGE_DB = (-5, 15)  # covers hard (gunshot at -5dB) to mild (wind at 15dB) cases
MIN_DURATION_SEC = 2
MAX_DURATION_SEC = 6
SEED = 42
# ----------------

random.seed(SEED)


def find_clean_files():
    files = glob.glob(os.path.join(CLEAN_ROOT, "**", "*.flac"), recursive=True)
    print(f"Found {len(files)} clean speech files")
    return files


def find_noise_files():
    files = []
    for root in NOISE_ROOTS:
        found = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
        print(f"Found {len(found)} noise files in {root}")
        files.extend(found)
    return files


def main():
    clean_files = find_clean_files()
    noise_files = find_noise_files()

    if not clean_files or not noise_files:
        print("\n[!] No files found. Did you run download_data.sh and add defence clips?")
        print("    Check CLEAN_ROOT / NOISE_ROOTS paths at the top of this script.")
        return

    os.makedirs("dataset", exist_ok=True)
    manifest_rows = []

    for i in range(N_PAIRS):
        clean_path = random.choice(clean_files)
        noise_path = random.choice(noise_files)
        target_snr = random.uniform(*SNR_RANGE_DB)

        clean = load_audio(clean_path)
        noise = load_audio(noise_path)

        # trim/pad clean speech to a random duration within range
        target_len = int(random.uniform(MIN_DURATION_SEC, MAX_DURATION_SEC) * SAMPLE_RATE)
        if len(clean) > target_len:
            start = random.randint(0, len(clean) - target_len)
            clean = clean[start:start + target_len]
        elif len(clean) < target_len:
            # skip clips too short rather than padding with silence (bad training signal)
            continue

        noisy, clean_out = mix_at_snr(clean, noise, target_snr)

        filename = f"pair_{i:04d}.wav"
        save_pair(noisy, clean_out, OUT_NOISY, OUT_CLEAN, filename)

        manifest_rows.append({
            "filename": filename,
            "clean_source": clean_path,
            "noise_source": noise_path,
            "snr_db": round(target_snr, 2),
            "duration_sec": round(len(clean_out) / SAMPLE_RATE, 2),
        })

        if (i + 1) % 50 == 0:
            print(f"  generated {i+1}/{N_PAIRS} pairs...")

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows)} pairs written to dataset/")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
