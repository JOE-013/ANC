"""
Core utilities for generating noisy/clean speech pairs at target SNR levels.

Key idea:
    Given a clean speech signal and a noise signal, we want to scale the
    noise so that when added to speech, the resulting mixture has a
    specific Signal-to-Noise Ratio (SNR) in dB.

SNR formula:
    SNR(dB) = 10 * log10( power(signal) / power(noise) )

    To hit a TARGET SNR, we solve for a scaling factor `k` applied to noise:
    k = sqrt( power(signal) / (power(noise) * 10^(target_snr/10)) )
"""

import numpy as np
import librosa
import soundfile as sf
import random
import os

SAMPLE_RATE = 16000  # standard for speech models (Whisper, wav2vec, DTLN, etc. all use 16k)


def load_audio(path, sr=SAMPLE_RATE):
    """Load an audio file, resample to sr, force mono."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def rms(signal):
    """Root-mean-square power of a signal."""
    return np.sqrt(np.mean(signal ** 2) + 1e-12)


def fit_noise_to_length(noise, target_len):
    """
    Make noise exactly target_len samples long.
    - If noise is shorter: loop it (tile) to cover the length.
    - If noise is longer: pick a random contiguous crop.
    This matters because gunshot/siren clips are often short bursts,
    while speech clips vary in length.
    """
    if len(noise) < target_len:
        repeats = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeats)
    start = random.randint(0, len(noise) - target_len)
    return noise[start:start + target_len]


def mix_at_snr(clean, noise, target_snr_db):
    """
    Mix clean speech and noise so the result has target_snr_db.

    Returns:
        noisy: the mixed signal (model INPUT)
        clean: the original clean signal, unchanged (model TARGET/label)
    """
    noise = fit_noise_to_length(noise, len(clean))

    clean_power = rms(clean)
    noise_power = rms(noise)

    # scaling factor to hit the target SNR
    target_factor = clean_power / (noise_power * (10 ** (target_snr_db / 20)))
    noise_scaled = noise * target_factor

    noisy = clean + noise_scaled

    # prevent clipping (values must stay within [-1, 1] for a valid wav)
    peak = np.max(np.abs(noisy))
    if peak > 1.0:
        noisy = noisy / peak
        clean_out = clean / peak  # scale clean the same way to keep them aligned
    else:
        clean_out = clean

    return noisy.astype(np.float32), clean_out.astype(np.float32)


def save_pair(noisy, clean, out_dir_noisy, out_dir_clean, filename, sr=SAMPLE_RATE):
    os.makedirs(out_dir_noisy, exist_ok=True)
    os.makedirs(out_dir_clean, exist_ok=True)
    sf.write(os.path.join(out_dir_noisy, filename), noisy, sr)
    sf.write(os.path.join(out_dir_clean, filename), clean, sr)
