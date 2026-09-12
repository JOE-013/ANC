# Raspberry Pi 5 DTLN TFLite Deployment & Evaluation Guide

This guide provides the **exact, copy-pasteable step-by-step CLI commands** to deploy and benchmark the unquantized Float32 DTLN TFLite model on a Raspberry Pi 5 (ARM64).

---

## Required Transfer File Manifest

Ensure the following files are copied from your PC workspace (`d:\SIH 2026\DTLN`) to the Raspberry Pi 5:

```
DTLN_Pi/
├── models_full_run/
│   ├── model_1.tflite          # Stage 1 frequency-domain separation submodel (387 KB)
│   └── model_2.tflite          # Stage 2 time-domain separation submodel (1.39 MB)
├── data_full/
│   ├── val_mix/                # 120 held-out noisy test WAV files
│   └── val_speech/             # 120 held-out clean target WAV files
├── DTLN_model.py               # Model definition module
├── sanity_check_tflite.py      # Quick 10-file numerical verification script
├── evaluate_tflite_full_run.py # Full 120-file evaluation & per-block latency benchmark script
└── requirements_pi.txt         # Lightweight Python dependencies
```

---

## Step-by-Step Command Execution Checklist

### Step 1: Transfer Files to Raspberry Pi 5 (Run on PC PowerShell)

Replace `pi_user` and `192.168.1.XXX` with your Pi's username and IP address:

```powershell
# 1. Create target directory on Pi
ssh pi_user@192.168.1.XXX "mkdir -p ~/DTLN_Pi/models_full_run ~/DTLN_Pi/data_full/val_mix ~/DTLN_Pi/data_full/val_speech"

# 2. Copy TFLite models and Python scripts
scp "d:\SIH 2026\DTLN\models_full_run\model_1.tflite" "d:\SIH 2026\DTLN\models_full_run\model_2.tflite" pi_user@192.168.1.XXX:~/DTLN_Pi/models_full_run/
scp "d:\SIH 2026\DTLN\DTLN_model.py" "d:\SIH 2026\DTLN\sanity_check_tflite.py" "d:\SIH 2026\DTLN\evaluate_tflite_full_run.py" "d:\SIH 2026\DTLN\requirements_pi.txt" pi_user@192.168.1.XXX:~/DTLN_Pi/

# 3. Copy validation dataset audio files
scp "d:\SIH 2026\DTLN\data_full\val_mix\*.wav" pi_user@192.168.1.XXX:~/DTLN_Pi/data_full/val_mix/
scp "d:\SIH 2026\DTLN\data_full\val_speech\*.wav" pi_user@192.168.1.XXX:~/DTLN_Pi/data_full/val_speech/
```

*Alternative (USB Flash Drive): Copy the `DTLN_Pi` folder to a FAT32/exFAT USB drive and copy it to `~/DTLN_Pi` on the Pi.*

---

### Step 2: Install System Prerequisites on Raspberry Pi OS ARM64 (Run on Pi Terminal)

The `pesq` package compiles a C extension during installation, requiring `build-essential` and `python3-dev`. `soundfile` requires `libsndfile1`.

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-venv \
    python3-pip \
    libsndfile1 \
    libsndfile1-dev \
    libatlas-base-dev \
    gfortran
```

---

### Step 3: Set Up Python Virtual Environment & Dependencies (Run on Pi Terminal)

```bash
# Navigate to project directory
cd ~/DTLN_Pi

# Create clean virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade build tools
pip install --upgrade pip setuptools wheel

# Install runtime dependencies
pip install -r requirements_pi.txt
```

*Note: If `tflite-runtime` wheel is unavailable for your specific Python version on Debian/Raspberry Pi OS, install `tensorflow` or `ai-edge-litert`:*
```bash
pip install tflite-runtime || pip install tensorflow
```

---

### Step 4: Run Numerical Sanity Check (Run on Pi Terminal)

Run the 10-file verification script to confirm model loading, LiteRT/TFLite execution, and tensor state passing:

```bash
python3 sanity_check_tflite.py
```

- **Expected Runtime**: ~1–3 seconds total.
- **Expected Output**: Confirmation table displaying `SI-SNR (dB)` per file and `Average SI-SNR (~9.53 dB)`.

---

### Step 5: Run Full 120-File Evaluation & Latency Instrumentation (Run on Pi Terminal)

Run full evaluation over all 120 held-out validation pairs:

```bash
python3 evaluate_tflite_full_run.py
```

- **Expected Runtime**: ~30–90 seconds total for 120 files (depending on PESQ metric calculation overhead).
- **Expected Latency Results on Pi 5 (Cortex-A76)**:
  - Estimated per-block latency: **~0.3–0.8 ms / block** (well within the 8.0 ms real-time hop budget).
  - Estimated Real-Time Factor (RTF): **~0.04x–0.10x** (10x–25x faster than real-time).
- **Output CSV Location**: `~/DTLN_Pi/data_full/eval_results_tflite_full_run.csv`

---

### Step 6: Retrieve Results CSV Back to PC (Run on PC PowerShell)

Copy the Pi's evaluation results back to your PC workspace for direct comparison:

```powershell
scp pi_user@192.168.1.XXX:~/DTLN_Pi/data_full/eval_results_tflite_full_run.csv "d:\SIH 2026\DTLN\data_full\eval_results_pi5_tflite.csv"
```

---

## Troubleshooting & System Dep Flags Summary

| Package | Reason Flagged | Fix / Resolution |
| :--- | :--- | :--- |
| `pesq` | Contains C extension requiring compilation | Install `build-essential` and `python3-dev` prior to `pip install` |
| `soundfile` | Requires C `libsndfile` library | Install `libsndfile1` via `apt-get` |
| `tflite-runtime` | Lightweight interpreter without full TF overhead | Preferred over `tensorflow` for embedded Pi 5 deployment |
| `pystoi` | NumPy/SciPy dependency for intelligibility metric | Installed via standard `pip` wheel |
