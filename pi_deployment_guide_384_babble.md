# Raspberry Pi 5 — DTLN 384-Unit Babble-Tuned TFLite Deployment Guide

> **Target Architecture**: Raspberry Pi 5 (ARM64, 4GB/8GB RAM)  
> **Model Specs**: 384-Unit Stateful DTLN Babble Fine-Tuned (`models_384_babble_run/model_1.tflite` [4.15 MB] + `models_384_babble_run/model_2.tflite` [5.15 MB], state shape `[1, 1, 384, 2]`).

---

## 🚀 Copy-Pasteable Raspberry Pi 5 Runbook

### Step 1: Prepare Remote Directories on Raspberry Pi 5
Run from your PC terminal (`d:\SIH 2026\DTLN`):
```bash
ssh create@192.168.137.207 "mkdir -p ~/DTLN_Pi/models_384_babble_run ~/DTLN_Pi/data_full/val_mix ~/DTLN_Pi/data_full/val_speech"
```

### Step 2: Transfer 384-Unit Babble TFLite Submodels & Scripts to Pi
Run from your PC terminal (`d:\SIH 2026\DTLN`):
```bash
# 1. Transfer latest TFLite submodels
scp models_384_babble_run/model_1.tflite create@192.168.137.207:~/DTLN_Pi/models_384_babble_run/
scp models_384_babble_run/model_2.tflite create@192.168.137.207:~/DTLN_Pi/models_384_babble_run/

# 2. Transfer sanity check & evaluation scripts
scp sanity_check_tflite_384_babble.py create@192.168.137.207:~/DTLN_Pi/
scp eval_matched120.py create@192.168.137.207:~/DTLN_Pi/

# 3. Transfer exact 120 validation audio pairs (no 20/25dB files)
scp validation_120.zip create@192.168.137.207:~/DTLN_Pi/
ssh create@192.168.137.207 "cd ~/DTLN_Pi/data_full && unzip -o ../validation_120.zip"
```

### Step 3: Run Fast Sanity Check & Benchmark on Pi 5
Connect to Raspberry Pi 5 and execute the sanity check script:
```bash
ssh create@192.168.137.207
cd ~/DTLN_Pi
source venv/bin/activate

# Execute sanity check & latency verification
python sanity_check_tflite_384_babble.py
```

### Step 4: Run Full Evaluation on Pi 5 (Matched 120-File Set)
```bash
# Execute evaluation on the baseline 120 validation files
python eval_matched120.py
```

### Step 5: Retrieve Results CSV Back to PC
Run from your PC terminal:
```bash
scp create@192.168.137.207:~/DTLN_Pi/data_full/eval_results_384_babble_run_matched120.csv data_full/eval_results_pi5_384_babble.csv
```

---

## 📊 Latency & Real-Time Performance Target on Pi 5

- **Frame Hop Size (`block_shift`)**: 128 samples @ 16 kHz = **8.0 ms real-time budget**.
- **Measured PC Latency**: **1.878 ms / block** (RTF: **0.235x**).
- **Pi 5 Expected Latency**: $< 2.5\text{ ms / block}$ ($> 3\text{x}$ real-time headroom).
