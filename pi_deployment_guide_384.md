# Raspberry Pi 5 — DTLN 384-Unit TFLite Deployment & Evaluation Guide

> **Target Architecture**: Raspberry Pi 5 (ARM64, 4GB/8GB RAM)  
> **Model Specs**: 384-Unit Stateful DTLN (`models_384_run/model_1.tflite` [4.15 MB] + `models_384_run/model_2.tflite` [5.15 MB], state shape `[1, 1, 384, 2]`).

---

## 🚀 Copy-Pasteable Deployment Runbook

### Step 1: Create Remote Directory on Raspberry Pi 5
```bash
ssh create@raspberrypi "mkdir -p ~/DTLN_Pi/models_384_run ~/DTLN_Pi/data_full/val_mix ~/DTLN_Pi/data_full/val_speech"
```

### Step 2: Transfer 384-Unit Models & Evaluation Script from PC
Run from your PC terminal (`d:\SIH 2026\DTLN`):
```bash
# Transfer 384-unit TFLite submodels
scp models_384_run/model_1.tflite create@raspberrypi:~/DTLN_Pi/models_384_run/
scp models_384_run/model_2.tflite create@raspberrypi:~/DTLN_Pi/models_384_run/

# Transfer evaluation script
scp evaluate_384_run.py create@raspberrypi:~/DTLN_Pi/

# Transfer validation audio files (if not already on Pi)
scp data_full/val_mix/*.wav create@raspberrypi:~/DTLN_Pi/data_full/val_mix/
scp data_full/val_speech/*.wav create@raspberrypi:~/DTLN_Pi/data_full/val_speech/
```

### Step 3: Connect to Raspberry Pi 5 & Activate Virtual Environment
```bash
ssh create@raspberrypi
cd ~/DTLN_Pi
source venv/bin/activate
```

### Step 4: Execute 384-Unit TFLite Evaluation & Latency Benchmark
```bash
python evaluate_384_run.py
```

### Step 5: Retrieve Results CSV back to PC
Run from your PC terminal:
```bash
scp create@raspberrypi:~/DTLN_Pi/data_full/eval_results_384_run.csv data_full/
```

---

## 📊 Latency & Real-Time Headroom Calculation

- **Frame Hop Size (`block_shift`)**: 128 samples @ 16 kHz = **8.0 ms budget**.
- **Execution Target**: Per-block latency $< 8.0\text{ ms}$ (RTF $< 1.0\text{x}$).
- **Measured PC Latency**: Evaluated live during PC-side evaluation run.
