# Raspberry Pi 5 — DTLN Scaled (256-unit) TFLite Deployment Guide

> **Prerequisites already met**: The previous deployment for the 64-unit model
> (from `pi_deployment_guide.md`) is complete. The Pi already has:
> - `~/DTLN_Pi/` directory with Python venv activated
> - All system packages installed (`libsndfile1`, `build-essential`, etc.)
> - `tflite-runtime` (or equivalent) installed in the venv
> - The validation dataset already transferred (`data_full/val_mix/`, `data_full/val_speech/`)
>
> This guide covers **only the incremental steps**: transfer the new TFLite models
> and new eval script, then run and retrieve results.

---

## What Is New vs the Previous Deployment

| Item | Previous (64-unit) | This Run (256-unit, Scaled) |
|------|--------------------|-----------------------------|
| Weights | `full_run.weights.h5` | `full_run_256.weights.h5` |
| TFLite model 1 | `models_full_run/model_1.tflite` (387 KB) | `models_scaled_run/model_1.tflite` (2.27 MB) |
| TFLite model 2 | `models_full_run/model_2.tflite` (1.39 MB) | `models_scaled_run/model_2.tflite` (3.27 MB) |
| Eval script | `evaluate_tflite_full_run.py` | `evaluate_tflite_scaled_run.py` |
| Results CSV | `eval_results_tflite_full_run.csv` | `eval_results_tflite_scaled_run.csv` |
| Validation files | 120 files (5 SNR bands) | 630 files (7 SNR bands, 90/band) |
| Expected latency | 0.208 ms/block (measured) | Higher — see latency budget below |

**Latency budget reminder**: The 8.0 ms real-time hop budget is per 128-sample block.  
The 256-unit model has ~2× the LSTM operations of the 64-unit model.  
Expected range on Raspberry Pi 5 (Cortex-A76): **~0.4–1.0 ms/block** (still well within budget).

---

## File Manifest — New Files to Transfer

Only these files need to be copied (old files stay untouched on the Pi):

```
PC: d:\SIH 2026\DTLN\
├── models_scaled_run/
│   ├── model_1.tflite          <-- NEW Stage 1 (256-unit frequency-domain)
│   └── model_2.tflite          <-- NEW Stage 2 (256-unit time-domain)
├── evaluate_tflite_scaled_run.py   <-- NEW full-evaluation + latency script
└── sanity_check_tflite_scaled.py   <-- NEW sanity check (optional on Pi)
```

---

## Step-by-Step Commands

Replace `pi_user` and `192.168.1.XXX` with your Pi's username and IP address throughout.

---

### Step 1: Create New Directories on Pi  *(Run on PC PowerShell)*

```powershell
ssh pi_user@192.168.1.XXX "mkdir -p ~/DTLN_Pi/models_scaled_run ~/DTLN_Pi/data_full"
```

---

### Step 2: Transfer New TFLite Models and Eval Script  *(Run on PC PowerShell)*

```powershell
# Transfer new TFLite submodels
scp "d:\SIH 2026\DTLN\models_scaled_run\model_1.tflite" `
    "d:\SIH 2026\DTLN\models_scaled_run\model_2.tflite" `
    pi_user@192.168.1.XXX:~/DTLN_Pi/models_scaled_run/

# Transfer new evaluation scripts
scp "d:\SIH 2026\DTLN\evaluate_tflite_scaled_run.py" `
    "d:\SIH 2026\DTLN\sanity_check_tflite_scaled.py" `
    pi_user@192.168.1.XXX:~/DTLN_Pi/
```

> **Note**: If you are using a USB drive instead of SCP, copy `models_scaled_run/` and the two new `.py` files into the existing `DTLN_Pi/` folder on the drive, then copy across to the Pi.

---

### Step 3: Verify Transfer  *(Run on Pi Terminal)*

```bash
ls -lh ~/DTLN_Pi/models_scaled_run/
# Expected: model_1.tflite and model_2.tflite both present, sizes larger than the old 64-unit ones

ls ~/DTLN_Pi/evaluate_tflite_scaled_run.py
# Expected: file exists, no error
```

---

### Step 4: Activate the Existing Virtual Environment  *(Run on Pi Terminal)*

```bash
cd ~/DTLN_Pi
source venv/bin/activate
```

---

### Step 5: Run Sanity Check (Optional but Recommended)  *(Run on Pi Terminal)*

This is a quick 10-file numerical check. **No H5 reference model is available on the Pi**, so on the Pi this script will only test TFLite inference completes without errors and produces non-zero output. The full TFLite-vs-H5 SI-SNR comparison was already run on PC (should be ~35-40+ dB).

```bash
python3 sanity_check_tflite_scaled.py
```

> The script will warn if it cannot find the H5 weights (expected on Pi — H5 is PC-only). TFLite loading and inference will still be verified.

---

### Step 6: Run Full 630-File Evaluation & Latency Benchmark  *(Run on Pi Terminal)*

```bash
python3 evaluate_tflite_scaled_run.py
```

- **Expected runtime**: 3–10 minutes for 630 files on Pi 5 (PESQ computation is the main overhead).
- **Progress**: Progress is printed every 90 files (one full SNR band).
- **Expected per-block latency**: ~0.4–1.0 ms/block (budget: 8.0 ms).
- **Output CSV**: `~/DTLN_Pi/data_full/eval_results_tflite_scaled_run.csv`

> If PESQ takes too long on the Pi, you can disable it by removing the `pesq` import guard. SI-SNR and STOI will still be computed. PESQ can be computed on PC from the raw CSV.

---

### Step 7: Retrieve Results CSV to PC  *(Run on PC PowerShell)*

```powershell
scp pi_user@192.168.1.XXX:~/DTLN_Pi/data_full/eval_results_tflite_scaled_run.csv `
    "d:\SIH 2026\DTLN\data_full\eval_results_pi5_tflite_scaled.csv"
```

Once retrieved, share the CSV or its contents and the comparison table (Task 5) will be produced immediately.

---

## Previous Model Files — Confirmed Untouched

The following files from the 64-unit deployment are **not modified by any step above**:

```
~/DTLN_Pi/models_full_run/model_1.tflite   -- 64-unit Stage 1 (unchanged)
~/DTLN_Pi/models_full_run/model_2.tflite   -- 64-unit Stage 2 (unchanged)
~/DTLN_Pi/evaluate_tflite_full_run.py      -- 64-unit eval script (unchanged)
~/DTLN_Pi/data_full/eval_results_tflite_full_run.csv  -- 64-unit results (unchanged)
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `RuntimeError: Tensor index for 'idx_s1_in' not found` | TFLite model was exported with wrong numUnits | Re-export on PC with `export_tflite_scaled_run.py` and retransfer |
| `FileNotFoundError: models_scaled_run/model_1.tflite` | Transfer missed or wrong destination | Re-run Step 2 and verify Step 3 |
| Very slow execution (>5 min per file) | PESQ computation bottleneck | Expected; let it run, or comment out PESQ calls |
| `ModuleNotFoundError: tflite_runtime` | Venv not activated | Run `source venv/bin/activate` again |
| Per-block latency > 8.0 ms | Unexpected regression or thermal throttle | Check Pi temperature (`vcgencmd measure_temp`); ensure adequate cooling |
