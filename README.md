# Active Noise Cancellation & Speech Enhancement with DTLN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow 2.18+](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

An end-to-end real-time speech enhancement and active noise cancellation (ANC) pipeline based on the **Dual-Signal Transformation LSTM Network (DTLN)** architecture, equipped with automated multi-dataset generation, hardware latency benchmarking, quantitative PESQ/STOI/SI-SNR evaluation, and a classical non-ML transient limiter post-processor.

---

## 📋 Key Features & Highlights

- **Lightweight Real-Time Architecture**: Stacked STFT + learned Conv1D analysis network with `numUnits=64` and `numLayer=1` (460,609 total parameters / 1.76 MB), designed for real-time block-by-block processing (32ms window, 8ms hop).
- **Multi-Dataset Synthesis Pipeline**: Automated pairing script (`generate_full_dataset.py`) combining clean speech (**LibriSpeech `dev-clean`**) with 50 categories of environmental noise (**ESC-50**) and background noise/babble (**MUSAN**).
- **Rigorous Data Integrity**: 100% disjoint speaker split between training (680 clips / 85%) and validation (120 clips / 15%), verified format (16kHz mono, 15.0s), and SNR spot-checking (`verify_full_dataset.py`).
- **Production Training Loop**: Optimized training using plain **SI-SNR Loss** (`snr_cost`), batch size 16, `EarlyStopping(patience=10)`, and `ReduceLROnPlateau` learning rate decay.
- **Quantitative Evaluation Suite**: Automated evaluation (`evaluate_full_run.py`) computing **wideband PESQ**, **STOI**, and **SI-SNR** across 120 held-out validation pairs.
- **Bolt-On Transient Limiter**: Classical short-window energy transient limiter (`transient_limiter.py`) with fast attack (2ms) and smooth release (50ms) to suppress residual high-energy impulse noise spikes.

---

## 📊 Benchmark Results

### 1. Quantitative Quality Evaluation (120 Held-Out Validation Files)

| Evaluation Stage | SI-SNR (dB) | STOI | PESQ (wideband) | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| **Noisy Reference** | 5.00 dB | 0.8350 | 1.474 | Raw input across 5 SNR levels (-5 to 15 dB) |
| **Smoke-Test Model (45 pairs)** | 3.39 dB (-1.63 dB) | 0.7242 (-0.0886) | 1.169 (-0.300) | Severe under-training artifacts & PESQ regression |
| **Full-Run Model (800 pairs, Ep 42)** | **8.68 dB (+3.68 dB)** | **0.8283 (-0.0067)** | **1.529 (+0.055)** | **PESQ regression RESOLVED; +5.31 dB SI-SNR improvement** |
| **Full-Run + Limiter (-3.0 dBFS)** | **8.67 dB (+3.67 dB)** | **0.8283 (-0.0067)** | **1.529 (+0.055)** | Absolute peak ceiling protection without speech degradation |

### 2. Validation Breakdown by Input SNR Condition

| Input SNR Condition | SI-SNR Before | SI-SNR After | SI-SNR Gain | STOI Before | STOI After | PESQ Before | PESQ After |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **-5.0 dB** | -4.98 dB | **0.52 dB** | **+5.50 dB** | 0.6871 | 0.6830 | 1.132 | **1.158** (+0.026) |
| **0.0 dB** | -0.00 dB | **5.84 dB** | **+5.84 dB** | 0.7970 | **0.7997** | 1.257 | **1.366** (+0.109) |
| **5.0 dB** | 5.00 dB | **9.06 dB** | **+4.06 dB** | 0.8554 | 0.8445 | 1.469 | **1.535** (+0.066) |
| **10.0 dB** | 9.99 dB | **12.80 dB** | **+2.81 dB** | 0.8984 | 0.8928 | 1.528 | **1.626** (+0.098) |
| **15.0 dB** | 14.99 dB | **15.17 dB** | **+0.18 dB** | 0.9368 | 0.9214 | 1.984 | 1.961 (-0.023) |
| **OVERALL** | **5.00 dB** | **8.68 dB** | **+3.68 dB** | **0.8350** | **0.8283** | **1.474** | **1.529** (+0.055) |

### 3. Real-Time Inference Latency

- **Target Real-Time Budget**: `< 8.0 ms` per 8ms block shift for real-time streaming capability.
- **Laptop CPU Benchmark**: **`1.15 ms - 1.60 ms`** per block (5–7x faster than the real-time threshold budget).

---

## 📁 Repository Structure

```
├── DTLN_model.py                             # Core DTLN model architecture & Keras 3 training setup
├── mix_pairs.py                              # Noisy speech pair mixer with controlled SNR
├── generate_full_dataset.py                  # ESC-50 + MUSAN + LibriSpeech dataset generator
├── verify_full_dataset.py                    # Automated dataset verification & SNR spot-checking
├── run_full_training.py                      # Production model training runner (numUnits=64, numLayer=1)
├── evaluate_full_run.py                      # PESQ / STOI / SI-SNR quantitative evaluation script
├── transient_limiter.py                      # Energy-based / crest-factor transient limiter post-processor
├── evaluate_full_run_with_limiter_thresh3.py # Evaluation script for chained DTLN + Limiter (-3.0 dBFS)
├── measure_execution_time.py                 # CPU block-by-block inference latency benchmark
├── measure_tflite_latency.py                 # TFLite block latency benchmark
├── real_time_processing.py                   # Real-time block streaming inference example
├── convert_weights_to_tf_lite.py             # TFLite export utility
├── convert_weights_to_onnx.py                # ONNX export utility
├── .gitignore                                # Git ignore configuration
└── README.md                                 # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Virtual Environment (`.venv`)

### 2. Environment Installation

```bash
# Clone the repository
git clone https://github.com/JOE-013/ANC.git
cd ANC

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install tensorflow librosa soundfile pystoi pesq scipy numpy
```

---

## 🚀 Usage Guide

### 1. Generating Training & Validation Datasets

To synthesize 800 paired noisy/clean audio files (16kHz mono, 15s clips) from ESC-50, MUSAN, and LibriSpeech:

```bash
python generate_full_dataset.py
```

To verify dataset integrity, disjoint speaker splits, and target SNR accuracy:

```bash
python verify_full_dataset.py
```

### 2. Training the Model

To launch the full training run using `numUnits=64`, `numLayer=1`, SI-SNR loss, and early stopping:

```bash
python run_full_training.py
```

Checkpoints will be saved to `models_full_run/full_run.weights.h5`, and training logs will be written to `models_full_run/training_full_run.log`.

### 3. Quantitative Evaluation (PESQ, STOI, SI-SNR)

To evaluate the trained model on all 120 held-out validation files:

```bash
python evaluate_full_run.py
```

### 4. Running with the Transient Limiter

To test the standalone limiter on raw audio:

```bash
python test_standalone_limiter.py
```

To evaluate the chained pipeline (**DTLN + Transient Limiter @ -3.0 dBFS**):

```bash
python evaluate_full_run_with_limiter_thresh3.py
```

### 5. Latency Benchmarking

To measure block-by-block CPU inference latency:

```bash
python measure_execution_time.py
```

---

## 📜 Citation & References

If you use this repository or model in your work, please cite the original DTLN paper:

```bibtex
@inproceedings{Westhausen2020,
  author={Nils L. Westhausen and Bernd T. Meyer},
  title={{Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={2477--2481},
  doi={10.21437/Interspeech.2020-2631}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
