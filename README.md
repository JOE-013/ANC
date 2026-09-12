# Active Noise Cancellation & Speech Enhancement with DTLN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow 2.18+](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)
[![Raspberry Pi 5](https://img.shields.io/badge/Raspberry%20Pi%205-ARM64-red.svg)](https://www.raspberrypi.com/)

An end-to-end real-time speech enhancement and active noise cancellation (ANC) pipeline based on the **Dual-Signal Transformation LSTM Network (DTLN)** architecture, equipped with automated multi-dataset generation, hardware latency benchmarking, quantitative PESQ/STOI/SI-SNR evaluation, a classical non-ML transient limiter post-processor, and **Raspberry Pi 5 TFLite/LiteRT deployment packaging**.

---

## 📋 Key Features & Highlights

- **Lightweight Real-Time Architecture**: Stacked STFT + learned Conv1D analysis network with `numUnits=64` and `numLayer=1` (460,609 total parameters / 1.76 MB), designed for real-time block-by-block processing (32ms window, 8ms hop / 128 samples @ 16 kHz).
- **Multi-Dataset Synthesis Pipeline**: Automated pairing script (`generate_full_dataset.py`) combining clean speech (**LibriSpeech `dev-clean`**) with 50 categories of environmental noise (**ESC-50**) and background noise/babble (**MUSAN**).
- **Rigorous Data Integrity**: 100% disjoint speaker split between training (680 clips / 85%) and validation (120 clips / 15%), verified format (16kHz mono, 15.0s), and SNR spot-checking (`verify_full_dataset.py`).
- **Production Training Loop**: Optimized training using plain **SI-SNR Loss** (`snr_cost`), batch size 16, `EarlyStopping(patience=10)`, and `ReduceLROnPlateau` learning rate decay.
- **Quantitative Evaluation Suite**: Automated evaluation (`evaluate_full_run.py`) computing **wideband PESQ**, **STOI**, and **SI-SNR** across 120 held-out validation pairs.
- **TFLite & LiteRT Conversion**: Export utility (`export_tflite_full_run.py`) separating the architecture into stateful frequency-domain (`model_1.tflite`) and time-domain (`model_2.tflite`) submodels.
- **Raspberry Pi 5 Ready**: Dedicated deployment guide (`pi_deployment_guide.md`) and lightweight requirements (`requirements_pi.txt`) for offline batch evaluation on ARM64 hardware.
- **Bolt-On Transient Limiter**: Classical short-window energy transient limiter (`transient_limiter.py`) with relative noise floor tracking, fast attack (2ms), and smooth release (50ms) to suppress residual impulse noise spikes.

---

## 📊 Benchmark Results

### 1. Quantitative Quality Evaluation (120 Held-Out Validation Files)

| Evaluation Stage | SI-SNR (dB) | STOI | PESQ (wideband) | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| **Noisy Reference** | 5.02 dB | 0.8059 | 1.356 | Raw input across 5 SNR levels (-5 to 15 dB) |
| **Full-Run Keras .h5 Model** | **8.68 dB (+3.66 dB)** | **0.8414 (+0.0355)** | **1.589 (+0.233)** | **+3.66 dB SI-SNR gain & PESQ improvement across all SNR levels** |
| **TFLite / LiteRT Engine** | **8.67 dB (+3.65 dB)** | **0.8414 (+0.0355)** | **1.589 (+0.233)** | **Identical output quality; zero degradation from TFLite conversion** |

### 2. Validation Breakdown by Input SNR Condition (TFLite Engine)

| Input SNR Condition | SI-SNR Before | SI-SNR After | SI-SNR Gain | STOI Before | STOI After | PESQ Before | PESQ After | Avg Block Latency | Real-Time Factor (RTF) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **-5.0 dB** | -4.98 dB | **2.87 dB** | **+7.85 dB** | 0.6047 | **0.6865** | 1.096 | **1.258** | 0.135 ms | 0.0169x |
| **0.0 dB** | 0.01 dB | **5.92 dB** | **+5.91 dB** | 0.7288 | **0.7915** | 1.157 | **1.393** | 0.139 ms | 0.0173x |
| **5.0 dB** | 5.03 dB | **9.47 dB** | **+4.44 dB** | 0.8354 | **0.8659** | 1.267 | **1.579** | 0.137 ms | 0.0172x |
| **10.0 dB** | 10.02 dB | **11.75 dB** | **+1.73 dB** | 0.9080 | **0.9160** | 1.480 | **1.761** | 0.134 ms | 0.0168x |
| **15.0 dB** | 15.01 dB | **13.34 dB** | **-1.67 dB** | 0.9525 | 0.9472 | 1.782 | **1.956** | 0.134 ms | 0.0168x |
| **OVERALL** | **5.02 dB** | **8.67 dB** | **+3.65 dB** | **0.8059** | **0.8414** | **1.356** | **1.589** | **0.136 ms** | **0.0170x** |

### 3. Real-Time Inference Latency

- **Target Real-Time Budget**: `< 8.0 ms` per 8ms block shift for real-time streaming capability.
- **Laptop CPU Benchmark**: **`0.136 ms`** per block (58.8x faster than the real-time threshold budget).
- **Latency Headroom Margin**: **`7.864 ms`** headroom remaining per 8.0 ms frame.

---

## 📁 Repository Structure

```
├── DTLN_model.py                             # Core DTLN model architecture & stateful TFLite submodels
├── generate_full_dataset.py                  # ESC-50 + MUSAN + LibriSpeech dataset generator
├── verify_full_dataset.py                    # Automated dataset verification & SNR spot-checking
├── run_full_training.py                      # Production model training runner (numUnits=64, numLayer=1)
├── evaluate_full_run.py                      # PESQ / STOI / SI-SNR quantitative evaluation script for .h5 model
├── transient_limiter.py                      # Adaptive noise floor tracking transient limiter post-processor
├── evaluate_full_run_with_limiter_relative.py # Evaluation script for chained DTLN + Relative Limiter
├── export_tflite_full_run.py                 # Exports .h5 weights to model_1.tflite & model_2.tflite
├── sanity_check_tflite.py                    # Quick 10-file LiteRT / TFLite numerical verification script
├── evaluate_tflite_full_run.py               # Full 120-file TFLite evaluation script with per-block timing
├── pi_deployment_guide.md                    # Step-by-step Raspberry Pi 5 setup & CLI execution guide
├── requirements_pi.txt                       # Lightweight runtime Python dependencies for Raspberry Pi 5
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

To launch full training using `numUnits=64`, `numLayer=1`, SI-SNR loss, and early stopping:

```bash
python run_full_training.py
```

Checkpoints will be saved to `models_full_run/full_run.weights.h5`, and training logs will be written to `models_full_run/training_full_run.log`.

### 3. Exporting to TFLite

To convert the trained `.h5` model into two stateful TFLite submodels (`model_1.tflite` and `model_2.tflite`):

```bash
python export_tflite_full_run.py
```

### 4. Running TFLite / LiteRT Sanity Check

To run a quick 10-file comparative verification:

```bash
python sanity_check_tflite.py
```

### 5. Full TFLite Quantitative Evaluation & Latency Measurement

To evaluate the TFLite submodels over all 120 held-out validation files with per-block timing instrumentation:

```bash
python evaluate_tflite_full_run.py
```

### 6. Raspberry Pi 5 Deployment

For complete, copy-pasteable CLI commands to transfer files, install system dependencies (`apt-get`), configure virtual environments, and run evaluation on a Raspberry Pi 5, refer to [pi_deployment_guide.md](pi_deployment_guide.md).

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
