# Active Noise Cancellation & Real-Time Speech Enhancement with DTLN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow 2.18+](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)
[![Raspberry Pi 5](https://img.shields.io/badge/Raspberry%20Pi%205-ARM64-red.svg)](https://www.raspberrypi.com/)
[![Real-Time Factor](https://img.shields.io/badge/RTF-0.0678x-brightgreen.svg)](#-benchmark-results)

An end-to-end real-time speech enhancement and active noise cancellation (ANC) pipeline based on the **Dual-Signal Transformation LSTM Network (DTLN)** architecture (`numUnits=256`, `numLayer=1`, ~1.45M parameters). Featuring automated multi-dataset synthesis, GPU acceleration on Google Colab, TFLite/LiteRT stateful streaming conversion, quantitative PESQ/STOI/SI-SNR evaluation, and real-time deployment on **Raspberry Pi 5**.

---

## 📋 Key Features & Highlights

- **High-Capacity DTLN Architecture**: 2-stage stacked architecture combining STFT-domain mask estimation and learned time-domain Conv1D feature synthesis (`numUnits=256`, `numLayer=1`, 1,451,266 parameters).
- **Ultra-Low Latency & High Real-Time Headroom**: Frame-by-frame block processing (512-sample STFT window, 128-sample hop size = **8.0 ms budget @ 16 kHz**). Achieves **0.543 ms per block** latency on Raspberry Pi 5 (**14.75x real-time speed**, **93.2% CPU headroom**).
- **Robust Multi-Dataset Pipeline**: Automated synthesis script (`generate_full_dataset.py`) pairing clean speech (**LibriSpeech `dev-clean`**) with environmental noise (**ESC-50**, 50 categories) and background noise (**MUSAN**). Trained on 4,200 pairs across 7 SNR bands (-5, 0, 5, 10, 15, 20, 25 dB).
- **Strict Data Integrity**: 100% disjoint speaker splits between training (3,570 pairs / 85%) and validation (630 pairs / 15%), verified 16 kHz mono format, and automated SNR verification (`verify_full_dataset.py`).
- **Google Colab GPU Training Workflow**: Self-contained notebook builder (`build_colab_notebook.py`) and preconfigured notebook (`train_dtln_colab.ipynb`) for fast GPU training.
- **Stateful TFLite / LiteRT Conversion**: Exporter script (`export_tflite_scaled_run.py`) splitting the architecture into stateful frequency-domain (`model_1.tflite`, 2.27 MB) and time-domain (`model_2.tflite`, 3.27 MB) submodels with carryover LSTM states (`[1, 1, 256, 2]`).
- **Raspberry Pi 5 Verified**: Complete step-by-step ARM64 deployment guide (`pi_deployment_guide_scaled.md`) and lightweight requirements (`requirements_pi.txt`).
- **Live Real-Time Audio Streaming**: Direct microphone/speaker real-time processing via `real_time_processing_tf_lite.py`.

---

## 📊 Benchmark Results

### 1. Raspberry Pi 5 Measured Performance

| Metric | Input Noisy Audio | Enhanced Output (DTLN Model) | Improvement / Performance |
| :--- | :--- | :--- | :--- |
| **SI-SNR** | 5.00 dB | **11.26 dB** | **+6.26 dB Gain** |
| **STOI** | 0.8349 | **0.8553** | **+0.0204 Gain** |
| **PESQ (Wideband)** | 1.475 | **1.782** | **+0.307 Gain** |
| **Avg Block Latency** | — | **0.543 ms / block** | 7.457 ms headroom (vs 8.0 ms budget) |
| **Real-Time Factor (RTF)**| — | **0.0678x** | **14.75x Faster than Real-Time** |
| **CPU Headroom** | — | **93.2%** | Remaining CPU capacity per 8ms frame |

---

### 2. Breakdown by Input SNR Level (120 Validation Files on Pi 5)

| Input SNR | Files | SI-SNR Before | SI-SNR After | STOI Before | STOI After | PESQ Before | PESQ After | Block Latency | Real-Time Factor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **-5.0 dB** | 24 | -4.98 dB | **2.45 dB** | 0.6870 | **0.7005** | 1.133 | **1.247** | 0.543 ms | 0.0678x |
| **0.0 dB** | 24 | -0.00 dB | **8.01 dB** | 0.7969 | **0.8322** | 1.258 | **1.508** | 0.543 ms | 0.0678x |
| **5.0 dB** | 24 | 5.00 dB | **11.24 dB** | 0.8554 | **0.8744** | 1.469 | **1.767** | 0.543 ms | 0.0678x |
| **10.0 dB** | 24 | 9.99 dB | **15.79 dB** | 0.8984 | **0.9234** | 1.528 | **1.999** | 0.543 ms | 0.0678x |
| **15.0 dB** | 24 | 14.99 dB | **18.80 dB** | 0.9369 | **0.9462** | 1.985 | **2.386** | 0.543 ms | 0.0678x |
| **OVERALL** | **120** | **5.00 dB** | **11.26 dB** | **0.8349** | **0.8553** | **1.475** | **1.782** | **0.543 ms** | **0.0678x** |

---

## 📁 Repository Structure

```
DTLN/
├── DTLN_model.py                     # Core DTLN architecture (numUnits=256, stateful submodels)
├── generate_full_dataset.py          # ESC-50 + MUSAN + LibriSpeech dataset synthesizer
├── verify_full_dataset.py            # Automated dataset integrity & SNR spot-checker
├── build_colab_notebook.py           # Generator script for Google Colab training notebook
├── train_dtln_colab.ipynb            # Google Colab GPU training notebook
├── export_tflite_scaled_run.py       # Exports 256u weights to model_1.tflite & model_2.tflite
├── sanity_check_tflite_scaled.py     # PC-side TFLite vs H5 numerical agreement checker
├── evaluate_tflite_scaled_run.py     # Full evaluation pipeline (PESQ, STOI, SI-SNR, RTF)
├── real_time_processing_tf_lite.py   # Live audio streaming inference (sounddevice / TFLite)
├── real_time_dtln_audio.py           # Real-time microphone audio processing runner
├── pi_deployment_guide_scaled.md     # Raspberry Pi 5 setup & CLI deployment guide
├── requirements_pi.txt               # Lightweight ARM64 dependencies for Raspberry Pi 5
├── models_scaled_run/                # Model weights & exported TFLite models
│   ├── scaled_run.weights.h5         # Trained Keras 256-unit model weights
│   ├── model_1.tflite                # Stage 1 STFT-domain TFLite submodel (2.27 MB)
│   └── model_2.tflite                # Stage 2 Time-domain TFLite submodel (3.27 MB)
├── data_full/                        # Dataset audio files and evaluation result CSVs
└── README.md                         # Project documentation
```

---

## 🛠️ Environment Setup

### 1. Prerequisites
- Python 3.10+
- Virtual Environment (`.venv`)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/JOE-013/ANC.git
cd ANC

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install tensorflow librosa soundfile pystoi pesq scipy numpy sounddevice
```

---

## 🚀 Quickstart & Usage

### 1. Generate Training & Validation Dataset
To synthesize clean speech / noise pairs (16 kHz mono, 15s clips) across -5 dB to 25 dB SNR:
```bash
python generate_full_dataset.py
python verify_full_dataset.py
```

### 2. Train on Google Colab (GPU)
Upload `train_dtln_colab.ipynb` and `data_full/` to Google Colab, select a GPU T4/A100 runtime, and execute training. Save the resulting `scaled_run.weights.h5` into `models_scaled_run/`.

### 3. Export Trained Weights to TFLite
To convert the 256-unit Keras model into stateful TFLite submodels:
```bash
python export_tflite_scaled_run.py
```
Outputs:
- `models_scaled_run/model_1.tflite` (2.27 MB)
- `models_scaled_run/model_2.tflite` (3.27 MB)

### 4. Run Numerical Sanity Check
To verify that TFLite predictions match the reference Keras model output (SI-SNR agreement ≥ 35 dB):
```bash
python sanity_check_tflite_scaled.py
```

### 5. Evaluate on Raspberry Pi 5
Transfer `models_scaled_run/` and `evaluate_tflite_scaled_run.py` to Raspberry Pi 5. Run the evaluation suite:
```bash
python evaluate_tflite_scaled_run.py
```
Follow [pi_deployment_guide_scaled.md](pi_deployment_guide_scaled.md) for full ARM64 system setup instructions.

### 6. Run Real-Time Microphone Streaming
To run real-time noise suppression from microphone input to speaker output:
```bash
python real_time_processing_tf_lite.py
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
