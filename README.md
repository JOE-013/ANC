# Active Noise Cancellation & Real-Time Speech Enhancement with DTLN (384-Unit Babble-Tuned)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow 2.18+](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)
[![Raspberry Pi 5](https://img.shields.io/badge/Raspberry%20Pi%205-ARM64-red.svg)](https://www.raspberrypi.com/)
[![Real-Time Factor](https://img.shields.io/badge/RTF-0.11x-brightgreen.svg)](#-benchmark-results)

An end-to-end real-time speech enhancement and active noise cancellation (ANC) pipeline based on the **Dual-Signal Transformation LSTM Network (DTLN)** architecture (`numUnits=384`, `numLayer=1`, ~3.1M parameters). This final model is specially fine-tuned to aggressively suppress human babble noise while preserving target speech. It features automated multi-dataset synthesis, GPU acceleration on Google Colab, TFLite/LiteRT stateful streaming conversion, and real-time deployment on the **Raspberry Pi 5**.

---

## 📋 Key Features & Highlights

- **High-Capacity DTLN Architecture**: 2-stage stacked architecture combining STFT-domain mask estimation and learned time-domain Conv1D feature synthesis (`numUnits=384`, `numLayer=1`, ~3.1M parameters).
- **Ultra-Low Latency & High Real-Time Headroom**: Frame-by-frame block processing (512-sample STFT window, 128-sample hop size = **8.0 ms budget @ 16 kHz**). Achieves **~0.88 ms per block** latency on Raspberry Pi 5 (**9x real-time speed**, **89% CPU headroom**).
- **Babble-Optimized Dataset Pipeline**: Automated synthesis script (`generate_babble_oversampled_dataset.py`) pairing clean speech (**LibriSpeech `dev-clean`**) with environmental noise (**ESC-50**) and specifically oversampling background voice chatter (**MUSAN Babble**). Trained on **5,610 pairs**.
- **Google Colab GPU Training Workflow**: Self-contained notebook builder (`build_colab_notebook.py`) and preconfigured notebook (`train_dtln_colab.ipynb`) for fast GPU training with epoch-level checkpointing.
- **Stateful TFLite / LiteRT Conversion**: Exporter script (`export_tflite_384_babble_run.py`) splitting the architecture into stateful frequency-domain (`model_1.tflite`) and time-domain (`model_2.tflite`) submodels with carryover LSTM states (`[1, 1, 384, 2]`).
- **Raspberry Pi 5 Verified**: Complete step-by-step ARM64 deployment guide (`pi_deployment_guide_384_babble.md`) and lightweight requirements (`requirements_pi.txt`).

---

## 📊 Benchmark Results

### 1. Raspberry Pi 5 Measured Performance (Babble-Tuned Model)

| Metric | Overall Performance | Notes |
| :--- | :--- | :--- |
| **SI-SNR Improvement** | **+12.89 dB Gain** | Tested on matched 120-file evaluation subset |
| **STOI** | **0.8805** | High objective speech intelligibility |
| **Babble SI-SNR Gain** | **+14.10 dB Gain** | Highly effective on complex human background noise |
| **Avg Block Latency** | **~0.88 ms / block** | 7.12 ms headroom (vs 8.0 ms budget) |
| **Real-Time Factor (RTF)**| **0.11x** | **9x Faster than Real-Time** |
| **CPU Headroom** | **89%** | Remaining CPU capacity per 8ms frame |

---

## 📁 Repository Structure

```
DTLN/
├── DTLN_model.py                        # Core DTLN architecture (numUnits=384, stateful submodels)
├── generate_babble_oversampled_dataset.py # Synthesizes 5,610 pairs with extreme babble focus
├── build_colab_notebook.py              # Generator script for Google Colab training notebook
├── train_dtln_colab.ipynb               # Google Colab GPU training notebook
├── export_tflite_384_babble_run.py      # Exports 384u weights to TFLite models
├── sanity_check_tflite_384_babble.py    # PC-side & Pi-side TFLite latency and SNR checker
├── eval_matched120.py                   # Full evaluation pipeline for exact 120-file comparison
├── pi_deployment_guide_384_babble.md    # Raspberry Pi 5 setup & CLI deployment guide
├── requirements_pi.txt                  # Lightweight ARM64 dependencies for Raspberry Pi 5
├── models_384_babble_run/               # Model weights & exported TFLite models
│   ├── model_1.tflite                   # Stage 1 STFT-domain TFLite submodel
│   └── model_2.tflite                   # Stage 2 Time-domain TFLite submodel
├── data_full/                           # Dataset audio files and evaluation result CSVs
└── README.md                            # Project documentation
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
To synthesize the babble-optimized training set (16 kHz mono, 5,610 pairs):
```bash
python generate_babble_oversampled_dataset.py
```

### 2. Train on Google Colab (GPU)
Upload `train_dtln_colab.ipynb` and `data_full/` to Google Colab, select a GPU T4/A100 runtime, and execute training.

### 3. Export Trained Weights to TFLite
To convert the 384-unit Keras model into stateful TFLite submodels:
```bash
python export_tflite_384_babble_run.py
```

### 4. Evaluate on Raspberry Pi 5
Transfer `models_384_babble_run/` and evaluation scripts to Raspberry Pi 5. Run the evaluation suite:
```bash
python eval_matched120.py
```
Follow [pi_deployment_guide_384_babble.md](pi_deployment_guide_384_babble.md) for full ARM64 system setup instructions and exact `scp` transfer commands.

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
