#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_tflite_384_babble_run.py -- Export DTLN 384-unit babble fine-tuned weights to stateful TFLite submodels.
Loads models_384_babble_run/384_babble_run.weights.h5 with numUnits=384, numLayer=1.
Exports model_1.tflite and model_2.tflite into models_384_babble_run/ directory.
"""

import os
import tensorflow as tf
from DTLN_model import DTLN_model

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "models_384_babble_run")
WEIGHTS_PATH  = os.path.join(OUTPUT_DIR, "384_babble_run.weights.h5")
TARGET_PREFIX = os.path.join(OUTPUT_DIR, "model")
MODEL1_PATH   = f"{TARGET_PREFIX}_1.tflite"
MODEL2_PATH   = f"{TARGET_PREFIX}_2.tflite"

NUM_UNITS = 384
NUM_LAYER = 1


def inspect_tflite(model_path, label):
    print(f"\n--- Inspecting {label} ({model_path}) ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    size_kb = os.path.getsize(model_path) / 1024
    print(f"  File size : {size_kb:.1f} KB  ({size_kb/1024:.2f} MB)")

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()

    print("  Input tensors:")
    for i, d in enumerate(interp.get_input_details()):
        print(f"    [{i}] {d['name']:40s}  shape={d['shape']}  dtype={d['dtype'].__name__}")

    print("  Output tensors:")
    for i, d in enumerate(interp.get_output_details()):
        print(f"    [{i}] {d['name']:40s}  shape={d['shape']}  dtype={d['dtype'].__name__}")


def main():
    print("=== Exporting DTLN 384-unit Babble Fine-Tuned Model to TFLite (Float32) ===")
    print(f"  Weights  : {WEIGHTS_PATH}")
    print(f"  Output   : {OUTPUT_DIR}/")
    print(f"  numUnits : {NUM_UNITS}  |  numLayer : {NUM_LAYER}")

    if not os.path.exists(WEIGHTS_PATH):
        # Fallback to root directory if present
        root_weights = os.path.join(SCRIPT_DIR, "384_babble_run.weights.h5")
        if os.path.exists(root_weights):
            import shutil
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            shutil.copy(root_weights, WEIGHTS_PATH)
        else:
            raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Instantiate model with 384-unit architecture
    converter = DTLN_model()
    converter.numUnits = NUM_UNITS
    converter.numLayer = NUM_LAYER

    print("\nExporting two-stage DTLN architecture to TFLite ...")
    converter.create_tf_lite_model(
        weights_file=WEIGHTS_PATH,
        target_name=TARGET_PREFIX,
        use_dynamic_range_quant=False
    )
    print("Export complete.")

    # Inspect both exported submodels
    inspect_tflite(MODEL1_PATH, "Stage 1  (Frequency-domain mask)")
    inspect_tflite(MODEL2_PATH, "Stage 2  (Time-domain separation)")

    # Quick sanity: confirm state tensors are 384-unit shaped
    print("\n=== Shape Verification ===")
    for path, label in [(MODEL1_PATH, "model_1"), (MODEL2_PATH, "model_2")]:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        for d in interp.get_input_details() + interp.get_output_details():
            sh = d["shape"]
            if len(sh) == 4:
                n_units = sh[2]
                ok = "OK" if n_units == NUM_UNITS else "*** MISMATCH ***"
                print(f"  {label}: {d['name']:40s}  state shape={list(sh)}  [{ok}]")

    print("\nAll done. 384-unit TFLite files written to:")
    print(f"  {MODEL1_PATH}")
    print(f"  {MODEL2_PATH}")


if __name__ == "__main__":
    main()
