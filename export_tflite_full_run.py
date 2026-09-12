#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_tflite_full_run.py — Export DTLN full-run weights to stateful TFLite submodels.
Loads models_full_run/full_run.weights.h5 with numUnits=64, numLayer=1, norm_stft=False.
Exports model_1.tflite and model_2.tflite into models_full_run directory, then verifies tensor shapes.
"""

import os
import tensorflow as tf
from DTLN_model import DTLN_model

def inspect_tflite_model(model_path, model_name):
    """
    Load a TFLite model and display detailed input/output tensor specs.
    """
    print(f"\n--- Inspecting {model_name} ({model_path}) ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found at {model_path}")
        
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nInput Tensors:")
    for idx, detail in enumerate(input_details):
        print(f"  [{idx}] Name: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype'].__name__}")
        
    print("\nOutput Tensors:")
    for idx, detail in enumerate(output_details):
        print(f"  [{idx}] Name: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype'].__name__}")

def main():
    print("=== Exporting DTLN Full-Run Model to TFLite (Float32 Unquantized) ===")
    
    weights_path = os.path.abspath("./models_full_run/full_run.weights.h5")
    target_prefix = os.path.abspath("./models_full_run/model")
    model1_path = f"{target_prefix}_1.tflite"
    model2_path = f"{target_prefix}_2.tflite"
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
    print(f"Weights Source: {weights_path}")
    print(f"Target Submodels: {model1_path} & {model2_path}")
    
    # 1. Instantiate DTLN_model with matching training parameters
    converter = DTLN_model()
    converter.numUnits = 64
    converter.numLayer = 1
    
    print("\nExporting DTLN two-stage architecture to TFLite...")
    converter.create_tf_lite_model(
        weights_file=weights_path,
        target_name=target_prefix,
        use_dynamic_range_quant=False
    )
    print("Export complete!")
    
    # 2. Inspect both exported models
    inspect_tflite_model(model1_path, "Stage 1 Model (Frequency Domain Separation)")
    inspect_tflite_model(model2_path, "Stage 2 Model (Time Domain Feature Separation)")

if __name__ == '__main__':
    main()
