#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_training.py - Full-scale training script for DTLN speech noise suppression model.
Trains on 680 training pairs and validates on 120 validation pairs across 50 epochs with EarlyStopping.
"""

import os
import sys
import time
import tensorflow as tf
from DTLN_model import DTLN_model

# Force CPU for Windows stability if native GPU is unavailable
os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def main():
    print("=== DTLN Full Training Run Configuration ===")
    
    # 1. Device Inspection
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print(f"Physical GPUs Available: {len(gpus)}")
    print(f"Physical CPUs Available: {len(cpus)}")
    if len(gpus) > 0:
        print(f"Execution Device: GPU ({gpus[0].name})")
    else:
        print("Execution Device: CPU (Intel x86_64)")

    # 2. Path Setup
    path_to_train_mix = os.path.abspath('./data_full/train_mix')
    path_to_train_speech = os.path.abspath('./data_full/train_speech')
    path_to_val_mix = os.path.abspath('./data_full/val_mix')
    path_to_val_speech = os.path.abspath('./data_full/val_speech')

    run_name = 'full_run'

    # 3. Model & Hyperparameter Configuration
    model_trainer = DTLN_model()
    model_trainer.numUnits = 64
    model_trainer.numLayer = 1
    model_trainer.max_epochs = 50
    model_trainer.batchsize = 16
    model_trainer.cost_function = model_trainer.snr_cost  # SI-SNR Loss

    print("\nTraining Parameters:")
    print(f"  LSTM Units (`numUnits`): {model_trainer.numUnits}")
    print(f"  LSTM Layers (`numLayer`): {model_trainer.numLayer}")
    print(f"  Max Epoch Budget:       {model_trainer.max_epochs}")
    print(f"  Batch Size:             {model_trainer.batchsize}")
    print(f"  Loss Function:          {model_trainer.cost_function.__name__} (SI-SNR Loss)")
    print(f"  Train Set Path:         {path_to_train_mix} (680 pairs)")
    print(f"  Validation Set Path:    {path_to_val_mix} (120 pairs)")

    # 4. Build & Compile
    print("\nBuilding model architecture...")
    model_trainer.build_DTLN_model()
    model_trainer.compile_model()

    print("\nStarting full training execution (Monitoring early stopping & val_loss checkpoints)...")
    start_time = time.time()
    
    model_trainer.train_model(
        run_name,
        path_to_train_mix,
        path_to_train_speech,
        path_to_val_mix,
        path_to_val_speech
    )
    
    elapsed = time.time() - start_time
    print(f"\n=== Full Training Run Complete in {elapsed / 60.0:.2f} minutes ===")

    checkpoint_h5 = os.path.abspath(f'./models_{run_name}/{run_name}.h5')
    checkpoint_weights = os.path.abspath(f'./models_{run_name}/{run_name}.weights.h5')
    log_file = os.path.abspath(f'./models_{run_name}/training_{run_name}.log')

    if os.path.exists(checkpoint_weights):
        print(f"SUCCESS: Best model weights saved at: {checkpoint_weights} ({os.path.getsize(checkpoint_weights)} bytes)")
    elif os.path.exists(checkpoint_h5):
        print(f"SUCCESS: Best model file saved at: {checkpoint_h5} ({os.path.getsize(checkpoint_h5)} bytes)")

    if os.path.exists(log_file):
        print(f"Training progress log written to: {log_file}")

if __name__ == '__main__':
    main()
