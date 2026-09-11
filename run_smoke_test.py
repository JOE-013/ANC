#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_smoke_test.py - Smoke test script for DTLN model training pipeline.
Validates end-to-end training loop execution using 45 test audio pairs.
"""

import os
import tensorflow as tf
from DTLN_model import DTLN_model

# Force CPU for smoke test reproducibility
os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def main():
    print("=== DTLN Training Smoke Test Configuration ===")
    
    # Path setup
    path_to_train_mix = os.path.abspath('./data_test/train_mix')
    path_to_train_speech = os.path.abspath('./data_test/train_speech')
    path_to_val_mix = path_to_train_mix  # Reusing train folder as val for smoke test
    path_to_val_speech = path_to_train_speech

    run_name = 'smoke_test'

    # Instantiate model trainer
    model_trainer = DTLN_model()

    # Apply required smoke test configuration
    model_trainer.numUnits = 64
    model_trainer.numLayer = 1
    model_trainer.max_epochs = 5
    model_trainer.batchsize = 4
    model_trainer.cost_function = model_trainer.snr_cost  # Plain SI-SNR loss

    print(f"  numUnits:       {model_trainer.numUnits}")
    print(f"  numLayer:       {model_trainer.numLayer}")
    print(f"  max_epochs:     {model_trainer.max_epochs}")
    print(f"  batchsize:      {model_trainer.batchsize}")
    print(f"  Loss Function:  {model_trainer.cost_function.__name__} (SI-SNR Loss)")
    print(f"  Train Mix Path: {path_to_train_mix}")
    print(f"  Train Clean Path: {path_to_train_speech}")
    print(f"  Val Mix Path:   {path_to_val_mix} (Note: Reusing train set for smoke test)")
    print(f"  Val Clean Path: {path_to_val_speech}")

    # Build and compile model
    print("\nBuilding model architecture...")
    model_trainer.build_DTLN_model()
    model_trainer.compile_model()

    print("\nStarting smoke test training run...")
    # Execute training loop
    model_trainer.train_model(
        run_name,
        path_to_train_mix,
        path_to_train_speech,
        path_to_val_mix,
        path_to_val_speech
    )

    print("\n=== Smoke Test Complete ===")
    checkpoint_file_h5 = os.path.abspath(f'./models_{run_name}/{run_name}.h5')
    checkpoint_file_weights = os.path.abspath(f'./models_{run_name}/{run_name}.weights.h5')
    log_file = os.path.abspath(f'./models_{run_name}/training_{run_name}.log')
    
    if os.path.exists(checkpoint_file_h5):
        print(f"SUCCESS: Saved checkpoint model file found at: {checkpoint_file_h5} ({os.path.getsize(checkpoint_file_h5)} bytes)")
    elif os.path.exists(checkpoint_file_weights):
        print(f"SUCCESS: Saved checkpoint weights file found at: {checkpoint_file_weights} ({os.path.getsize(checkpoint_file_weights)} bytes)")
    else:
        print(f"WARNING: Checkpoint file not found at expected paths.")

    if os.path.exists(log_file):
        print(f"Training log file written to: {log_file}")

if __name__ == '__main__':
    main()
