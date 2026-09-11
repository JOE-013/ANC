import time
import os
import platform
import numpy as np
import tensorflow as tf

print("=== System Info ===")
print("Python:", platform.python_version())
print("OS:", platform.system(), platform.release(), platform.version())
print("Machine:", platform.machine())
print("Processor:", platform.processor())
print("TensorFlow:", tf.__version__)

def benchmark_tflite_pipeline(model1_path, model2_path, num_iterations=500, warmup=20):
    print(f"\n--- Benchmarking TFLite Pipeline ---")
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path}")

    # Load interpreters using tf.lite.Interpreter
    interp1 = tf.lite.Interpreter(model_path=model1_path)
    interp1.allocate_tensors()

    interp2 = tf.lite.Interpreter(model_path=model2_path)
    interp2.allocate_tensors()

    in_det1 = interp1.get_input_details()
    out_det1 = interp1.get_output_details()

    in_det2 = interp2.get_input_details()
    out_det2 = interp2.get_output_details()

    block_len = 512
    block_shift = 128

    # States
    states_1 = np.zeros(in_det1[1]['shape'], dtype=np.float32)
    states_2 = np.zeros(in_det2[1]['shape'], dtype=np.float32)

    in_buffer = np.zeros(block_len, dtype=np.float32)
    dummy_audio = np.random.randn(num_iterations * block_shift + block_len).astype(np.float32)

    latencies = []

    for idx in range(num_iterations):
        t0 = time.perf_counter()

        # Frame extraction & FFT
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = dummy_audio[idx * block_shift : idx * block_shift + block_shift]

        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft).reshape(1, 1, -1).astype(np.float32)
        in_phase = np.angle(in_block_fft)

        # Stage 1 execution
        interp1.set_tensor(in_det1[1]['index'], states_1)
        interp1.set_tensor(in_det1[0]['index'], in_mag)
        interp1.invoke()

        out_mask = interp1.get_tensor(out_det1[0]['index'])
        states_1 = interp1.get_tensor(out_det1[1]['index'])

        # IFFT
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex).reshape(1, 1, -1).astype(np.float32)

        # Stage 2 execution
        interp2.set_tensor(in_det2[1]['index'], states_2)
        interp2.set_tensor(in_det2[0]['index'], estimated_block)
        interp2.invoke()

        out_block = interp2.get_tensor(out_det2[0]['index'])
        states_2 = interp2.get_tensor(out_det2[1]['index'])

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    # Discard warmup
    latencies_measured = np.array(latencies[warmup:])

    mean_lat = np.mean(latencies_measured)
    median_lat = np.median(latencies_measured)
    min_lat = np.min(latencies_measured)
    max_lat = np.max(latencies_measured)
    p95_lat = np.percentile(latencies_measured, 95)

    print(f"Total blocks measured: {len(latencies_measured)} (discarded {warmup} warmup blocks)")
    print(f"  Mean:   {mean_lat:.3f} ms")
    print(f"  Median: {median_lat:.3f} ms")
    print(f"  Min:    {min_lat:.3f} ms")
    print(f"  Max:    {max_lat:.3f} ms")
    print(f"  P95:    {p95_lat:.3f} ms")

    return {
        'mean': mean_lat,
        'median': median_lat,
        'min': min_lat,
        'max': max_lat,
        'p95': p95_lat
    }

if __name__ == '__main__':
    print("\n--- Standard TFLite Models (model_1.tflite + model_2.tflite) ---")
    res_std = benchmark_tflite_pipeline('./pretrained_model/model_1.tflite', './pretrained_model/model_2.tflite')

    print("\n--- Quantized TFLite Models (model_quant_1.tflite + model_quant_2.tflite) ---")
    res_quant = benchmark_tflite_pipeline('./pretrained_model/model_quant_1.tflite', './pretrained_model/model_quant_2.tflite')
