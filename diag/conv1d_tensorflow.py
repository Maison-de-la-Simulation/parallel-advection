#!/usr/bin/env python
import tensorflow as tf
import time
import numpy as np

real_t = tf.float64

# Device selection
gpus = tf.config.experimental.list_physical_devices('GPU')
device = "/GPU:0" if gpus else "/CPU:0"
print(f"Using device: {device}")

# Parameters
n0 = 16384
n1 = 512
n2 = 1
k = 3  # Kernel size
channel_out = 1
channel_in = 1

batch_size = n0 * n2
l = n1

# Input data
i0, i1, i2 = tf.meshgrid(
    tf.range(batch_size, dtype=real_t),
    tf.range(channel_in, dtype=real_t),
    tf.range(l, dtype=real_t),
    indexing="ij",
)
data = (i0 + i1 + i2) % 10
# print(f"Is data contiguous: {tf.experimental.numpy.isfortran(data.numpy())}")
data = tf.reshape(data, (batch_size, n1, channel_in))

# Weight and bias initialization
weight = tf.constant(1.5, shape=(k, channel_in, channel_out), dtype=real_t)
bias = tf.constant(1.0, shape=(channel_out,), dtype=real_t)

# Conv1D layer
def conv1d_layer(inputs):
    inputs = tf.cast(inputs, tf.float64)
    inputs = tf.expand_dims(inputs, axis=0)  # Add batch dimension
    conv_output = tf.nn.conv1d(inputs, weight, stride=1, padding='VALID')
    conv_output = tf.nn.bias_add(conv_output, bias)
    return tf.squeeze(conv_output, axis=0)

conv1d_scripted = tf.function(conv1d_layer)

def sum_and_normalize(data):
    return tf.reduce_sum(data).numpy() / tf.size(data).numpy()

# Compute initial sum and normalization
error_before = sum_and_normalize(data)
print(f"Normalized Array before: {error_before:.1f}")

for _ in range(3):
    _ = conv1d_scripted(data)

# Run convolution with timing
start_time = time.time()
output = conv1d_scripted(data)
elapsed_time = time.time() - start_time

# Compute final sum and normalization
error_after = sum_and_normalize(output)
print(f"Normalized Array after: {error_after:.1f}")

# Performance metrics
gcells = (n0 * n1 * n2) / elapsed_time / 1e9
throughput = gcells * tf.size(data).numpy() * data.dtype.size * 2

print("\nPERF_DIAGS:")
print(f"elapsed_time: {elapsed_time:.6f} s")
print(f"upd_cells_per_sec: {gcells:.6f} Gcell/sec")
print(f"estimated_throughput: {throughput:.6f} GB/s")
