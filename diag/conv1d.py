#!/usr/bin/env python
import torch
import time

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
n0 = 16384
n1 = 1024
n2 = 1
k = 3  # Kernel size
channel_out = 1
channel_in = 1

batch_size=n0
l=n1

i0, i1, i2 = torch.meshgrid(
    torch.arange(batch_size, device=device, dtype=torch.float64),
    torch.arange(channel_in, device=device, dtype=torch.float64),
    torch.arange(l, device=device, dtype=torch.float64),
    indexing="ij",
)
data = (i0 + i1 + i2) % 10
print(f"Is data contiguous: {data.is_contiguous()}")

# Weight and bias initialization
weight = torch.full((channel_out, channel_in, k), 1.5, dtype=torch.float64, device=device)
bias = torch.full((channel_out,), 1.0, dtype=torch.float64, device=device)

# Conv1D layer
conv1d = torch.nn.Conv1d(in_channels=channel_in, out_channels=channel_out, kernel_size=k, bias=True, dtype=torch.float64).to(device)
conv1d.weight.data = weight
conv1d.bias.data = bias

# Compile model with TorchScript
conv1d_scripted = torch.jit.trace(conv1d, data)

def sum_and_normalize(data):
    return data.sum().item() / data.numel()

# Compute initial sum and normalization
error_before = sum_and_normalize(data)
print(f"Normalized Array before: {error_before}")

# Run convolution
start_time = time.time()
output = conv1d_scripted(data)  # Use pre-compiled model
elapsed_time = time.time() - start_time

# Compute final sum and normalization
error_after = sum_and_normalize(output)
print(f"Normalized Array after: {error_after}")

# Performance metrics
gcells = (n0 * n1 * n2) / elapsed_time / 1e9
throughput = gcells * data.element_size() * 2

print("\nPERF_DIAGS:")
print(f"elapsed_time: {elapsed_time:.6f} s")
print(f"upd_cells_per_sec: {gcells:.6f} Gcell/sec")
print(f"estimated_throughput: {throughput:.6f} GB/s")
