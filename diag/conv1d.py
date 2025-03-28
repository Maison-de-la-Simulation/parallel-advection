#!/usr/bin/env python
import torch
import time

real_t = torch.float64

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(device)}")

# Parameters

channel_in  = 3
channel_out = channel_in
length = 512

n0 = 512
n1 = length*channel_out
n2 = 512
batch_size=n0*n2

k = 1

# i0, i1, i2 = torch.meshgrid(
#     torch.arange(batch_size, device=device, dtype=real_t),
#     torch.arange(channel_in, device=device, dtype=real_t),
#     torch.arange(length, device=device, dtype=real_t),
#     indexing="ij",
# )
# data = (i0 + i1 + i2) % 10
data = torch.full((batch_size, channel_in, length), 7.3, dtype=real_t, device=device)

print(f"Is data contiguous: {data.is_contiguous()}")

# Weight and bias initialization
weight = torch.full((channel_out, channel_in, k), 1.5, dtype=real_t, device=device)
bias = torch.full((channel_out,), 1.0, dtype=real_t, device=device)

# Conv1D layer
conv1d = torch.nn.Conv1d(in_channels=channel_in, out_channels=channel_out, kernel_size=k, bias=True, dtype=real_t).to(device)
conv1d.weight.data = weight
conv1d.bias.data = bias

# Compile model with TorchScript
conv1d_scripted = torch.jit.trace(conv1d, data)

def sum_and_normalize(data):
    return data.sum().item() / data.numel()

# Compute initial sum and normalization
error_before = sum_and_normalize(data)
print(f"Normalized Array before: {error_before:.1f}")

for _ in range(3):
    _ = conv1d_scripted(data)
    
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
# Run convolution
start.record()
output = conv1d_scripted(data)  # Use pre-compiled model
torch.cuda.synchronize()
end.record()
elapsed_time = start.elapsed_time(end)/1000

# Compute final sum and normalization
error_after = sum_and_normalize(output)
print(f"Normalized Array after: {error_after:.1f}")

# Performance metrics
gcells = (n0 * n1 * n2) / elapsed_time / 1e9
throughput = gcells * data.element_size() * 2

print("\nPERF_DIAGS:")
print(f"elapsed_time: {elapsed_time:.6f} s")
print(f"upd_cells_per_sec: {gcells:.6f} Gcell/sec")
print(f"estimated_throughput: {throughput:.6f} GB/s")
