"""
Script per verificare che PyTorch usa la GPU (CUDA) correttamente.
Esegui questo script MANUALMENTE dopo aver installato le dipendenze:

    .siamese\Scripts\python check_gpu_pytorch.py
"""

import torch
import time
import numpy as np
from datetime import datetime

print("=" * 90)
print("PyTorch CUDA/GPU Verification Script")
print("=" * 90)
print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. INFORMAZIONI AMBIENTE
# ============================================================================
print("1. ENVIRONMENT INFORMATION")
print("-" * 90)
print(f"PyTorch version:        {torch.__version__}")
print(f"Python version:         {torch.__version__.split('+')[0]}")
print(f"NumPy version:          {np.__version__}")

# ============================================================================
# 2. VERIFICA CUDA
# ============================================================================
print("\n2. CUDA AVAILABILITY CHECK")
print("-" * 90)

cuda_available = torch.cuda.is_available()
print(f"CUDA available:         {cuda_available}")

if cuda_available:
    print(f"CUDA version:           {torch.version.cuda}")
    print(f"cuDNN version:          {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled:          {torch.backends.cudnn.enabled}")
else:
    print("⚠️  WARNING: CUDA not available! GPU testing will be skipped.")

# ============================================================================
# 3. INFORMAZIONI GPU
# ============================================================================
print("\n3. GPU DEVICE INFORMATION")
print("-" * 90)

if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs:         {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i}:")
        print(f"  Name:                 {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability:   {torch.cuda.get_device_capability(i)}")
        print(f"  Total Memory:         {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

        # Current GPU
        if torch.cuda.current_device() == i:
            print(f"  Status:               ✓ ACTIVE (current device)")

    print(f"\nCurrent GPU:            GPU {torch.cuda.current_device()} ({torch.cuda.get_device_name()})")
else:
    print("❌ No GPU devices found")

# ============================================================================
# 4. TEST DI PERFORMANCE
# ============================================================================
print("\n4. PERFORMANCE TEST - Matrix Multiplication")
print("-" * 90)

# Parametri test
matrix_size = 4096
num_iterations = 5
print(f"Matrix size:            {matrix_size} x {matrix_size}")
print(f"Iterations:             {num_iterations}")
print(f"Operation:              Matrix @ Matrix (matmul)")

# ---- TEST CPU ----
print(f"\n{'Test CPU':^90}")
print("-" * 90)

device_cpu = torch.device('cpu')
torch.cuda.empty_cache() if cuda_available else None
torch.manual_seed(42)

# Crea matrici sul CPU
A_cpu = torch.randn(matrix_size, matrix_size, device=device_cpu, dtype=torch.float32)
B_cpu = torch.randn(matrix_size, matrix_size, device=device_cpu, dtype=torch.float32)

# Warmup
for _ in range(2):
    _ = torch.matmul(A_cpu, B_cpu)

# Misurazione
cpu_times = []
for i in range(num_iterations):
    start = time.perf_counter()
    C_cpu = torch.matmul(A_cpu, B_cpu)
    torch.cuda.synchronize() if cuda_available else None  # Sincronizza se CUDA è disponibile
    end = time.perf_counter()
    elapsed = (end - start) * 1000  # Converti a ms
    cpu_times.append(elapsed)
    print(f"  Iteration {i+1}: {elapsed:.2f} ms")

cpu_mean = np.mean(cpu_times)
cpu_std = np.std(cpu_times)
print(f"\n  Mean time:            {cpu_mean:.2f} ± {cpu_std:.2f} ms")

# ---- TEST GPU ----
if cuda_available:
    print(f"\n{'Test GPU (CUDA)':^90}")
    print("-" * 90)

    device_gpu = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.manual_seed(42)

    # Crea matrici sulla GPU
    A_gpu = torch.randn(matrix_size, matrix_size, device=device_gpu, dtype=torch.float32)
    B_gpu = torch.randn(matrix_size, matrix_size, device=device_gpu, dtype=torch.float32)

    # Warmup
    for _ in range(2):
        _ = torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()

    # Misurazione
    gpu_times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C_gpu = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = (end - start) * 1000  # Converti a ms
        gpu_times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f} ms")

    gpu_mean = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    print(f"\n  Mean time:            {gpu_mean:.2f} ± {gpu_std:.2f} ms")

    # ============================================================================
    # 5. COMPARISON
    # ============================================================================
    print("\n5. PERFORMANCE COMPARISON")
    print("-" * 90)

    speedup = cpu_mean / gpu_mean
    time_saved = cpu_mean - gpu_mean

    print(f"CPU time (mean):        {cpu_mean:.2f} ms")
    print(f"GPU time (mean):        {gpu_mean:.2f} ms")
    print(f"Speedup (CPU/GPU):      {speedup:.2f}x")
    print(f"Time saved:             {time_saved:.2f} ms ({(time_saved/cpu_mean)*100:.1f}%)")

    if speedup > 1.5:
        print(f"\n✅ GPU is significantly faster than CPU!")
    else:
        print(f"\n⚠️  GPU speedup is minimal. Check GPU memory and utilization.")
else:
    print("\n❌ GPU tests skipped (CUDA not available)")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

if cuda_available:
    print("✓ CUDA is available and working")
    print(f"✓ GPU: {torch.cuda.get_device_name()}")
    print(f"✓ Performance: GPU is {speedup:.2f}x faster than CPU for matrix operations")
    print("\nPyTorch is correctly configured for GPU computation!")
else:
    print("❌ CUDA is NOT available")
    print("⚠️  Check your installation:")
    print("   1. Verify NVIDIA drivers are installed")
    print("   2. Reinstall PyTorch with CUDA support:")
    print("      pip install torch --index-url https://download.pytorch.org/whl/cu121")

print("=" * 90 + "\n")
