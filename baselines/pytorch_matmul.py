"""
PyTorch baseline implementation for matrix multiplication.
This demonstrates the performance of optimized PyTorch operations on GPU.
"""

import torch
import time
import numpy as np
from typing import List, Dict


def pytorch_matmul_cpu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    PyTorch matrix multiplication on CPU
    
    Args:
        A: Tensor of shape (M, K)
        B: Tensor of shape (K, N)
    
    Returns:
        C: Result tensor of shape (M, N)
    """
    return torch.matmul(A, B)


def pytorch_matmul_gpu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    PyTorch matrix multiplication on GPU
    
    Args:
        A: Tensor of shape (M, K) on GPU
        B: Tensor of shape (K, N) on GPU
    
    Returns:
        C: Result tensor of shape (M, N) on GPU
    """
    return torch.matmul(A, B)


def benchmark_pytorch(sizes: List[int], num_runs: int = 10, device: str = 'cuda') -> Dict:
    """
    Benchmark PyTorch matrix multiplication
    
    Args:
        sizes: List of matrix sizes to test
        num_runs: Number of runs for averaging
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        Dictionary with benchmark results
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Benchmarking PyTorch on {device.upper()}")
    
    results = {}
    
    for size in sizes:
        print(f"\nBenchmarking PyTorch matmul for size {size}x{size} on {device}")
        
        # Generate random tensors
        A = torch.randn(size, size, dtype=torch.float32, device=device)
        B = torch.randn(size, size, dtype=torch.float32, device=device)
        
        # Warm up
        if device == 'cuda':
            torch.cuda.synchronize()
            for _ in range(3):
                _ = torch.matmul(A, B)
            torch.cuda.synchronize()
        else:
            _ = torch.matmul(A, B)
        
        # Benchmark
        times = []
        
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                C = torch.matmul(A, B)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
                C = torch.matmul(A, B)
                end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate FLOPS
        flops = 2 * size * size * size
        gflops = flops / (avg_time * 1e9)
        
        results[size] = {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'gflops': gflops,
            'implementation': f'pytorch_{device}',
            'device': device
        }
        
        print(f"  Time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Performance: {gflops:.2f} GFLOPS")
    
    return results


def compare_cpu_gpu(sizes: List[int]) -> None:
    """
    Compare PyTorch performance between CPU and GPU
    """
    print("Comparing PyTorch CPU vs GPU performance...")
    
    cpu_results = benchmark_pytorch(sizes, device='cpu')
    
    if torch.cuda.is_available():
        gpu_results = benchmark_pytorch(sizes, device='cuda')
        
        print("\n" + "="*60)
        print("PYTORCH CPU vs GPU COMPARISON")
        print("="*60)
        print(f"{'Size':<6} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
        print("-" * 60)
        
        for size in sizes:
            cpu_time = cpu_results[size]['avg_time_ms']
            gpu_time = gpu_results[size]['avg_time_ms']
            speedup = cpu_time / gpu_time
            
            print(f"{size:<6} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.2f}x")
    else:
        print("CUDA not available - showing CPU results only")
        print("\n" + "="*50)
        print("PYTORCH CPU BENCHMARK")
        print("="*50)
        for size, result in cpu_results.items():
            print(f"Size {size:4d}: {result['avg_time_ms']:8.2f} ms, {result['gflops']:6.2f} GFLOPS")


def test_correctness():
    """Test correctness against NumPy"""
    print("Testing PyTorch vs NumPy correctness...")
    
    # Small test matrices
    A_np = np.random.randn(4, 4).astype(np.float32)
    B_np = np.random.randn(4, 4).astype(np.float32)
    
    # NumPy result
    C_np = np.dot(A_np, B_np)
    
    # PyTorch CPU result
    A_torch = torch.from_numpy(A_np)
    B_torch = torch.from_numpy(B_np)
    C_torch_cpu = torch.matmul(A_torch, B_torch).numpy()
    
    print(f"NumPy vs PyTorch CPU difference: {np.max(np.abs(C_np - C_torch_cpu))}")
    assert np.allclose(C_np, C_torch_cpu, atol=1e-5), "CPU implementation mismatch!"
    
    # PyTorch GPU result (if available)
    if torch.cuda.is_available():
        A_gpu = A_torch.cuda()
        B_gpu = B_torch.cuda()
        C_torch_gpu = torch.matmul(A_gpu, B_gpu).cpu().numpy()
        
        print(f"NumPy vs PyTorch GPU difference: {np.max(np.abs(C_np - C_torch_gpu))}")
        assert np.allclose(C_np, C_torch_gpu, atol=1e-5), "GPU implementation mismatch!"
    
    print("✓ Correctness test passed!")


if __name__ == "__main__":
    # Test correctness
    test_correctness()
    
    # System info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Run benchmarks
    test_sizes = [64, 128, 256, 512, 1024]
    compare_cpu_gpu(test_sizes)
