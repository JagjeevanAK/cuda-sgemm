"""
Pure NumPy baseline implementation for matrix multiplication.
This serves as our CPU baseline for performance comparison.
"""

import numpy as np
import time
from typing import Tuple, List


def numpy_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pure NumPy matrix multiplication using np.dot
    
    Args:
        A: Matrix of shape (M, K)
        B: Matrix of shape (K, N)
    
    Returns:
        C: Result matrix of shape (M, N)
    """
    return np.dot(A, B)


def numpy_matmul_manual(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Manual implementation using nested loops (very slow, for educational purposes)
    
    Args:
        A: Matrix of shape (M, K)
        B: Matrix of shape (K, N)
    
    Returns:
        C: Result matrix of shape (M, N)
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"
    
    C = np.zeros((M, N), dtype=np.float32)
    
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    
    return C


def benchmark_numpy(sizes: List[int], num_runs: int = 5) -> dict:
    """
    Benchmark NumPy matrix multiplication across different sizes
    
    Args:
        sizes: List of matrix sizes to test (square matrices)
        num_runs: Number of runs for averaging
    
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for size in sizes:
        print(f"\nBenchmarking NumPy matmul for size {size}x{size}")
        
        # Generate random matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Warm up
        _ = numpy_matmul(A, B)
        
        # Benchmark optimized NumPy
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            C = numpy_matmul(A, B)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate FLOPS (2 * M * N * K operations for matrix multiplication)
        flops = 2 * size * size * size
        gflops = flops / (avg_time * 1e9)
        
        results[size] = {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'gflops': gflops,
            'implementation': 'numpy'
        }
        
        print(f"  Time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Performance: {gflops:.2f} GFLOPS")
    
    return results


if __name__ == "__main__":
    # Test correctness first
    print("Testing correctness...")
    A = np.random.randn(4, 4).astype(np.float32)
    B = np.random.randn(4, 4).astype(np.float32)
    
    C_numpy = numpy_matmul(A, B)
    C_manual = numpy_matmul_manual(A, B)
    
    print(f"NumPy vs Manual difference: {np.max(np.abs(C_numpy - C_manual))}")
    assert np.allclose(C_numpy, C_manual, atol=1e-5), "Implementation mismatch!"
    print("✓ Correctness test passed!")
    
    # Run benchmarks
    test_sizes = [64, 128, 256, 512, 1024]
    results = benchmark_numpy(test_sizes)
    
    print("\n" + "="*50)
    print("NUMPY BENCHMARK SUMMARY")
    print("="*50)
    for size, result in results.items():
        print(f"Size {size:4d}: {result['avg_time_ms']:8.2f} ms, {result['gflops']:6.2f} GFLOPS")
