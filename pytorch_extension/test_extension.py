"""
Test script for the PyTorch CUDA extension
Verifies correctness and benchmarks performance against PyTorch's built-in operations
"""

import torch
import numpy as np
import time
import sys
import os

# Add the extension to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import matmul_cuda
    EXTENSION_AVAILABLE = True
    print("✓ Custom CUDA extension loaded successfully!")
except ImportError as e:
    print(f"✗ Failed to import custom CUDA extension: {e}")
    print("Please build the extension first using: python setup.py build_ext --inplace")
    EXTENSION_AVAILABLE = False


def test_correctness():
    """Test correctness of all implementations against PyTorch"""
    print("\n=== Correctness Testing ===")
    
    if not EXTENSION_AVAILABLE:
        print("Skipping correctness tests - extension not available")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available - skipping tests")
        return
    
    # Test with small matrices first
    sizes = [32, 64, 128]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices...")
        
        # Generate test matrices
        A = torch.randn(size, size, device=device, dtype=torch.float32)
        B = torch.randn(size, size, device=device, dtype=torch.float32)
        
        # PyTorch reference
        C_torch = torch.matmul(A, B)
        
        # Test our implementations
        implementations = [
            ('naive', matmul_cuda.naive_matmul),
            ('tiled', matmul_cuda.tiled_matmul),
            ('optimized', matmul_cuda.optimized_matmul),
            ('smart', matmul_cuda.smart_matmul)
        ]
        
        for name, func in implementations:
            try:
                C_custom = func(A, B)
                
                # Check shapes
                assert C_custom.shape == C_torch.shape, f"Shape mismatch in {name}"
                
                # Check values (allowing for floating point differences)
                max_diff = torch.max(torch.abs(C_custom - C_torch)).item()
                relative_error = max_diff / torch.max(torch.abs(C_torch)).item()
                
                print(f"  {name:>10}: max_diff = {max_diff:.2e}, relative_error = {relative_error:.2e}")
                
                if relative_error > 1e-4:
                    print(f"    ⚠️  Large error detected in {name} implementation!")
                else:
                    print(f"    ✓ {name} implementation passed")
                    
            except Exception as e:
                print(f"    ✗ {name} implementation failed: {e}")


def benchmark_implementations():
    """Benchmark all implementations against PyTorch"""
    print("\n=== Performance Benchmarking ===")
    
    if not EXTENSION_AVAILABLE:
        print("Skipping benchmarks - extension not available")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available - skipping benchmarks")
        return
    
    sizes = [128, 256, 512, 1024]
    num_runs = 10
    
    print(f"Benchmarking on {torch.cuda.get_device_name()}")
    print(f"Number of runs per test: {num_runs}")
    print()
    
    results = {}
    
    for size in sizes:
        print(f"Size: {size}x{size}")
        print("-" * 60)
        
        # Generate test matrices
        A = torch.randn(size, size, device=device, dtype=torch.float32)
        B = torch.randn(size, size, device=device, dtype=torch.float32)
        
        # Warm up GPU
        for _ in range(3):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        size_results = {}
        
        # Test PyTorch baseline
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            C = torch.matmul(A, B)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        std_time = np.std(times) * 1000
        flops = 2 * size * size * size
        gflops = flops / (avg_time * 1e6)
        
        size_results['pytorch'] = {'time': avg_time, 'std': std_time, 'gflops': gflops}
        print(f"{'PyTorch':<12}: {avg_time:8.2f} ± {std_time:5.2f} ms, {gflops:6.2f} GFLOPS")
        
        # Test our implementations
        implementations = [
            ('naive', matmul_cuda.naive_matmul),
            ('tiled', matmul_cuda.tiled_matmul),
            ('optimized', matmul_cuda.optimized_matmul),
            ('smart', matmul_cuda.smart_matmul)
        ]
        
        for name, func in implementations:
            try:
                # Warm up
                for _ in range(3):
                    _ = func(A, B)
                torch.cuda.synchronize()
                
                times = []
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    C = func(A, B)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append(end - start)
                
                avg_time = np.mean(times) * 1000
                std_time = np.std(times) * 1000
                gflops = flops / (avg_time * 1e6)
                speedup = size_results['pytorch']['time'] / avg_time
                
                size_results[name] = {'time': avg_time, 'std': std_time, 'gflops': gflops, 'speedup': speedup}
                print(f"{name:<12}: {avg_time:8.2f} ± {std_time:5.2f} ms, {gflops:6.2f} GFLOPS, {speedup:5.2f}x")
                
            except Exception as e:
                print(f"{name:<12}: Failed - {e}")
        
        results[size] = size_results
        print()
    
    # Summary
    print("=== BENCHMARK SUMMARY ===")
    print(f"{'Size':<6} {'PyTorch':<10} {'Naive':<10} {'Tiled':<10} {'Optimized':<12} {'Smart':<10}")
    print("-" * 70)
    
    for size in sizes:
        row = f"{size:<6}"
        for impl in ['pytorch', 'naive', 'tiled', 'optimized', 'smart']:
            if impl in results[size]:
                gflops = results[size][impl]['gflops']
                row += f"{gflops:>10.1f}"
            else:
                row += f"{'N/A':>10}"
        print(row)


def test_batched_operations():
    """Test batched matrix multiplication for transformer workloads"""
    print("\n=== Batched Operations Testing ===")
    
    if not EXTENSION_AVAILABLE:
        print("Skipping batched tests - extension not available")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available - skipping tests")
        return
    
    # Simulate transformer attention computation: Q @ K^T
    batch_size = 8
    seq_len = 128
    hidden_dim = 512
    
    print(f"Testing attention-like computation:")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Hidden dim: {hidden_dim}")
    
    # Generate test tensors (Q and K for attention)
    Q = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)
    K_T = K.transpose(-2, -1)  # Transpose for attention computation
    
    # PyTorch batched matmul
    torch.cuda.synchronize()
    start = time.perf_counter()
    attention_torch = torch.bmm(Q, K_T)
    torch.cuda.synchronize()
    end = time.perf_counter()
    pytorch_time = (end - start) * 1000
    
    print(f"PyTorch bmm: {pytorch_time:.2f} ms")
    
    # Our batched implementation
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        attention_custom = matmul_cuda.batched_matmul(Q, K_T, "optimized")
        torch.cuda.synchronize()
        end = time.perf_counter()
        custom_time = (end - start) * 1000
        
        # Check correctness
        max_diff = torch.max(torch.abs(attention_custom - attention_torch)).item()
        print(f"Custom batched: {custom_time:.2f} ms")
        print(f"Max difference: {max_diff:.2e}")
        print(f"Speedup: {pytorch_time / custom_time:.2f}x")
        
    except Exception as e:
        print(f"Custom batched failed: {e}")


def test_gradient_compatibility():
    """Test that our implementations work with PyTorch's autograd"""
    print("\n=== Gradient Compatibility Testing ===")
    
    if not EXTENSION_AVAILABLE:
        print("Skipping gradient tests - extension not available")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available - skipping tests")
        return
    
    size = 64
    A = torch.randn(size, size, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn(size, size, device=device, dtype=torch.float32, requires_grad=True)
    
    # Test with our optimized implementation
    try:
        C = matmul_cuda.optimized_matmul(A, B)
        loss = torch.sum(C)
        loss.backward()
        
        print("✓ Gradient computation successful")
        print(f"A.grad shape: {A.grad.shape}")
        print(f"B.grad shape: {B.grad.shape}")
        print(f"A.grad norm: {torch.norm(A.grad).item():.4f}")
        print(f"B.grad norm: {torch.norm(B.grad).item():.4f}")
        
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")


if __name__ == "__main__":
    print("PyTorch CUDA Extension Test Suite")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("CUDA not available")
    
    # Run all tests
    test_correctness()
    benchmark_implementations()
    test_batched_operations()
    test_gradient_compatibility()
    
    print("\n=== Test Suite Complete ===")
