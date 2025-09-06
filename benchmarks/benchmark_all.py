"""
Comprehensive benchmark comparing all matrix multiplication implementations
This script provides detailed performance analysis and scaling behavior
"""

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

# Add paths for our implementations
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baselines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pytorch_extension'))

# Import baseline implementations
try:
    from numpy_matmul import numpy_matmul, benchmark_numpy
    NUMPY_AVAILABLE = True
except ImportError:
    print("NumPy baseline not available")
    NUMPY_AVAILABLE = False

try:
    from pytorch_matmul import benchmark_pytorch
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available")
    PYTORCH_AVAILABLE = False

try:
    import matmul_cuda
    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    print("CUDA extension not available - please build it first")
    CUDA_EXTENSION_AVAILABLE = False


class BenchmarkSuite:
    """Comprehensive benchmarking suite for matrix multiplication implementations"""
    
    def __init__(self):
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def run_all_benchmarks(self, sizes: List[int], num_runs: int = 10) -> Dict:
        """Run benchmarks for all available implementations"""
        print("=== COMPREHENSIVE MATRIX MULTIPLICATION BENCHMARK ===")
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Test sizes: {sizes}")
        print(f"Runs per test: {num_runs}")
        print()
        
        all_results = {}
        
        for size in sizes:
            print(f"\n{'='*60}")
            print(f"BENCHMARKING {size}x{size} MATRICES")
            print(f"{'='*60}")
            
            size_results = {}
            
            # NumPy baseline (CPU)
            if NUMPY_AVAILABLE:
                print("Running NumPy (CPU) baseline...")
                numpy_result = self.benchmark_numpy_implementation(size, num_runs)
                size_results['numpy_cpu'] = numpy_result
            
            # PyTorch CPU
            if PYTORCH_AVAILABLE:
                print("Running PyTorch CPU...")
                pytorch_cpu_result = self.benchmark_pytorch_implementation(size, num_runs, 'cpu')
                size_results['pytorch_cpu'] = pytorch_cpu_result
                
                # PyTorch GPU
                if self.device == 'cuda':
                    print("Running PyTorch CUDA...")
                    pytorch_gpu_result = self.benchmark_pytorch_implementation(size, num_runs, 'cuda')
                    size_results['pytorch_cuda'] = pytorch_gpu_result
            
            # Our CUDA implementations
            if CUDA_EXTENSION_AVAILABLE and self.device == 'cuda':
                print("Running custom CUDA implementations...")
                
                cuda_implementations = [
                    ('naive_cuda', matmul_cuda.naive_matmul),
                    ('tiled_cuda', matmul_cuda.tiled_matmul),
                    ('optimized_cuda', matmul_cuda.optimized_matmul),
                    ('smart_cuda', matmul_cuda.smart_matmul)
                ]
                
                for name, func in cuda_implementations:
                    try:
                        result = self.benchmark_cuda_implementation(size, num_runs, func, name)
                        size_results[name] = result
                    except Exception as e:
                        print(f"  {name} failed: {e}")
            
            all_results[size] = size_results
            self.print_size_summary(size, size_results)
        
        self.results = all_results
        return all_results
    
    def benchmark_numpy_implementation(self, size: int, num_runs: int) -> Dict:
        """Benchmark NumPy implementation"""
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Warm up
        _ = numpy_matmul(A, B)
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            C = numpy_matmul(A, B)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        flops = 2 * size * size * size
        gflops = flops / (avg_time * 1e6)
        
        print(f"  NumPy CPU: {avg_time:8.2f} ± {std_time:5.2f} ms, {gflops:6.2f} GFLOPS")
        
        return {
            'time_ms': avg_time,
            'std_ms': std_time,
            'gflops': gflops,
            'device': 'cpu'
        }
    
    def benchmark_pytorch_implementation(self, size: int, num_runs: int, device: str) -> Dict:
        """Benchmark PyTorch implementation"""
        A = torch.randn(size, size, dtype=torch.float32, device=device)
        B = torch.randn(size, size, dtype=torch.float32, device=device)
        
        # Warm up
        if device == 'cuda':
            torch.cuda.synchronize()
        for _ in range(3):
            _ = torch.matmul(A, B)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            C = torch.matmul(A, B)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        flops = 2 * size * size * size
        gflops = flops / (avg_time * 1e6)
        
        print(f"  PyTorch {device.upper()}: {avg_time:8.2f} ± {std_time:5.2f} ms, {gflops:6.2f} GFLOPS")
        
        return {
            'time_ms': avg_time,
            'std_ms': std_time,
            'gflops': gflops,
            'device': device
        }
    
    def benchmark_cuda_implementation(self, size: int, num_runs: int, func, name: str) -> Dict:
        """Benchmark our CUDA implementations"""
        A = torch.randn(size, size, dtype=torch.float32, device='cuda')
        B = torch.randn(size, size, dtype=torch.float32, device='cuda')
        
        # Warm up
        torch.cuda.synchronize()
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
        flops = 2 * size * size * size
        gflops = flops / (avg_time * 1e6)
        
        print(f"  {name:<15}: {avg_time:8.2f} ± {std_time:5.2f} ms, {gflops:6.2f} GFLOPS")
        
        return {
            'time_ms': avg_time,
            'std_ms': std_time,
            'gflops': gflops,
            'device': 'cuda'
        }
    
    def print_size_summary(self, size: int, results: Dict):
        """Print summary for a specific matrix size"""
        print(f"\n--- Summary for {size}x{size} ---")
        
        # Find best performance
        best_gflops = 0
        best_impl = ""
        
        for impl, result in results.items():
            if result['gflops'] > best_gflops:
                best_gflops = result['gflops']
                best_impl = impl
        
        print(f"Best performance: {best_impl} at {best_gflops:.2f} GFLOPS")
        
        # Calculate speedups relative to NumPy
        if 'numpy_cpu' in results:
            numpy_time = results['numpy_cpu']['time_ms']
            print("Speedups vs NumPy:")
            for impl, result in results.items():
                if impl != 'numpy_cpu':
                    speedup = numpy_time / result['time_ms']
                    print(f"  {impl}: {speedup:.2f}x")
    
    def generate_performance_plots(self, save_dir: str = "benchmark_plots"):
        """Generate performance visualization plots"""
        if not self.results:
            print("No benchmark results to plot")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data for plotting
        sizes = sorted(self.results.keys())
        implementations = set()
        for size_results in self.results.values():
            implementations.update(size_results.keys())
        implementations = sorted(implementations)
        
        # Performance vs Matrix Size
        plt.figure(figsize=(12, 8))
        
        for impl in implementations:
            gflops_data = []
            size_data = []
            
            for size in sizes:
                if impl in self.results[size]:
                    gflops_data.append(self.results[size][impl]['gflops'])
                    size_data.append(size)
            
            if gflops_data:
                plt.plot(size_data, gflops_data, marker='o', label=impl, linewidth=2)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Performance (GFLOPS)')
        plt.title('Matrix Multiplication Performance vs Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(os.path.join(save_dir, 'performance_vs_size.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Speedup comparison
        if 'numpy_cpu' in implementations:
            plt.figure(figsize=(12, 8))
            
            for impl in implementations:
                if impl == 'numpy_cpu':
                    continue
                
                speedups = []
                size_data = []
                
                for size in sizes:
                    if impl in self.results[size] and 'numpy_cpu' in self.results[size]:
                        numpy_time = self.results[size]['numpy_cpu']['time_ms']
                        impl_time = self.results[size][impl]['time_ms']
                        speedup = numpy_time / impl_time
                        speedups.append(speedup)
                        size_data.append(size)
                
                if speedups:
                    plt.plot(size_data, speedups, marker='o', label=impl, linewidth=2)
            
            plt.xlabel('Matrix Size')
            plt.ylabel('Speedup vs NumPy')
            plt.title('Speedup Comparison vs NumPy Baseline')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig(os.path.join(save_dir, 'speedup_vs_numpy.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Performance plots saved to {save_dir}/")
    
    def generate_report(self, filename: str = "benchmark_report.csv"):
        """Generate CSV report of all results"""
        if not self.results:
            print("No benchmark results to report")
            return
        
        # Flatten results into DataFrame
        rows = []
        for size, size_results in self.results.items():
            for impl, result in size_results.items():
                rows.append({
                    'matrix_size': size,
                    'implementation': impl,
                    'time_ms': result['time_ms'],
                    'std_ms': result['std_ms'],
                    'gflops': result['gflops'],
                    'device': result['device']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Benchmark report saved to {filename}")
        
        # Print summary statistics
        print("\n=== BENCHMARK SUMMARY ===")
        pivot = df.pivot(index='matrix_size', columns='implementation', values='gflops')
        print(pivot.to_string(float_format='%.2f'))


def main():
    """Main benchmarking routine"""
    benchmark = BenchmarkSuite()
    
    # Test different matrix sizes
    sizes = [64, 128, 256, 512, 1024]
    
    # Add larger sizes if we have powerful GPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name().lower()
        if any(gpu in device_name for gpu in ['a100', 'v100', 'rtx', 'titan']):
            sizes.extend([2048, 4096])
    
    # Run benchmarks
    results = benchmark.run_all_benchmarks(sizes, num_runs=10)
    
    # Generate outputs
    benchmark.generate_performance_plots()
    benchmark.generate_report()
    
    print("\n=== BENCHMARK COMPLETE ===")
    print("Results saved to:")
    print("  - benchmark_report.csv")
    print("  - benchmark_plots/performance_vs_size.png")
    print("  - benchmark_plots/speedup_vs_numpy.png")


if __name__ == "__main__":
    main()
