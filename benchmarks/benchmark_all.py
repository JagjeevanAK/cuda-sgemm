"""
Comprehensive benchmark comparing matrix multiplication implementations:
- NumPy (CPU baseline)
- PyTorch (CPU and CUDA)
- Custom CUDA kernels (naive, tiled, optimized)
"""

import numpy as np
import time
import sys
import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baselines'))

# Import baseline implementations
try:
    from numpy_matmul import numpy_matmul
    NUMPY_AVAILABLE = True
except ImportError:
    print("NumPy baseline not available")
    NUMPY_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available")
    PYTORCH_AVAILABLE = False

# Check if CUDA kernels are built
BUILD_DIR = os.path.join(os.path.dirname(__file__), '..', 'build')
CUDA_KERNELS = {
    'naive_cuda': os.path.join(BUILD_DIR, 'naive_matmul'),
    'tiled_cuda': os.path.join(BUILD_DIR, 'tiled_matmul'),
    'optimized_cuda': os.path.join(BUILD_DIR, 'optimized_matmul')
}

# Check which CUDA kernels are available
AVAILABLE_CUDA_KERNELS = {}
for name, path in CUDA_KERNELS.items():
    if os.path.exists(path):
        AVAILABLE_CUDA_KERNELS[name] = path
    else:
        print(f"CUDA kernel {name} not found at {path}")

CUDA_KERNELS_AVAILABLE = len(AVAILABLE_CUDA_KERNELS) > 0


class BenchmarkSuite:
    """Comprehensive benchmarking suite for matrix multiplication implementations"""
    
    def __init__(self):
        self.results = {}
        self.device = 'cuda' if PYTORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
    def run_all_benchmarks(self, sizes: List[int], num_runs: int = 10) -> Dict:
        """Run benchmarks for all available implementations"""
        print("=== COMPREHENSIVE MATRIX MULTIPLICATION BENCHMARK ===")
        print(f"Device: {self.device}")
        if self.device == 'cuda' and PYTORCH_AVAILABLE:
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
            if CUDA_KERNELS_AVAILABLE and self.device == 'cuda':
                print("Running custom CUDA implementations...")
                
                for name, executable_path in AVAILABLE_CUDA_KERNELS.items():
                    try:
                        result = self.benchmark_cuda_kernel(size, num_runs, executable_path, name)
                        size_results[name] = result
                    except Exception as e:
                        print(f"  {name} failed: {e}")
            
            all_results[size] = size_results
            self.print_size_summary(size, size_results)
        
        self.results = all_results
        return all_results
    
    def benchmark_numpy_implementation(self, size: int, num_runs: int) -> Dict:
        """Benchmark NumPy implementation"""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy not available")
            
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
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
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
    
    def benchmark_cuda_kernel(self, size: int, num_runs: int, executable_path: str, name: str) -> Dict:
        """Benchmark CUDA kernel by running the executable and parsing output"""
        try:
            # Run the CUDA executable - assuming it outputs timing information
            result = subprocess.run([executable_path], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise RuntimeError(f"CUDA kernel failed: {result.stderr}")
            
            # Parse output to extract timing information
            # This is a simplified approach - you may need to modify based on actual output format
            output_lines = result.stdout.strip().split('\n')
            
            # Look for performance information in the output
            gflops = 0.0
            time_ms = 0.0
            
            for line in output_lines:
                if 'GFLOPS' in line and str(size) in line:
                    # Try to extract GFLOPS value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'GFLOPS' in part and i > 0:
                            try:
                                gflops = float(parts[i-1])
                            except ValueError:
                                pass
                        if 'ms' in part and i > 0:
                            try:
                                time_ms = float(parts[i-1])
                            except ValueError:
                                pass
            
            # If we couldn't parse the output, use default values
            if gflops == 0.0 or time_ms == 0.0:
                print(f"  {name:<15}: Could not parse performance data from output")
                return {
                    'time_ms': 999999.0,
                    'std_ms': 0.0,
                    'gflops': 0.0,
                    'device': 'cuda'
                }
            
            print(f"  {name:<15}: {time_ms:8.2f} ms, {gflops:6.2f} GFLOPS")
            
            return {
                'time_ms': time_ms,
                'std_ms': 0.0,  # Standalone executables don't provide std dev
                'gflops': gflops,
                'device': 'cuda'
            }
            
        except subprocess.TimeoutExpired:
            print(f"  {name:<15}: Timeout")
            return {
                'time_ms': 999999.0,
                'std_ms': 0.0,
                'gflops': 0.0,
                'device': 'cuda'
            }
        except Exception as e:
            print(f"  {name:<15}: Error - {e}")
            return {
                'time_ms': 999999.0,
                'std_ms': 0.0,
                'gflops': 0.0,
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
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name().lower()
            if any(gpu in device_name for gpu in ['a100', 'v100', 'rtx', 'titan']):
                sizes.extend([2048, 4096])
        except:
            pass  # If we can't get device name, just use default sizes
    
    # Run benchmarks
    results = benchmark.run_all_benchmarks(sizes, num_runs=10)
    
    # Generate outputs
    benchmark.generate_performance_plots()
    benchmark.generate_report()
    
    print("\nCOMPLETE BENCHMARK\n\n")
    print("Results saved to:")
    print("  - benchmark_report.csv")
    print("  - benchmark_plots/performance_vs_size.png")
    print("  - benchmark_plots/speedup_vs_numpy.png")


if __name__ == "__main__":
    main()
