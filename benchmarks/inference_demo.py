"""
ML Inference Demo: Transformer Block Acceleration
Demonstrates how custom CUDA kernels can speed up real ML inference workloads
"""

import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os

# Add extension path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pytorch_extension'))

try:
    import matmul_cuda
    CUSTOM_KERNELS_AVAILABLE = True
    print("✓ Custom CUDA kernels loaded")
except ImportError:
    CUSTOM_KERNELS_AVAILABLE = False
    print("✗ Custom CUDA kernels not available")


class OptimizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using our custom CUDA kernels
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, implementation: str = "optimized"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.implementation = implementation
        
        # Initialize weights and bias same as nn.Linear
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not CUSTOM_KERNELS_AVAILABLE:
            # Fallback to standard PyTorch
            return torch.nn.functional.linear(input, self.weight, self.bias)
        
        # Use our custom implementation
        if self.implementation == "naive":
            output = matmul_cuda.naive_matmul(input, self.weight.t())
        elif self.implementation == "tiled":
            output = matmul_cuda.tiled_matmul(input, self.weight.t())
        elif self.implementation == "smart":
            output = matmul_cuda.smart_matmul(input, self.weight.t())
        else:  # optimized
            output = matmul_cuda.optimized_matmul(input, self.weight.t())
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class OptimizedMultiHeadAttention(nn.Module):
    """
    Multi-head attention using custom CUDA kernels for Q@K^T and Attention@V
    """
    def __init__(self, d_model: int, num_heads: int, implementation: str = "optimized"):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.implementation = implementation
        
        # Linear projections for Q, K, V
        self.q_proj = OptimizedLinear(d_model, d_model, implementation=implementation)
        self.k_proj = OptimizedLinear(d_model, d_model, implementation=implementation)
        self.v_proj = OptimizedLinear(d_model, d_model, implementation=implementation)
        self.out_proj = OptimizedLinear(d_model, d_model, implementation=implementation)
        
        self.scale = np.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.q_proj(x)  # [batch, seq_len, d_model]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation using custom kernels
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.out_proj(attention_output)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        if CUSTOM_KERNELS_AVAILABLE:
            # Use custom matmul for attention computation
            # Q @ K^T
            K_T = K.transpose(-2, -1)
            
            # Reshape for batched matrix multiplication
            Q_flat = Q.view(-1, seq_len, d_k)
            K_T_flat = K_T.view(-1, d_k, seq_len)
            
            if self.implementation == "optimized":
                attention_scores = matmul_cuda.batched_matmul(Q_flat, K_T_flat, "optimized")
            else:
                attention_scores = matmul_cuda.batched_matmul(Q_flat, K_T_flat, self.implementation)
            
            attention_scores = attention_scores.view(batch_size, num_heads, seq_len, seq_len)
            attention_scores = attention_scores / self.scale
            
            # Apply softmax
            attention_weights = torch.softmax(attention_scores, dim=-1)
            
            # Attention @ V
            V_flat = V.view(-1, seq_len, d_k)
            attention_weights_flat = attention_weights.view(-1, seq_len, seq_len)
            
            if self.implementation == "optimized":
                attention_output = matmul_cuda.batched_matmul(attention_weights_flat, V_flat, "optimized")
            else:
                attention_output = matmul_cuda.batched_matmul(attention_weights_flat, V_flat, self.implementation)
            
            attention_output = attention_output.view(batch_size, num_heads, seq_len, d_k)
            
        else:
            # Fallback to PyTorch
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_weights, V)
        
        return attention_output


class TransformerBlock(nn.Module):
    """
    Complete transformer block with optimized components
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, implementation: str = "optimized"):
        super().__init__()
        
        self.attention = OptimizedMultiHeadAttention(d_model, num_heads, implementation)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            OptimizedLinear(d_model, d_ff, implementation=implementation),
            nn.ReLU(),
            OptimizedLinear(d_ff, d_model, implementation=implementation)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class TransformerDemo:
    """
    Demonstration of transformer inference acceleration
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def benchmark_transformer_block(self, 
                                   d_model: int = 512, 
                                   num_heads: int = 8, 
                                   d_ff: int = 2048,
                                   seq_len: int = 128, 
                                   batch_size: int = 8,
                                   num_runs: int = 20):
        """
        Benchmark transformer block with different implementations
        """
        print(f"\n=== Transformer Block Benchmark ===")
        print(f"Config: d_model={d_model}, heads={num_heads}, d_ff={d_ff}")
        print(f"Input: batch_size={batch_size}, seq_len={seq_len}")
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
        print()
        
        # Generate input
        x = torch.randn(batch_size, seq_len, d_model, device=self.device, dtype=torch.float32)
        
        results = {}
        
        # Standard PyTorch implementation
        print("Benchmarking PyTorch baseline...")
        pytorch_block = self.create_pytorch_transformer_block(d_model, num_heads, d_ff)
        pytorch_time = self.benchmark_single_implementation(pytorch_block, x, num_runs, "PyTorch")
        results['pytorch'] = pytorch_time
        
        # Our implementations
        if CUSTOM_KERNELS_AVAILABLE:
            implementations = ['naive', 'tiled', 'optimized', 'smart']
            
            for impl in implementations:
                print(f"Benchmarking {impl} implementation...")
                custom_block = TransformerBlock(d_model, num_heads, d_ff, implementation=impl)
                custom_block = custom_block.to(self.device)
                
                try:
                    custom_time = self.benchmark_single_implementation(custom_block, x, num_runs, impl)
                    results[impl] = custom_time
                except Exception as e:
                    print(f"  {impl} failed: {e}")
        
        # Print summary
        self.print_benchmark_summary(results)
        
        return results
    
    def create_pytorch_transformer_block(self, d_model: int, num_heads: int, d_ff: int):
        """Create standard PyTorch transformer block for comparison"""
        class StandardTransformerBlock(nn.Module):
            def __init__(self, d_model, num_heads, d_ff):
                super().__init__()
                self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
                self.norm1 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Linear(d_ff, d_model)
                )
                self.norm2 = nn.LayerNorm(d_model)
            
            def forward(self, x):
                attn_output, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_output)
                ffn_output = self.ffn(x)
                x = self.norm2(x + ffn_output)
                return x
        
        block = StandardTransformerBlock(d_model, num_heads, d_ff)
        return block.to(self.device)
    
    def benchmark_single_implementation(self, model: nn.Module, x: torch.Tensor, num_runs: int, name: str) -> float:
        """Benchmark a single implementation"""
        model.eval()
        
        # Warm up
        with torch.no_grad():
            if self.device == 'cuda':
                torch.cuda.synchronize()
            for _ in range(5):
                _ = model(x)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output = model(x)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        
        print(f"  {name}: {avg_time:.2f} ± {std_time:.2f} ms")
        
        return avg_time
    
    def print_benchmark_summary(self, results: dict):
        """Print benchmark summary with speedups"""
        print(f"\n{'='*50}")
        print("TRANSFORMER BLOCK BENCHMARK SUMMARY")
        print(f"{'='*50}")
        
        pytorch_time = results.get('pytorch', 0)
        
        print(f"{'Implementation':<15} {'Time (ms)':<12} {'Speedup':<10}")
        print("-" * 40)
        
        for impl, time_ms in results.items():
            if impl == 'pytorch':
                speedup = 1.0
            else:
                speedup = pytorch_time / time_ms if pytorch_time > 0 else 0
            
            print(f"{impl:<15} {time_ms:<12.2f} {speedup:<10.2f}x")
    
    def profile_attention_components(self, d_model: int = 512, num_heads: int = 8, seq_len: int = 128, batch_size: int = 8):
        """Profile individual attention components"""
        print(f"\n=== Attention Component Profiling ===")
        
        if not CUSTOM_KERNELS_AVAILABLE:
            print("Custom kernels not available for profiling")
            return
        
        d_k = d_model // num_heads
        
        # Generate Q, K, V matrices (simulating output of linear projections)
        Q = torch.randn(batch_size * num_heads, seq_len, d_k, device=self.device, dtype=torch.float32)
        K = torch.randn(batch_size * num_heads, seq_len, d_k, device=self.device, dtype=torch.float32)
        V = torch.randn(batch_size * num_heads, seq_len, d_k, device=self.device, dtype=torch.float32)
        K_T = K.transpose(-2, -1)
        
        num_runs = 50
        
        # Profile Q @ K^T computation
        print(f"Profiling Q @ K^T ({seq_len}x{d_k} @ {d_k}x{seq_len} x {batch_size * num_heads} batches)")
        
        implementations = ['pytorch', 'naive', 'tiled', 'optimized']
        qk_results = {}
        
        for impl in implementations:
            if impl == 'pytorch':
                # PyTorch bmm
                torch.cuda.synchronize()
                times = []
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    scores = torch.bmm(Q, K_T)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append(end - start)
                
                avg_time = np.mean(times) * 1000
                qk_results[impl] = avg_time
                print(f"  {impl}: {avg_time:.2f} ms")
            
            else:
                try:
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(num_runs):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        scores = matmul_cuda.batched_matmul(Q, K_T, impl)
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        times.append(end - start)
                    
                    avg_time = np.mean(times) * 1000
                    qk_results[impl] = avg_time
                    print(f"  {impl}: {avg_time:.2f} ms")
                    
                except Exception as e:
                    print(f"  {impl}: Failed - {e}")
        
        # Show speedups
        pytorch_time = qk_results.get('pytorch', 1)
        print("\nSpeedups vs PyTorch:")
        for impl, time_ms in qk_results.items():
            if impl != 'pytorch':
                speedup = pytorch_time / time_ms
                print(f"  {impl}: {speedup:.2f}x")


def main():
    """Run the transformer inference demo"""
    print("🚀 TRANSFORMER INFERENCE ACCELERATION DEMO")
    print("=" * 60)
    
    demo = TransformerDemo()
    
    # Test different configurations
    configs = [
        {"d_model": 256, "num_heads": 4, "d_ff": 1024, "seq_len": 64, "batch_size": 8},
        {"d_model": 512, "num_heads": 8, "d_ff": 2048, "seq_len": 128, "batch_size": 8},
        {"d_model": 768, "num_heads": 12, "d_ff": 3072, "seq_len": 256, "batch_size": 4},
    ]
    
    all_results = {}
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"CONFIGURATION {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        results = demo.benchmark_transformer_block(**config)
        all_results[f"config_{i+1}"] = results
    
    # Profile attention components for detailed analysis
    demo.profile_attention_components()
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE!")
    print(f"{'='*60}")
    
    # Summary across all configurations
    print("\nOverall Performance Summary:")
    for config_name, results in all_results.items():
        pytorch_time = results.get('pytorch', 0)
        best_custom = min([time for impl, time in results.items() if impl != 'pytorch'], default=pytorch_time)
        best_speedup = pytorch_time / best_custom if best_custom > 0 else 1.0
        print(f"  {config_name}: Best speedup = {best_speedup:.2f}x")


if __name__ == "__main__":
    main()
