# Fast Matrix Multiplication with Custom CUDA Kernels

**Accelerating ML Inference with Custom CUDA Implementations**

This project demonstrates how custom CUDA kernels can outperform standard PyTorch/NumPy operations for matrix multiplication - the backbone of transformer inference (attention layers, feed-forward layers).

## Project Overview

Matrix multiplications (GEMM) are critical for ML inference performance. This project shows:
- How inference workloads can be **accelerated on GPU**
- How **custom CUDA kernels** can outperform naive PyTorch/NumPy ops
- How to integrate CUDA with ML inference via PyTorch extensions

## Project Structure

```
Fast-Matmul/
├── baselines/                 # Python and PyTorch implementations
│   ├── numpy_matmul.py       # Pure NumPy baseline
│   └── pytorch_matmul.py     # PyTorch baseline
├── cuda_kernels/             # CUDA implementations
│   ├── naive_matmul.cu       # Basic CUDA kernel
│   ├── tiled_matmul.cu       # Shared memory tiled version
│   └── optimized_matmul.cu   # Fully optimized version
├── pytorch_extension/        # PyTorch integration
│   ├── setup.py             # Extension build script
│   ├── matmul_cuda.cpp      # C++ wrapper
│   └── test_extension.py    # Test the extension
├── benchmarks/               # Performance comparisons
│   ├── benchmark_all.py     # Compare all implementations
│   └── inference_demo.py    # Transformer block demo
├── requirements.txt          # Python dependencies
└── Makefile                 # Build automation
```

## Implementation Steps

### 1. Baseline Implementations
- [x] Pure Python/NumPy matrix multiplication
- [x] PyTorch `torch.matmul` baseline
- [x] Performance benchmarking

### 2. CUDA Kernel Development
- [ ] Naive CUDA kernel (1 thread per element)
- [ ] Shared memory tiled implementation
- [ ] Optimized version with loop unrolling & register blocking

### 3. PyTorch Integration
- [ ] Wrap CUDA kernel as PyTorch extension
- [ ] Replace `torch.matmul` in transformer components
- [ ] End-to-end inference demonstration

## Quick Start

### Prerequisites
- CUDA-capable GPU (Compute Capability 6.0+)
- CUDA Toolkit (11.0+)
- Python 3.8+
- PyTorch with CUDA support

### Installation
```bash
# Clone and setup environment
git clone <your-repo>
cd Fast-Matmul

# Install dependencies
pip install -r requirements.txt

# OR with make-cmd
make setup

# Build CUDA kernels
make build

# Run benchmarks
python benchmarks/benchmark_all.py
```

## Performance Results

| Implementation | Matrix Size | Time (ms) | Speedup |
|---------------|-------------|-----------|---------|
| NumPy         | 1024x1024   | TBD       | 1.0x    |
| PyTorch       | 1024x1024   | TBD       | TBD     |
| Naive CUDA    | 1024x1024   | TBD       | TBD     |
| Tiled CUDA    | 1024x1024   | TBD       | TBD     |
| Optimized     | 1024x1024   | TBD       | TBD     |

## ML Inference Integration

This project demonstrates CUDA acceleration in real ML inference scenarios:
- **Attention mechanisms**: Accelerated Q@K^T and Attention@V operations
- **Feed-forward layers**: Fast linear transformations
- **Transformer blocks**: End-to-end speedup demonstration

## Technical Highlights

- **Memory Coalescing**: Optimized global memory access patterns
- **Shared Memory Tiling**: Reduced global memory bandwidth requirements
- **Register Blocking**: Maximized arithmetic intensity
- **PyTorch Integration**: Seamless drop-in replacement for `torch.matmul`

## Scaling Analysis

Performance analysis across different matrix sizes and hardware configurations, demonstrating when custom kernels outperform highly optimized libraries like cuBLAS.

---

*This project showcases practical CUDA optimization skills relevant to ML inference acceleration, hardware-aware algorithm design, and performance engineering.*
