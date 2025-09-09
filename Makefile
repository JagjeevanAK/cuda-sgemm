# Makefile for Fast Matrix Multiplication Project
# Demonstrates performance comparison between CUDA kernels, PyTorch, and NumPy

# Variables
PYTHON := python3
PIP := pip3
NVCC := nvcc
CUDA_ARCH := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80

# Directories
SRC_DIR := cuda_kernels
BENCH_DIR := benchmarks
BUILD_DIR := build

# CUDA compilation flags
NVCC_FLAGS := -O3 -std=c++14 --use_fast_math $(CUDA_ARCH)
CUDA_INC := -I/usr/local/cuda/include
CUDA_LIB := -L/usr/local/cuda/lib64 -lcudart -lcublas

.PHONY: all setup build test benchmark clean help

# Default target
all: setup build

# Help
help:
	@echo "Fast Matrix Multiplication - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make setup          - Install Python dependencies"
	@echo "  make build          - Build all CUDA kernels"
	@echo ""
	@echo "Testing and Benchmarking:"
	@echo "  make test           - Test CUDA kernels and baselines"
	@echo "  make benchmark      - Run comprehensive performance comparison"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean build artifacts"

# Setup Python environment
setup:
	@echo "Setting up Python environment..."
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

# Build CUDA kernels
build:
	@echo "Building CUDA kernels..."
	@mkdir -p $(BUILD_DIR)
	
	@echo "Building naive matrix multiplication..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(SRC_DIR)/naive_matmul.cu -o $(BUILD_DIR)/naive_matmul $(CUDA_LIB)
	
	@echo "Building tiled matrix multiplication..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(SRC_DIR)/tiled_matmul.cu -o $(BUILD_DIR)/tiled_matmul $(CUDA_LIB)
	
	@echo "Building optimized matrix multiplication..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(SRC_DIR)/optimized_matmul.cu -o $(BUILD_DIR)/optimized_matmul $(CUDA_LIB)
	
	@echo "Building CUBLAS matrix multiplication..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(SRC_DIR)/cublas_matmul.cu -o $(BUILD_DIR)/cublas_matmul $(CUDA_LIB)
	
	@echo "✓ CUDA kernels built successfully"

# Test CUDA kernels and baselines
test: build
	@echo "Testing CUDA kernels..."
	@echo "Running naive implementation..."
	./$(BUILD_DIR)/naive_matmul
	@echo "Running tiled implementation..."
	./$(BUILD_DIR)/tiled_matmul
	@echo "Running optimized implementation..."
	./$(BUILD_DIR)/optimized_matmul
	@echo "Running CUBLAS implementation..."
	./$(BUILD_DIR)/cublas_matmul
	@echo "Testing baseline implementations..."
	cd baselines && $(PYTHON) numpy_matmul.py
	cd baselines && $(PYTHON) pytorch_matmul.py
	@echo "✓ All tests completed"

# Run comprehensive benchmarks
benchmark: build
	@echo "Running comprehensive benchmarks..."
	cd $(BENCH_DIR) && $(PYTHON) benchmark_all.py
	@echo "✓ Benchmarks completed - check benchmark_plots/ and benchmark_report.csv"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -f benchmark_report.csv
	rm -rf benchmark_plots/
	@echo "✓ Clean completed"

# Uninstall
uninstall:
	@echo "Uninstalling..."
	$(PIP) uninstall matmul_cuda -y
	@echo "✓ Uninstallation completed"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@echo "NVCC version:"
	$(NVCC) --version || echo "NVCC not found"
	@echo "CUDA devices:"
	nvidia-smi || echo "nvidia-smi not found"
	@echo "PyTorch CUDA:"
	$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" || echo "PyTorch not available"

# Docker build (if Dockerfile exists)
docker:
	@if [ -f Dockerfile ]; then \
		echo "Building Docker container..."; \
		docker build -t fast-matmul .; \
		echo "✓ Docker container built"; \
	else \
		echo "Dockerfile not found - skipping Docker build"; \
	fi

# Create a release package
package: clean build test
	@echo "Creating release package..."
	@mkdir -p release
	tar -czf release/fast-matmul-$(shell date +%Y%m%d).tar.gz \
		--exclude='.git' \
		--exclude='release' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		.
	@echo "✓ Release package created in release/"

# Performance comparison report
report: benchmark
	@echo "Generating performance report..."
	@echo "Report generated - check benchmark_report.csv and benchmark_plots/"

# Show project statistics
stats:
	@echo "Project Statistics:"
	@echo "Lines of code:"
	@find . -name "*.py" -o -name "*.cu" -o -name "*.cpp" | xargs wc -l | tail -1
	@echo "CUDA files:"
	@find . -name "*.cu" | wc -l
	@echo "Python files:"
	@find . -name "*.py" | wc -l
	@echo "C++ files:"
	@find . -name "*.cpp" | wc -l
