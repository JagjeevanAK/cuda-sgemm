# Makefile for Fast Matrix Multiplication Project
# Provides convenient commands for building, testing, and benchmarking

# Variables
PYTHON := python3
PIP := pip3
NVCC := nvcc
CUDA_ARCH := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80

# Directories
SRC_DIR := cuda_kernels
EXT_DIR := pytorch_extension
BENCH_DIR := benchmarks
BUILD_DIR := build

# CUDA compilation flags
NVCC_FLAGS := -O3 -std=c++14 --use_fast_math $(CUDA_ARCH)
CUDA_INC := -I/usr/local/cuda/include
CUDA_LIB := -L/usr/local/cuda/lib64 -lcudart -lcublas

.PHONY: all setup build-cuda build-extension test benchmark clean help

# Default target
all: setup build

# Help
help:
	@echo "Fast Matrix Multiplication - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make setup          - Install Python dependencies"
	@echo "  make build          - Build all CUDA kernels and extensions"
	@echo "  make build-cuda     - Build standalone CUDA executables"
	@echo "  make build-extension - Build PyTorch extension"
	@echo ""
	@echo "Testing and Benchmarking:"
	@echo "  make test           - Run all tests"
	@echo "  make test-extension - Test PyTorch extension"
	@echo "  make test-cuda      - Test standalone CUDA kernels"
	@echo "  make benchmark      - Run comprehensive benchmarks"
	@echo "  make demo           - Run transformer inference demo"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make format         - Format Python code"
	@echo "  make profile        - Run performance profiling"
	@echo "  make docker         - Build Docker container (if available)"

# Setup Python environment
setup:
	@echo "Setting up Python environment..."
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

# Build all components
build: build-cuda build-extension

# Build standalone CUDA executables
build-cuda:
	@echo "Building CUDA kernels..."
	@mkdir -p $(BUILD_DIR)
	
	@echo "Building naive matrix multiplication..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(SRC_DIR)/naive_matmul.cu -o $(BUILD_DIR)/naive_matmul $(CUDA_LIB)
	
	@echo "Building tiled matrix multiplication..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(SRC_DIR)/tiled_matmul.cu -o $(BUILD_DIR)/tiled_matmul $(CUDA_LIB)
	
	@echo "Building optimized matrix multiplication..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(SRC_DIR)/optimized_matmul.cu -o $(BUILD_DIR)/optimized_matmul $(CUDA_LIB)
	
	@echo "✓ CUDA kernels built successfully"

# Build PyTorch extension
build-extension:
	@echo "Building PyTorch extension..."
	cd $(EXT_DIR) && $(PYTHON) setup.py build_ext --inplace
	@echo "✓ PyTorch extension built successfully"

# Test standalone CUDA kernels
test-cuda: build-cuda
	@echo "Testing CUDA kernels..."
	@echo "Running naive implementation..."
	./$(BUILD_DIR)/naive_matmul
	@echo "Running tiled implementation..."
	./$(BUILD_DIR)/tiled_matmul
	@echo "Running optimized implementation..."
	./$(BUILD_DIR)/optimized_matmul
	@echo "✓ CUDA tests completed"

# Test PyTorch extension
test-extension: build-extension
	@echo "Testing PyTorch extension..."
	cd $(EXT_DIR) && $(PYTHON) test_extension.py
	@echo "✓ Extension tests completed"

# Run all tests
test: test-cuda test-extension
	@echo "Running baseline tests..."
	cd baselines && $(PYTHON) numpy_matmul.py
	cd baselines && $(PYTHON) pytorch_matmul.py
	@echo "✓ All tests completed"

# Run comprehensive benchmarks
benchmark: build
	@echo "Running comprehensive benchmarks..."
	cd $(BENCH_DIR) && $(PYTHON) benchmark_all.py
	@echo "✓ Benchmarks completed - check benchmark_plots/ and benchmark_report.csv"

# Run transformer inference demo
demo: build-extension
	@echo "Running transformer inference demo..."
	cd $(BENCH_DIR) && $(PYTHON) inference_demo.py
	@echo "✓ Demo completed"

# Performance profiling with nvidia profiler
profile: build-extension
	@echo "Running performance profiling..."
	@echo "Note: This requires nsight-systems or nsight-compute to be installed"
	cd $(BENCH_DIR) && nsys profile --output=profile_results $(PYTHON) inference_demo.py
	@echo "✓ Profiling completed - check profile_results.nsys-rep"

# Format Python code
format:
	@echo "Formatting Python code..."
	black baselines/ $(EXT_DIR)/ $(BENCH_DIR)/ --line-length 100
	@echo "✓ Code formatted"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(EXT_DIR)/build
	rm -rf $(EXT_DIR)/*.so
	rm -rf $(EXT_DIR)/*.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -f benchmark_report.csv
	rm -rf benchmark_plots/
	@echo "✓ Clean completed"

# Quick development cycle
dev: clean build test
	@echo "✓ Development cycle completed"

# Install for system-wide use
install: build
	@echo "Installing PyTorch extension..."
	cd $(EXT_DIR) && $(PYTHON) setup.py install
	@echo "✓ Installation completed"

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
