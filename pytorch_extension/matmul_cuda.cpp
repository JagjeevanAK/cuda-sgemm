/*
 * PyTorch C++ extension wrapper for custom CUDA matrix multiplication kernels
 * This file provides the interface between PyTorch tensors and our CUDA kernels
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

// Forward declarations of CUDA functions
void naive_matmul_cuda(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

void tiled_matmul_cuda(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

void optimized_matmul_cuda(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

// Check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * Naive CUDA matrix multiplication wrapper
 */
torch::Tensor naive_matmul(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    // Get dimensions
    auto A_sizes = A.sizes();
    auto B_sizes = B.sizes();
    
    TORCH_CHECK(A_sizes.size() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B_sizes.size() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A_sizes[1] == B_sizes[0], "Matrix dimensions must be compatible");
    
    int M = A_sizes[0];
    int K = A_sizes[1];
    int N = B_sizes[1];
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(A.device())
        .requires_grad(A.requires_grad() || B.requires_grad());
    
    torch::Tensor C = torch::zeros({M, N}, options);
    
    // Call CUDA kernel
    naive_matmul_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

/**
 * Tiled CUDA matrix multiplication wrapper
 */
torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    auto A_sizes = A.sizes();
    auto B_sizes = B.sizes();
    
    TORCH_CHECK(A_sizes.size() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B_sizes.size() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A_sizes[1] == B_sizes[0], "Matrix dimensions must be compatible");
    
    int M = A_sizes[0];
    int K = A_sizes[1];
    int N = B_sizes[1];
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(A.device())
        .requires_grad(A.requires_grad() || B.requires_grad());
    
    torch::Tensor C = torch::zeros({M, N}, options);
    
    tiled_matmul_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

/**
 * Optimized CUDA matrix multiplication wrapper
 */
torch::Tensor optimized_matmul(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    auto A_sizes = A.sizes();
    auto B_sizes = B.sizes();
    
    TORCH_CHECK(A_sizes.size() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B_sizes.size() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A_sizes[1] == B_sizes[0], "Matrix dimensions must be compatible");
    
    int M = A_sizes[0];
    int K = A_sizes[1];
    int N = B_sizes[1];
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(A.device())
        .requires_grad(A.requires_grad() || B.requires_grad());
    
    torch::Tensor C = torch::zeros({M, N}, options);
    
    optimized_matmul_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

/**
 * Batched matrix multiplication for transformer workloads
 */
torch::Tensor batched_matmul(torch::Tensor A, torch::Tensor B, string implementation = "optimized") {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    auto A_sizes = A.sizes();
    auto B_sizes = B.sizes();
    
    // Support both 2D and 3D tensors (batched)
    if (A_sizes.size() == 3 && B_sizes.size() == 3) {
        // Batched operation
        TORCH_CHECK(A_sizes[0] == B_sizes[0], "Batch sizes must match");
        TORCH_CHECK(A_sizes[2] == B_sizes[1], "Matrix dimensions must be compatible");
        
        int batch_size = A_sizes[0];
        int M = A_sizes[1];
        int K = A_sizes[2];
        int N = B_sizes[2];
        
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(A.device())
            .requires_grad(A.requires_grad() || B.requires_grad());
        
        torch::Tensor C = torch::zeros({batch_size, M, N}, options);
        
        // Process each batch
        for (int b = 0; b < batch_size; b++) {
            torch::Tensor A_batch = A[b];
            torch::Tensor B_batch = B[b];
            torch::Tensor C_batch = C[b];
            
            if (implementation == "naive") {
                naive_matmul_cuda(
                    A_batch.data_ptr<float>(),
                    B_batch.data_ptr<float>(),
                    C_batch.data_ptr<float>(),
                    M, N, K
                );
            } else if (implementation == "tiled") {
                tiled_matmul_cuda(
                    A_batch.data_ptr<float>(),
                    B_batch.data_ptr<float>(),
                    C_batch.data_ptr<float>(),
                    M, N, K
                );
            } else {
                optimized_matmul_cuda(
                    A_batch.data_ptr<float>(),
                    B_batch.data_ptr<float>(),
                    C_batch.data_ptr<float>(),
                    M, N, K
                );
            }
        }
        
        return C;
    } else {
        // Single matrix operation
        if (implementation == "naive") {
            return naive_matmul(A, B);
        } else if (implementation == "tiled") {
            return tiled_matmul(A, B);
        } else {
            return optimized_matmul(A, B);
        }
    }
}

/**
 * Convenience function that automatically selects the best implementation
 * based on matrix size and GPU characteristics
 */
torch::Tensor smart_matmul(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    auto A_sizes = A.sizes();
    auto B_sizes = B.sizes();
    
    // Simple heuristic: use optimized for large matrices, tiled for medium, naive for small
    int total_elements = A_sizes[0] * A_sizes[1] * B_sizes[1];
    
    if (total_elements > 1024 * 1024) {
        return optimized_matmul(A, B);
    } else if (total_elements > 64 * 64) {
        return tiled_matmul(A, B);
    } else {
        return naive_matmul(A, B);
    }
}

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom CUDA matrix multiplication kernels for PyTorch";
    
    m.def("naive_matmul", &naive_matmul, "Naive CUDA matrix multiplication");
    m.def("tiled_matmul", &tiled_matmul, "Tiled CUDA matrix multiplication with shared memory");
    m.def("optimized_matmul", &optimized_matmul, "Highly optimized CUDA matrix multiplication");
    m.def("batched_matmul", &batched_matmul, "Batched matrix multiplication",
          py::arg("A"), py::arg("B"), py::arg("implementation") = "optimized");
    m.def("smart_matmul", &smart_matmul, "Automatically select best implementation");
}
