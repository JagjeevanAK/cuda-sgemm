#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <vector>
using namespace std;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << status << endl; \
            exit(1); \
        } \
    } while(0)

class CublasMatmul {
private:
    cublasHandle_t handle;
    bool handle_created;
    
public:
    CublasMatmul() : handle_created(false) {
        CHECK_CUBLAS(cublasCreate(&handle));
        handle_created = true;
        
        // Set math mode for better performance on newer GPUs
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    }
    
    ~CublasMatmul() {
        if (handle_created) {
            cublasDestroy(handle);
        }
    }
    
    void matmul(const float* d_A, const float* d_B, float* d_C, 
                int M, int N, int K) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // CUBLAS uses column-major order, so we need to transpose our operation
        // C = A * B becomes C^T = B^T * A^T in column-major
        // Since we're working with row-major data, we compute: C = B^T * A^T
        CHECK_CUBLAS(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,    // No transpose for B, no transpose for A
            N, M, K,                      // Dimensions: N x M result, with K inner dimension
            &alpha,                       // Scaling factor for A*B
            d_B, N,                       // Matrix B with leading dimension N
            d_A, K,                       // Matrix A with leading dimension K
            &beta,                        // Scaling factor for C (0 means overwrite)
            d_C, N                        // Matrix C with leading dimension N
        ));
    }
    
    void matmul_batched(const float* const* d_A_array, 
                       const float* const* d_B_array, 
                       float* const* d_C_array,
                       int M, int N, int K, int batch_count) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        CHECK_CUBLAS(cublasSgemmBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B_array, N,
            d_A_array, K,
            &beta,
            d_C_array, N,
            batch_count
        ));
    }
    
    void matmul_strided_batched(const float* d_A, const float* d_B, float* d_C,
                               int M, int N, int K, int batch_count,
                               long long stride_A, long long stride_B, long long stride_C) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N, stride_B,
            d_A, K, stride_A,
            &beta,
            d_C, N, stride_C,
            batch_count
        ));
    }
};

void cublas_matmul(const float* h_A, const float* h_B, float* h_C, 
                   int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Create CUBLAS instance and perform multiplication
    CublasMatmul cublas;
    cublas.matmul(d_A, d_B, d_C, M, N, K);
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

// High-performance version that keeps data on GPU
void cublas_matmul_device_only(const float* d_A, const float* d_B, float* d_C,
                              int M, int N, int K) {
    static CublasMatmul cublas; // Static to avoid recreating handle
    cublas.matmul(d_A, d_B, d_C, M, N, K);
}

float benchmark_cublas_matmul(int M, int N, int K, int num_runs = 10) {
    // Allocate host memory
    vector<float> h_A(M * K);
    vector<float> h_B(K * N);
    vector<float> h_C(M * N);
    
    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    
    // Create CUBLAS instance
    CublasMatmul cublas;
    
    // Warm up
    cublas.matmul(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = chrono::high_resolution_clock::now();
    
    for (int run = 0; run < num_runs; run++) {
        cublas.matmul(d_A, d_B, d_C, M, N, K);
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    float avg_time_ms = duration.count() / (1000.0f * num_runs);
    
    // Calculate GFLOPS
    long long flops = 2LL * M * N * K;
    float gflops = flops / (avg_time_ms * 1e6);
    
    cout << "CUBLAS - Size: " << M << "x" << N << "x" << K 
         << ", Time: " << avg_time_ms << " ms"
         << ", Performance: " << gflops << " GFLOPS" << endl;
    
    // Copy result back to verify correctness if needed
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    return avg_time_ms;
}

void benchmark_batched_operations(int M, int N, int K, int batch_size) {
    cout << "\n=== Batched Operations Benchmark ===" << endl;
    
    // Allocate memory for batch of matrices
    size_t matrix_size_A = M * K * sizeof(float);
    size_t matrix_size_B = K * N * sizeof(float);
    size_t matrix_size_C = M * N * sizeof(float);
    
    float *d_A_batch, *d_B_batch, *d_C_batch;
    CHECK_CUDA(cudaMalloc(&d_A_batch, batch_size * matrix_size_A));
    CHECK_CUDA(cudaMalloc(&d_B_batch, batch_size * matrix_size_B));
    CHECK_CUDA(cudaMalloc(&d_C_batch, batch_size * matrix_size_C));
    
    // Initialize with random data
    vector<float> h_data_A(batch_size * M * K);
    vector<float> h_data_B(batch_size * K * N);
    
    srand(42);
    for (size_t i = 0; i < h_data_A.size(); i++) {
        h_data_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < h_data_B.size(); i++) {
        h_data_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    CHECK_CUDA(cudaMemcpy(d_A_batch, h_data_A.data(), 
                         batch_size * matrix_size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_batch, h_data_B.data(), 
                         batch_size * matrix_size_B, cudaMemcpyHostToDevice));
    
    CublasMatmul cublas;
    
    // Benchmark strided batched operation
    auto start = chrono::high_resolution_clock::now();
    
    cublas.matmul_strided_batched(d_A_batch, d_B_batch, d_C_batch,
                                 M, N, K, batch_size,
                                 M * K, K * N, M * N);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    float time_ms = duration.count() / 1000.0f;
    
    long long total_flops = 2LL * M * N * K * batch_size;
    float gflops = total_flops / (time_ms * 1e6);
    
    cout << "Strided Batched GEMM - Batch size: " << batch_size 
         << ", Matrix size: " << M << "x" << N << "x" << K
         << ", Time: " << time_ms << " ms"
         << ", Performance: " << gflops << " GFLOPS" << endl;
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A_batch));
    CHECK_CUDA(cudaFree(d_B_batch));
    CHECK_CUDA(cudaFree(d_C_batch));
}

void print_gpu_info() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    cout << "GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << endl;
    cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << endl;
    cout << "Peak Memory Bandwidth: " 
         << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 
         << " GB/s" << endl;
    
    // Check for Tensor Core support
    if (prop.major >= 7) {
        cout << "Tensor Cores: Supported" << endl;
    } else {
        cout << "Tensor Cores: Not supported" << endl;
    }
    cout << endl;
}

int main() {
    cout << "=== CUBLAS Matrix Multiplication Benchmark ===" << endl;
    
    print_gpu_info();
    
    // Test different sizes
    vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    
    cout << "=== Single Matrix Multiplication ===" << endl;
    for (int size : sizes) {
        benchmark_cublas_matmul(size, size, size);
    }
    
    // Test rectangular matrices
    cout << "\n=== Rectangular Matrices ===" << endl;
    benchmark_cublas_matmul(1024, 512, 2048);
    benchmark_cublas_matmul(2048, 1024, 512);
    benchmark_cublas_matmul(512, 2048, 1024);
    
    // Test batched operations
    benchmark_batched_operations(512, 512, 512, 8);
    benchmark_batched_operations(1024, 1024, 1024, 4);
    
    return 0;
}
