#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
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

__global__ void naive_matmul_kernel( const float* A, const float* B, float* C, int M, int N, int K ) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row from A and column from B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

void naive_matmul( const float* h_A, const float* h_B, float* h_C, int M, int N, int K) {

    float *d_A, *d_B, *d_C;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    const int BLOCK_SIZE = 16; 
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    naive_matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

float benchmark_naive_matmul(int M, int N, int K, int num_runs = 10) {

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    
    naive_matmul(h_A, h_B, h_C, M, N, K);
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int run = 0; run < num_runs; run++) {
        naive_matmul(h_A, h_B, h_C, M, N, K);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    
    float avg_time_ms = duration.count() / (1000.0f * num_runs);
    
    // Calculate GFLOPS
    long long flops = 2LL * M * N * K;
    float gflops = flops / (avg_time_ms * 1e6);
    
    cout << "Naive CUDA - Size: " << M << "x" << N << "x" << K 
              << ", Time: " << avg_time_ms << " ms"
              << ", Performance: " << gflops << " GFLOPS" << endl;
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return avg_time_ms;
}

int main() {
    cout << "Naive CUDA Matrix Multiplication Benchmark \n\n" << endl;
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << endl;
    cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << endl;
    cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << endl;
    
    int sizes[] = {64, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        benchmark_naive_matmul(size, size, size);
    }
    
    return 0;
}
