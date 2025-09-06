/*
 * Tiled CUDA matrix multiplication kernel using shared memory
 * This implementation uses shared memory to reduce global memory accesses
 * and improve memory bandwidth utilization
 */

#include <cuda_runtime.h>
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

// Tile size for shared memory - can be tuned for different GPUs
#define TILE_SIZE 16

/**
 * Tiled matrix multiplication kernel using shared memory
 * 
 * Key optimizations:
 * 1. Use shared memory to cache frequently accessed data
 * 2. Coalesce global memory accesses
 * 3. Reduce total number of global memory transactions
 * 
 * Each thread block computes a TILE_SIZE x TILE_SIZE submatrix of C
 * Threads cooperatively load tiles from A and B into shared memory
 */
__global__ void tiled_matmul_kernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, int N, int K
) {
    // Shared memory for tiles of A and B
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global indices for output matrix C
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles in the K dimension
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Collaboratively load tile from A into shared memory
        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            tile_A[ty][tx] = A[row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }
        
        // Collaboratively load tile from B into shared memory
        int b_row = tile * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            tile_B[ty][tx] = B[b_row * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Host function to launch tiled matrix multiplication
 */
void tiled_matmul(
    const float* h_A, 
    const float* h_B, 
    float* h_C, 
    int M, int N, int K
) {
    // Device memory pointers
    float *d_A, *d_B, *d_C;
    
    // Calculate memory sizes
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
    
    // Launch configuration
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    // Launch kernel
    tiled_matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

/**
 * Optimized version with better memory coalescing
 * Transposes B to improve memory access patterns
 */
__global__ void tiled_matmul_kernel_optimized(
    const float* A, 
    const float* B_T,  // B transposed for better memory access
    float* C, 
    int M, int N, int K
) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile from A
        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            tile_A[ty][tx] = A[row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;
        }
        
        // Load tile from B_T (transposed B for better coalescing)
        int b_col = tile * TILE_SIZE + ty;
        if (col < N && b_col < K) {
            tile_B[ty][tx] = B_T[col * K + b_col];  // Note: accessing B_T
        } else {
            tile_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Benchmark function for tiled implementation
 */
float benchmark_tiled_matmul(int M, int N, int K, int num_runs = 10) {
    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    
    // Initialize matrices with random values
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    
    // Warm up
    tiled_matmul(h_A, h_B, h_C, M, N, K);
    
    // Benchmark
    auto start = chrono::high_resolution_clock::now();
    
    for (int run = 0; run < num_runs; run++) {
        tiled_matmul(h_A, h_B, h_C, M, N, K);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    
    float avg_time_ms = duration.count() / (1000.0f * num_runs);
    
    // Calculate GFLOPS
    long long flops = 2LL * M * N * K;
    float gflops = flops / (avg_time_ms * 1e6);
    
    cout << "Tiled CUDA - Size: " << M << "x" << N << "x" << K 
              << ", Time: " << avg_time_ms << " ms"
              << ", Performance: " << gflops << " GFLOPS" << endl;
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return avg_time_ms;
}

/**
 * Compare tiled vs naive implementation
 */
void compare_implementations(int M, int N, int K) {
    cout << "\n=== Comparing Implementations for " << M << "x" << N << "x" << K << " ===" << endl;
    
    // We'll implement this comparison in the benchmark script
    // For now, just run tiled benchmark
    benchmark_tiled_matmul(M, N, K);
}

int main() {
    cout << "=== Tiled CUDA Matrix Multiplication Benchmark ===" << endl;
    
    // Get device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << endl;
    cout << "Tile size: " << TILE_SIZE << "x" << TILE_SIZE << endl;
    cout << "Shared memory per tile: " << (2 * TILE_SIZE * TILE_SIZE * sizeof(float)) << " bytes" << endl;
    cout << "Available shared memory: " << prop.sharedMemPerBlock << " bytes" << endl;
    cout << endl;
    
    // Test different matrix sizes
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        benchmark_tiled_matmul(size, size, size);
    }
    
    return 0;
}
