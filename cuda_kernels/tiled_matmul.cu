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

#define TILE_SIZE 16

__global__ void tiled_matmul_kernel( const float* A, const float* B, float* C, int M, int N, int K) {

    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            tile_A[ty][tx] = A[row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0f; 
        }
        
        int b_row = tile * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            tile_B[ty][tx] = B[b_row * N + col];
        } else {
            tile_B[ty][tx] = 0.0f; 
        }
        
        __syncthreads();
        
        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


void tiled_matmul( const float* h_A, const float* h_B, float* h_C, int M, int N, int K ) {

    float *d_A, *d_B, *d_C;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    tiled_matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}


__global__ void tiled_matmul_kernel_optimized(
    const float* A, 
    const float* B_T,  
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

        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            tile_A[ty][tx] = A[row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;
        }
        
        // Load tile from B_T (transposed B for better coalescing)
        int b_col = tile * TILE_SIZE + ty;
        if (col < N && b_col < K) {
            tile_B[ty][tx] = B_T[col * K + b_col];
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

float benchmark_tiled_matmul(int M, int N, int K, int num_runs = 10) {
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    
    srand(42); 
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < K * N; i++){
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    tiled_matmul(h_A, h_B, h_C, M, N, K);
    
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
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return avg_time_ms;
}

void compare_implementations(int M, int N, int K) {
    cout << "\n=== Comparing Implementations for " << M << "x" << N << "x" << K << " ===" << endl;
    
    // We'll implement this comparison in the benchmark script
    // For now, just run tiled benchmark
    benchmark_tiled_matmul(M, N, K);
}

int main() {
    cout << "=== Tiled CUDA Matrix Multiplication Benchmark ===" << endl;
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << endl;
    cout << "Tile size: " << TILE_SIZE << "x" << TILE_SIZE << endl;
    cout << "Shared memory per tile: " << (2 * TILE_SIZE * TILE_SIZE * sizeof(float)) << " bytes" << endl;
    cout << "Available shared memory: " << prop.sharedMemPerBlock << " bytes" << endl;
    cout << endl;
    
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        benchmark_tiled_matmul(size, size, size);
    }
    
    return 0;
}
