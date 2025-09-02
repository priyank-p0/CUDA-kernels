#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// naive softmax kernel
__global__ void naive_softmax_kernel(float* input, float* output, int n){

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float shared_data[];
    float* s_data = shared_data;

    float local_max = -INFINITY;
    if (thread_id < n){
        s_data[threadIdx.x] = input[thread_id];
        local_max = input[thread_id];
    }
    else{
        s_data[threadIdx.x] = -INFINITY;
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>=1){
        if (threadIdx.x < s && threadIdx.x + s < blockDim.x){
            if (s_data[threadIdx.x + s] > s_data[threadIdx.x]){
                s_data[threadIdx.x] = s_data[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    float max_val = s_data[0];
    __syncthreads();

    if (thread_id < n){
        s_data[threadIdx.x] = expf(input[thread_id] - max_val);
    }
    else{
        s_data[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>=1){
        if (threadIdx.x < s){
            if (threadIdx.x + s < blockDim.x){
                s_data[threadIdx.x] += s_data[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float exp_sum = s_data[0];
    __syncthreads();

    if (thread_id < n){
        output[thread_id] = expf(input[thread_id] - max_val) / exp_sum;
    }
}


// Utility function to print array
void print_array(const char* name, float* arr, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n; i++) {
        printf("%.6f", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}



int main() {
    // Test parameters
    const int n = 8;  // Size of input vector (must be <= block size for this naive implementation)
    const int block_size = 256;  // CUDA block size
    
    printf("Naive Softmax CUDA Implementation\n");
    printf("Input size: %d\n\n", n);
    
    // Allocate host memory
    float* h_input = (float*)malloc(n * sizeof(float));
    float* h_output_gpu = (float*)malloc(n * sizeof(float));
    
    // Initialize input with sample data
    printf("Initializing input data...\n");
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f - 5.0f;  // Random values between -5 and 5
    }
    
    print_array("Input", h_input, n);
    // Allocate device memory
    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    // Launch kernel
    printf("\nLaunching CUDA kernel...\n");
    dim3 grid(1);  // Single block for this naive implementation
    dim3 block(block_size);
    size_t shared_mem_size = block_size * sizeof(float);
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // Record start time
    CUDA_CHECK(cudaEventRecord(start));
    // Launch kernel
    naive_softmax_kernel<<<grid, block, shared_mem_size>>>(d_input, d_output, n);
    // Record end time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    // Calculate elapsed time
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    // Print results
    print_array("GPU Output", h_output_gpu, n);    
    // Print timing
    printf("GPU execution time: %.3f ms\n", gpu_time);
    
    // Verify that output sums to 1 (property of softmax)
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += h_output_gpu[i];
    }
    printf("Sum of GPU output: %.6f (should be ~1.0)\n", sum);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output_gpu);
    
    printf("\nProgram completed successfully!\n");
    return 0;
}

/*
  
 * To run:
 *   ./naive_softmax
 */
