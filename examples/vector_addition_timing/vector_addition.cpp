#include "hip/hip_runtime.h"
#include <stdio.h>

// Macro for checking errors in GPU API calls
#define gpuErrorCheck(call)                                                                 \
do{                                                                                         \
    hipError_t gpuErr = call;                                                               \
    if(hipSuccess != gpuErr){                                                               \
        printf("GPU Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                            \
    }                                                                                       \
}while(0)

// Size of array
#define N 1048576

// Kernel
__global__ void vector_addition(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}

// Main program
int main()
{
    // Number of bytes to allocate for N doubles
    size_t bytes = N*sizeof(double);

    // Allocate memory for arrays A, B, and C on host
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    // Initialize the GPU
    gpuErrorCheck( hipFree(NULL) );

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    gpuErrorCheck( hipMalloc(&d_A, bytes) );	
    gpuErrorCheck( hipMalloc(&d_B, bytes) );
    gpuErrorCheck( hipMalloc(&d_C, bytes) );

    // Fill host arrays A, B, and C
    for(int i=0; i<N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    // Create start/stop event objects and variable for elapsed time in ms 
    hipEvent_t start, stop;
    gpuErrorCheck( hipEventCreate(&start) );
    gpuErrorCheck( hipEventCreate(&stop) );
    float elapsed_time_ms;

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrorCheck( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );
    gpuErrorCheck( hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice) );

    // Set execution configuration parameters
    //      thr_per_blk: number of GPU threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 128;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    /* =================================================
       Start timing GPU kernel
    ================================================= */
    gpuErrorCheck( hipEventRecord(start, NULL) );

    // Launch kernel
    hipLaunchKernelGGL(vector_addition, blk_in_grid, thr_per_blk , 0, 0, d_A, d_B, d_C);

    // Check for synchronous errors during kernel launch (e.g. invalid execution configuration paramters)
    gpuErrorCheck( hipGetLastError() );

    gpuErrorCheck( hipEventRecord(stop, NULL) );
    gpuErrorCheck( hipEventSynchronize(stop) );
    gpuErrorCheck( hipEventElapsedTime(&elapsed_time_ms, start, stop) ); 
    /* =================================================
       Stop timing GPU kernel
    ================================================= */

    // Check for asynchronous errors during GPU execution (after control is returned to CPU)
    gpuErrorCheck( hipDeviceSynchronize() );

    // Copy data from device array d_C to host array C
    gpuErrorCheck( hipMemcpy(C, d_C, bytes, hipMemcpyDeviceToHost) );

    // Verify results
    double tolerance = 1.0e-14;
    for(int i=0; i<N; i++)
    {
        if( fabs(C[i] - 3.0) > tolerance )
        { 
            printf("Error: value of C[%d] = %f instead of 3.0\n", i, C[i]);
            exit(1);
        }
    }	

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    // Free GPU memory
    gpuErrorCheck( hipFree(d_A) );
    gpuErrorCheck( hipFree(d_B) );
    gpuErrorCheck( hipFree(d_C) );

    gpuErrorCheck( hipEventDestroy(start) );
    gpuErrorCheck( hipEventDestroy(stop) );

    printf("\n-------------------------------\n");
    printf("__SUCCESS__\n");
    printf("-------------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("Elapsed Time (ms) = %f\n", elapsed_time_ms);
    printf("-------------------------------\n\n");

    return 0;
}
