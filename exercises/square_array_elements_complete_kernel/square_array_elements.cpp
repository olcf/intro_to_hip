#include "hip/hip_runtime.h"
#include <stdio.h>
#include <math.h>

// Macro for checking errors in GPU API calls
#define gpuErrorCheck(call)                                                                  \
do{                                                                                          \
    hipError_t gpuErr = call;                                                                \
    if(hipSuccess != gpuErr){                                                                \
        printf("GPU Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr));  \
        exit(1);                                                                             \
    }                                                                                        \
}while(0)


// Size of array
#define N (1024 * 1024)

// GPU kernel to square the elements of an array
__global__ void square_array_elements(double *a)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // TODO: Add in code to square the array elements
}

// Main program
int main()
{
    // Number of bytes to allocate for array of size N
    size_t bytes = N*sizeof(double);

    // Allocate memory for arrays A and A_squared, which will
    // hold the results returned from the GPU.
    double A[N];
    double A_squared[N];

    // Allocate memory for array d_A on the device
    double *d_A;
    gpuErrorCheck( hipMalloc(&d_A, bytes) );

    // Initialize host arrays A and A_squared
    for(int i=0; i<N; i++)
    {
        A[i]         = (double)i;
        A_squared[i] = 0.0;
    }

    // Copy data from host array A to device array d_A
    gpuErrorCheck( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );

    // Set execution configuration parameters
    // 		thr_per_blk: number of GPU threads per grid block
    //		blk_in_grid: number of blocks in grid
    int thr_per_blk = 128;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    // Launch kernel
    hipLaunchKernelGGL(square_array_elements, blk_in_grid, thr_per_blk , 0, 0, d_A);

    // Check for synchronous errors during kernel launch (e.g. invalid execution configuration paramters)
    gpuErrorCheck( hipGetLastError() );

    // Check for asynchronous errors during GPU execution (after control is returned to CPU)
    gpuErrorCheck( hipDeviceSynchronize() );

    // Copy data from device array d_A to host array A_squared
    gpuErrorCheck( hipMemcpy(A_squared, d_A, bytes, hipMemcpyDeviceToHost) );

    // Verify results
    double tolerance = 1.0e-14;
    for(int i=0; i<N; i++)
    {
        if ( fabs( (A_squared[i] - A[i] * A[i]) ) > tolerance )
        {
            printf("A[%d] = %f instread of %f\n", i, A_squared[i], A[i] * A[i]);
            printf("Exiting...\n");
            exit(1);
        }
    }

    // Free GPU memory
    gpuErrorCheck( hipFree(d_A) );

    printf("\n--------------------------------\n");
    printf("__SUCCESS__\n");
    printf("--------------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("--------------------------------\n\n");

    return 0;
}
