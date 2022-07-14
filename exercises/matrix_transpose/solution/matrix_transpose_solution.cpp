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

// Values for NxN matrix
#define N 2048

// GPU kernel to transpose a matrix
__global__ void matrix_transpose(double *a, double *a_transpose)
{
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row    = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < N && column < N)
    {
        int thread_id   = row * N + column;
        int thread_id_T = column * N + row;

        a_transpose[thread_id_T] = a[thread_id];
    }
}

// Main program
int main()
{
    // Number of bytes to allocate for NxN matrix
    size_t bytes = N*N*sizeof(double);

    // Allocate memory for arrays A and A_from_gpu
    double A[N][N];
    double A_from_gpu[N][N];

    // Allocate memory for arrays d_A and d_A_transpose
    double *d_A, *d_A_transpose;
    gpuErrorCheck( hipMalloc(&d_A, bytes) );
    gpuErrorCheck( hipMalloc(&d_A_transpose, bytes) );

    // Initialize host arrays A, B, C
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            A[i][j]           = (double)rand()/(double)RAND_MAX;
            A_from_gpu[i][j] = 0.0;
        }
    }

    // Copy data from host array A to device array d_A
    gpuErrorCheck( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );

    // Set execution configuration parameters
    // 		threads_per_block: number of GPU threads per grid block
    //		blocks_in_grid   : number of blocks in grid
    dim3 threads_per_block( 16, 16, 1 );
    dim3 blocks_in_grid( ceil( float(N) / threads_per_block.x ), ceil( float(N) / threads_per_block.y ), 1 );

    // Launch kernel
    hipLaunchKernelGGL(matrix_transpose, blocks_in_grid, threads_per_block , 0, 0, d_A, d_A_transpose);

    // Check for synchronous errors during kernel launch (e.g. invalid execution configuration paramters)
    gpuErrorCheck( hipGetLastError() );

    // Check for asynchronous errors during GPU execution (after control is returned to CPU)
    gpuErrorCheck( hipDeviceSynchronize() );

    // Launch kernel
    hipLaunchKernelGGL(matrix_transpose, blocks_in_grid, threads_per_block , 0, 0, d_A_transpose, d_A);

    // Check for synchronous errors during kernel launch (e.g. invalid execution configuration paramters)
    gpuErrorCheck( hipGetLastError() );

    // Check for asynchronous errors during GPU execution (after control is returned to CPU)
    gpuErrorCheck( hipDeviceSynchronize() );

    // Copy data from device array d_A to A_from_gpu
    gpuErrorCheck( hipMemcpy(A_from_gpu, d_A, bytes, hipMemcpyDeviceToHost) );

    // Verify results
    double tolerance = 1.0e-14;
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            if( fabs(A[i][j] - A_from_gpu[i][j]) > tolerance)
            {
                printf("A_from_gpu[%d][%d] = %0.14f instead of A[%d][%d] = %0.14f\n", i, j, A_from_gpu[i][j], i, j, A[i][j]);
                exit(1);
            }
        }
    }

    // Free GPU memory
    gpuErrorCheck( hipFree(d_A) );
    gpuErrorCheck( hipFree(d_A_transpose) );

    printf("\n--------------------------------\n");
    printf("__SUCCESS__\n");
    printf("--------------------------------\n");
    printf("N                         = %d\n", N);
    printf("Threads Per Block (x-dim) = %d\n", threads_per_block.x);
    printf("Threads Per Block (y-dim) = %d\n", threads_per_block.y);
    printf("Blocks In Grid (x-dim)    = %d\n", blocks_in_grid.x);
    printf("Blocks In Grid (y-dim)    = %d\n", blocks_in_grid.y);
    printf("--------------------------------\n\n");

    return 0;
}
