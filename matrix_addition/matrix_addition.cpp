#include "hip/hip_runtime.h"
#include <stdio.h>

// Macro for checking errors in HIP API calls
#define hipErrorCheck(call)                                                                 \
do{                                                                                         \
    hipError_t hipErr = call;                                                               \
    if(hipSuccess != hipErr){                                                               \
        printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErr)); \
        exit(0);                                                                            \
    }                                                                                       \
}while(0)

// Values for MxN matrix
#define M 100
#define N 200

// GPU kernel to add matrices
__global__ void matrix_addition(double *a, double *b, double *c)
{
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row    = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && column < N)
    {
        int thread_id = row * N + column;
        c[thread_id] = a[thread_id] + b[thread_id];
    }
}

// Main program
int main()
{
    // Number of bytes to allocate for MxN matrix
    size_t bytes = M*N*sizeof(double);

    // Allocate memory for arrays A, B, and C on host
    double A[M][N];
    double B[M][N];
    double C[M][N];

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    hipErrorCheck( hipMalloc(&d_A, bytes) );
    hipErrorCheck( hipMalloc(&d_B, bytes) );
    hipErrorCheck( hipMalloc(&d_C, bytes) );

    // Initialize host arrays A, B, C
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    hipErrorCheck( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );
    hipErrorCheck( hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice) );

    // Set execution configuration parameters
    // 		threads_per_block: number of HIP threads per grid block
    //		blocks_in_grid   : number of blocks in grid
    //		(These are c structs with 3 member variables x, y, x)
    dim3 threads_per_block( 16, 16, 1 );
    dim3 blocks_in_grid( ceil( float(N) / threads_per_block.x ), ceil( float(M) / threads_per_block.y ), 1 );

    // Launch kernel
    hipLaunchKernelGGL(matrix_addition, blocks_in_grid, threads_per_block , 0, 0, d_A, d_B, d_C);

    // Check for errors in kernel launch (e.g. invalid execution configuration paramters)
    hipError_t hipErrSync  = hipGetLastError();

    // Check for errors on the GPU after control is returned to CPU
    hipError_t hipErrAsync = hipDeviceSynchronize();

    if (hipErrSync != hipSuccess) 
    { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrSync)); exit(0); }

    if (hipErrAsync != hipSuccess) 
    { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrAsync)); exit(0); }

    // Copy data from device array d_C to host array C
    hipErrorCheck( hipMemcpy(C, d_C, bytes, hipMemcpyDeviceToHost) );

    // Verify results
    double tolerance = 1.0e-14;
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            if ( (C[i][j] - 3.0) > tolerance )
            {
                printf("C[%d][%d] = %f instread of 3.0\n", i, j, C[i][j]);
            }
        }
    }

    // Free GPU memory
    hipErrorCheck( hipFree(d_A) );
    hipErrorCheck( hipFree(d_B) );
    hipErrorCheck( hipFree(d_C) );

    printf("\n--------------------------------\n");
    printf("__SUCCESS__\n");
    printf("--------------------------------\n");
    printf("M                         = %d\n", M);
    printf("N                         = %d\n", N);
    printf("Threads Per Block (x-dim) = %d\n", threads_per_block.x);
    printf("Threads Per Block (y-dim) = %d\n", threads_per_block.y);
    printf("Blocks In Grid (x-dim)    = %d\n", blocks_in_grid.x);
    printf("Blocks In Grid (y-dim)    = %d\n", blocks_in_grid.y);
    printf("--------------------------------\n\n");

    return 0;
}
