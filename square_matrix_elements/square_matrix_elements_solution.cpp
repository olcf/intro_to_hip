#include "hip/hip_runtime.h"
#include <stdio.h>
#include <math.h>

// Macro for checking errors in HIP API calls
#define hipErrorCheck(call)                                                                  \
do{                                                                                          \
    hipError_t hipErr = call;                                                                \
    if(hipSuccess != hipErr){                                                                \
        printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErr)); \
        exit(0);                                                                             \
    }                                                                                        \
}while(0)


// Values for MxN matrix
#define M 100
#define N 200

// GPU kernel to square the elements of a matrix
__global__ void square_matrix_elements(double *a)
{
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row    = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && column < N)
    {
        int thread_id = row * N + column;
        // TODO Square the elements of a matrix
        a[thread_id] = a[thread_id] * a[thread_id];
    }
}

// Main program
int main()
{
    // Number of bytes to allocate for MxN matrix
    size_t bytes = M*N*sizeof(double);

    // Allocate memory for arrays A and A_squared, which will
    // hold the results returned from the GPU.
    double A[M][N];
    double A_squared[M][N];

    // Allocate memory for array d_A on the device
    double *d_A;
    hipErrorCheck( hipMalloc(&d_A, bytes) );

    // Initialize host arrays A and A_squared
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            A[i][j]         = i * j;
            A_squared[i][j] = 0.0;
        }
    }

    // Copy data from host array A to device array d_A
    // TODO Replace the ?s with the correct buffers (pointers)
    hipErrorCheck( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );

    // Set execution configuration parameters
    // 		threads_per_block: number of HIP threads per grid block
    //		blocks_in_grid   : number of blocks in grid
    //		(These are c structs with 3 member variables x, y, x)
    dim3 threads_per_block( 16, 16, 1 );
    dim3 blocks_in_grid( ceil( float(N) / threads_per_block.x ), ceil( float(M) / threads_per_block.y ), 1 );

    // Launch kernel
    hipLaunchKernelGGL(square_matrix_elements, blocks_in_grid, threads_per_block , 0, 0, d_A);

    // Check for errors in kernel launch (e.g. invalid execution configuration paramters)
    hipError_t hipErrSync  = hipGetLastError();

    // Check for errors on the GPU after control is returned to CPU
    hipError_t hipErrAsync = hipDeviceSynchronize();

    if (hipErrSync != hipSuccess) 
    { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrSync)); exit(0); }

    if (hipErrAsync != hipSuccess) 
    { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrAsync)); exit(0); }

    // Copy data from device array d_A to host array A_squared
    hipErrorCheck( hipMemcpy(A_squared, d_A, bytes, hipMemcpyDeviceToHost) );

    // Verify results
    double tolerance = 1.0e-14;
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            if ( fabs( (A_squared[i][j] - A[i][j] * A[i][j]) ) > tolerance )
            {
                printf("C[%d][%d] = %f instread of %f\n", i, j, A_squared[i][j], A[i][j] * A[i][j]);
                printf("Exiting...\n");
                exit(1);
            }
        }
    }

    // Free GPU memory
    hipErrorCheck( hipFree(d_A) );

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
