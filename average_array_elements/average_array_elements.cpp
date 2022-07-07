#include "hip/hip_runtime.h"
#include <stdio.h>
#include <omp.h>

// Macro for checking errors in HIP API calls
#define hipErrorCheck(call)                                                                 \
do{                                                                                         \
    hipError_t hipErr = call;                                                               \
    if(hipSuccess != hipErr){                                                               \
        printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErr)); \
        exit(0);                                                                            \
    }                                                                                       \
}while(0)

// Size of array
#define N (64 * 1024 * 1024)

#define stencil_size 7
#define stencil_radius 3

#define block_size 128

// Kernel
__global__ void average_array_elements(double *a, double *a_average)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // If id is not in the halo...
    if( (id >= stencil_radius) && (id < (stencil_radius + N)) ){

        // Calculate sum of stencil elements
        double sum = 0.0;
        for(int j=-stencil_radius; j<(stencil_radius + 1); j++){
            sum = sum + a[id + j];
        }

        // Use sum to find average and store it in a_average.
        a_average[id] = sum / stencil_size;
    }
}
// Main program
int main()
{
    // Number of bytes to allocate for N + 2*stencil_radius array elements
    size_t bytes = (N + 2*stencil_radius)*sizeof(double);

    // Allocate memory for arrays A, A_average_cpu, A_average_gpu on host
    double *A             = (double*)malloc(bytes);
    double *A_average_cpu = (double*)malloc(bytes);
    double *A_average_gpu = (double*)malloc(bytes);

    // Initialize the GPU
    hipErrorCheck( hipFree(NULL) );

    // Allocate memory for arrays d_A, d_A_average on device
    double *d_A, *d_A_average;
    hipErrorCheck( hipMalloc(&d_A, bytes) );	
    hipErrorCheck( hipMalloc(&d_A_average, bytes) );

    // Fill host array A with random numbers on host
    for(int i=0; i<(N+2*stencil_radius); i++)
    {
        A[i] = (double)rand()/(double)RAND_MAX;
    }

    // Start timer for total time
    double total_start, total_stop;
    total_start = omp_get_wtime();

    // Start timer for CPU calculations
    double cpu_compute_start, cpu_compute_stop;
    cpu_compute_start = omp_get_wtime();

    // Average values of array elements on the host - excluding halo 
    #pragma omp parallel for default(shared)
    for(int i=0; i<(N+2*stencil_radius); i++)
    {
        if( (i >= stencil_radius) && (i < (stencil_radius + N)) ){
 
            double sum = 0;
            for(int j=-stencil_radius; j<(stencil_radius+1); j++){
    
                sum = sum + A[i + j];
            }
    
            A_average_cpu[i] = sum / stencil_size; 
        }
    }

    // Stop timer for CPU calculations
    cpu_compute_stop = omp_get_wtime();

    // Copy data from host array A to device array d_A
    hipErrorCheck( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );

    // Set execution configuration parameters
    //      thr_per_blk: number of HIP threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = block_size;
    int blk_in_grid = ceil( float(N+2*stencil_radius) / thr_per_blk );

    // Launch kernel
    hipLaunchKernelGGL(average_array_elements, blk_in_grid, thr_per_blk, 0, 0, d_A, d_A_average);

    // Copy data from device array d_A_average to host array A_average_gpu
    hipErrorCheck( hipMemcpy(A_average_gpu, d_A_average, bytes, hipMemcpyDeviceToHost) );

    // Stop timer for total time
    total_stop = omp_get_wtime();

    // Verify results - ignoring the halo at start and end of main array
    double tolerance = 1.0e-14;
    for(int i=0; i<(N+2*stencil_radius); i++)
    {
        if( (i >= stencil_radius) && (i < (stencil_radius + N)) ){

            if( fabs(A_average_cpu[i] - A_average_gpu[i]) > tolerance )
            { 
                printf("Error: value of A_average_gpu[%d] = %f instead of %f\n", i, A_average_gpu[i], A_average_cpu[i]);
                exit(-1);
            }
        }
    }	

    // Free CPU memory
    free(A);
    free(A_average_cpu);
    free(A_average_gpu);

    // Free GPU memory
    hipErrorCheck( hipFree(d_A) );
    hipErrorCheck( hipFree(d_A_average) );

    printf("\n-------------------------------------\n");
    printf("__SUCCESS__\n");
    printf("-------------------------------------\n");
    printf("N                            = %d\n", N);
    printf("Threads Per Block            = %d\n", thr_per_blk);
    printf("Blocks In Grid               = %d\n", blk_in_grid);
    printf("Elapsed CPU Compute Time (s) = %f\n", cpu_compute_stop - cpu_compute_start);
    printf("Total Elapsed Time (s)       = %f\n", total_stop - total_start);
    printf("-------------------------------------\n\n");

    return 0;
}
