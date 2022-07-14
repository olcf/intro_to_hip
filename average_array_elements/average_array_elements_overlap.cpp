#include "hip/hip_runtime.h"
#include <stdio.h>
#include <omp.h>

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
#define N (64 * 1024 * 1024)

// Stencil values
#define stencil_radius 3
#define stencil_size (2 * stencil_radius + 1)

// Number of threads in each block
#define threads_per_block 128

// Kernel
__global__ void average_array_elements(double *a, double *a_average)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x + stencil_radius;

    // If id is not in the halo...
    if( id < (N + stencil_radius) ){

        // Calculate sum of stencil elements
        double sum = 0.0;
        for(int j=-stencil_radius; j<=stencil_radius; j++){
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
    double *A_average_cpu = (double*)malloc(bytes);
    double *A, *A_average_gpu;
    gpuErrorCheck( hipHostMalloc(&A, bytes) );
    gpuErrorCheck( hipHostMalloc(&A_average_gpu, bytes) );

    // Initialize the GPU
    gpuErrorCheck( hipFree(NULL) );

    // Allocate memory for arrays d_A, d_A_average on device
    double *d_A, *d_A_average;
    gpuErrorCheck( hipMalloc(&d_A, bytes) );	
    gpuErrorCheck( hipMalloc(&d_A_average, bytes) );

    // Create start/stop event objects and variable for elapsed time in ms
    hipEvent_t start, stop;
    gpuErrorCheck( hipEventCreate(&start) );
    gpuErrorCheck( hipEventCreate(&stop) );
    float elapsed_time_ms;

    // Fill host array A with random numbers on host
    for(int i=0; i<(N+2*stencil_radius); i++)
    {
        A[i] = (double)rand()/(double)RAND_MAX;
    }

    // Start timer for total time
    double total_start, total_stop;
    total_start = omp_get_wtime();

    // Copy data from host array A to device array d_A
    gpuErrorCheck( hipMemcpyAsync(d_A, A, bytes, hipMemcpyHostToDevice, 0) );

    // Set execution configuration parameters
    //      thr_per_blk: number of GPU threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = threads_per_block;
    int blk_in_grid = ceil( float(N+2*stencil_radius) / thr_per_blk );

    // Place start event into gpu stream
    gpuErrorCheck( hipEventRecord(start, NULL) );

    // Launch kernel
    hipLaunchKernelGGL(average_array_elements, blk_in_grid, thr_per_blk, 0, 0, d_A, d_A_average);

    // Place stop event into gpu stream and calculate elapsed time in ms
    gpuErrorCheck( hipEventRecord(stop, NULL) );
    gpuErrorCheck( hipEventSynchronize(stop) );
    gpuErrorCheck( hipEventElapsedTime(&elapsed_time_ms, start, stop) );

    // Copy data from device array d_A_average to host array A_average_gpu
    gpuErrorCheck( hipMemcpyAsync(A_average_gpu, d_A_average, bytes, hipMemcpyDeviceToHost, 0) );

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

    // Sync to ensure all data is on CPU before verifying results
    gpuErrorCheck( hipDeviceSynchronize() );

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
                exit(1);
            }
        }
    }	

    // Free CPU memory
    free(A_average_cpu);
    gpuErrorCheck( hipHostFree(A) );
    gpuErrorCheck( hipHostFree(A_average_gpu) );

    // Free GPU memory
    gpuErrorCheck( hipFree(d_A) );
    gpuErrorCheck( hipFree(d_A_average) );

    printf("\n-------------------------------------\n");
    printf("__SUCCESS__\n");
    printf("-------------------------------------\n");
    printf("N                            = %d\n", N);
    printf("Threads Per Block            = %d\n", thr_per_blk);
    printf("Blocks In Grid               = %d\n", blk_in_grid);
    printf("Elapsed CPU Compute Time (s) = %f\n", cpu_compute_stop - cpu_compute_start);
    printf("Elapsed GPU Compute Time (s) = %f\n", elapsed_time_ms / 1000);
    printf("Total Elapsed Time (s)       = %f\n", total_stop - total_start);
    printf("-------------------------------------\n\n");

    return 0;
}
