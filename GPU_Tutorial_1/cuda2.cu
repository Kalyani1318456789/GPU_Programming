#include <stdio.h>
#include <cuda_runtime.h>

__device__ int count = 0;

__device__ void barrier()
{
    atomicAdd(&count, 1);

    __syncthreads();

    if (threadIdx.x == 0)
    {
        while (count < blockDim.x)
        {
        }
    }

    __syncthreads();
}

__global__ void test()
{
    printf("Thread %d: Before barrier\n", threadIdx.x);

    barrier();

    printf("Thread %d: After barrier\n", threadIdx.x);
}

int main()
{
    test<<<1, 8>>>();

    cudaDeviceSynchronize();

    return 0;
}
