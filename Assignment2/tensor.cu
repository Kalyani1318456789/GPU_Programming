#include <cstdio>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace wmma;

// size of the tile (for simplicity use 16x16x16)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void init(half *A, half *B){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    A[tid] = tid;
    B[tid] = tid;
}

// A: (M×K) ; B: (K×N) ; C: (M×N)
__global__ void tensorCoreGemmKernel(
    half *A,
    half *B,
    float *C,
    int M,
    int N,
    int K)
{
    // each warp computes one tile of output C
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    int warpM = warpId / 4;
    int warpN = warpId % 4;

    int mTile = warpM * WMMA_M;
    int nTile = warpN * WMMA_N;

    if (mTile >= M || nTile >= N)
        return;

    // Declare the fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K,
             half, row_major> aFrag;

    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K,
             half, row_major> bFrag;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K,
             float> cFrag;

    // Initialize the output to zero
    fill_fragment(cFrag, 0.0f);

    // K=64 so 4 tiles are required
    for (int i = 0; i < 4; i++)
    {
        int bind = nTile + i * 16 * N;
        int aind = mTile * K + i * 16;

        // load submatrix A into registers
        load_matrix_sync(aFrag, A + aind, K);

        // load submatrix B into registers
        load_matrix_sync(bFrag, B + bind, N);

        // do the multiplication
        mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    // starting address of the tile C
    int cind = mTile * N + nTile;

    // write result into C
    store_matrix_sync(C + cind, cFrag, N, mem_row_major);
}

int main()
{
    int M = 64, N = 64, K = 64;

    half *devA;
    half *devB;

    float *devC, *hostC;

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&devA, M * K * sizeof(half));
    cudaMalloc(&devB, K * N * sizeof(half));
    cudaMalloc(&devC, M * N * sizeof(float));

    hostC = (float *)malloc(M * N * sizeof(float));

    cudaError_t err = cudaGetLastError();

    // 64x64 matrix has 16 tiles
    init<<<4, 1024>>>(devA, devB);

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    // 16 tiles, one warp computes one tile
    cudaEventRecord(start, 0);

    tensorCoreGemmKernel<<<1, 512>>>(
        devA,
        devB,
        devC,
        M,
        N,
        K
    );

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Kernel execution time: %f milli seconds\n",
           elapsedTime);

    err = cudaGetLastError();

    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    cudaMemcpy(
        hostC,
        devC,
        sizeof(float) * M * N,
        cudaMemcpyDeviceToHost
    );

    // for a matrix of size 64x64 we have 16 tiles of size 16x16
    // (0,0) (0,1) (0,2) (0,3)
    // (1,0) (1,1) (1,2) (1,3)
    // (2,0) (2,1) (2,2) (2,3)
    // (3,0) (3,1) (3,2) (3,3)

    // starting address of the tiles
    // (0,0) - 0
    // (0,1) - 16
    // (0,2) - 32
    // (0,3) - 48
    //
    // (1,0) - X
    // (1,1) - X+16
    // (1,2) - X+32
    // (1,3) - X+48
    //
    // (2,0) - Y
    // (2,1) - Y+16
    // (2,2) - Y+32
    // (2,3) - Y+48
    //
    // (3,0) - Z
    // (3,1) - Z+16
    // (3,2) - Z+32
    // (3,3) - Z+48
    //
    // X=64x16, Y=64x32, Z=64x48

    fprintf(stderr,
            "last two elements in C(0,0) %f %f\n",
            hostC[15 * 64 + 14],
            hostC[15 * 64 + 15]);

    fprintf(stderr,
            "last two elements in C(0,1) %f %f\n",
            hostC[15 * 64 + 30],
            hostC[15 * 64 + 31]);

    fprintf(stderr,
            "last two elements in C(0,2) %f %f\n",
            hostC[15 * 64 + 46],
            hostC[15 * 64 + 47]);

    fprintf(stderr,
            "last two elements in C(0,3) %f %f\n",
            hostC[15 * 64 + 62],
            hostC[15 * 64 + 63]);

    fprintf(stderr,
            "last two elements in C(1,0) %f %f\n",
            hostC[31 * 64 + 14],
            hostC[31 * 64 + 15]);

    fprintf(stderr,
            "last two elements in C(1,1) %f %f\n",
            hostC[31 * 64 + 30],
            hostC[31 * 64 + 31]);

    fprintf(stderr,
            "last two elements in C(1,2) %f %f\n",
            hostC[31 * 64 + 46],
            hostC[31 * 64 + 47]);

    fprintf(stderr,
            "last two elements in C(1,3) %f %f\n",
            hostC[31 * 64 + 62],
            hostC[31 * 64 + 63]);

    fprintf(stderr,
            "last two elements in C(2,0) %f %f\n",
            hostC[47 * 64 + 14],
            hostC[47 * 64 + 15]);

    fprintf(stderr,
            "last two elements in C(2,1) %f %f\n",
            hostC[47 * 64 + 30],
            hostC[47 * 64 + 31]);

    fprintf(stderr,
            "last two elements in C(2,2) %f %f\n",
            hostC[47 * 64 + 46],
            hostC[47 * 64 + 47]);

    fprintf(stderr,
            "last two elements in C(2,3) %f %f\n",
            hostC[47 * 64 + 62],
            hostC[47 * 64 + 63]);

    fprintf(stderr,
            "last two elements in C(3,0) %f %f\n",
            hostC[63 * 64 + 14],
            hostC[63 * 64 + 15]);

    fprintf(stderr,
            "last two elements in C(3,1) %f %f\n",
            hostC[63 * 64 + 30],
            hostC[63 * 64 + 31]);

    fprintf(stderr,
            "last two elements in C(3,2) %f %f\n",
            hostC[63 * 64 + 46],
            hostC[63 * 64 + 47]);

    fprintf(stderr,
            "last two elements in C(3,3) %f %f\n",
            hostC[63 * 64 + 62],
            hostC[63 * 64 + 63]);


    // checking that all 16 tiles are written
    int count = 0;

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            int row = i * 16;
            int col = j * 16;

            if(hostC[row * 64 + col] != 0)
                count++;
        }
    }

    printf("Count=%d \n", count);

    return 0;
}
