
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"

cudaError_t matMulWithCuda(int* c, const int* a, const int* b);

constexpr int TILE_SIZE = 16;

__global__ void MatMul(int* c, const int* a, const int* b, int matWidth) {
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int baseRow = blockIdx.x * TILE_SIZE;
    int baseCol = blockIdx.y * TILE_SIZE;
    int loop = matWidth / TILE_SIZE;
    int val = 0;
    for (int startMulIndex = 0; startMulIndex < 1024; startMulIndex += TILE_SIZE) {
        tileA[threadIdx.x][threadIdx.y] = a[(baseRow + threadIdx.x)       * matWidth + startMulIndex + threadIdx.y];
        tileB[threadIdx.x][threadIdx.y] = b[(startMulIndex + threadIdx.x) * matWidth + baseCol       + threadIdx.y];

        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i) {
            val += tileA[threadIdx.x][i] * tileB[i][threadIdx.y];
        }
        __syncthreads();
    }
    c[(baseRow + threadIdx.x) * matWidth + baseCol + threadIdx.y] = val;
}

int main()
{
    int* a = new int[1024 * 1024];
    int* b = new int [1024 * 1024];
    int* c = new int [1024 * 1024];
    printf("Creating data\n");
    for (int row = 0; row < 1024; ++row) {
        for (int col = 0; col < 1024; ++col) {
            a[row * 1024 + col] = row > col ? row : col;
            b[row * 1024 + col] = row > col ? col : row;
            c[row * 1024 + col] = 0;
        }
    }
    matMulWithCuda(c, a, b);
    printf("Checking calculate result...\n");
    for (int row = 0; row < 1024; ++row) {
        for (int col = 0; col < 1024; ++col) {
            int val = 0;
            for (int i = 0; i < 1024; ++i) {
                val += a[row*1024+i] * b[i * 1024 + col];
            }
            if (val != c[row * 1024 + col]) {
                printf("Calculation wrong at %d %d %d != %d\n", row, col, c[row * 1024 + col], val);
                return -1;
            }
        }
    }
    printf("Correct calculation...\n");

    return 0;
}

cudaError_t matMulWithCuda(int* c, const int* a, const int* b) {
    //Malloc
    cudaError_t result = cudaErrorUnknown;

    result = cudaSetDevice(0);
    if (result != cudaSuccess) {
        printf("Set device failed");
        return result;
    }

    int* devA, * devB, * devC;
    result = cudaMalloc((void**)&devA, 1024 * 1024 * sizeof(int));
    if (result != cudaSuccess) {
        printf("Malloc Mat A failed");
        return result;
    }
    result = cudaMalloc((void**)&devB, 1024 * 1024 * sizeof(int));
    if (result != cudaSuccess) {
        printf("Malloc Mat B failed");
        cudaFree(devA);
        return result;
    }
    result = cudaMalloc((void**)&devC, 1024 * 1024 * sizeof(int));
    if (result != cudaSuccess) {
        printf("Malloc Mat C failed");
        cudaFree(devA);
        cudaFree(devB);
        return result;
    }
    if(result == cudaSuccess) result = cudaMemcpy(devA, a, 1024 * 1024 * sizeof(int), cudaMemcpyHostToDevice);

    if(result != cudaSuccess) printf("Input data failed for A %d\n", result);

    if(result == cudaSuccess) result = cudaMemcpy(devB, b, 1024 * 1024 * sizeof(int), cudaMemcpyHostToDevice);

    if (result != cudaSuccess) printf("Input data failed for B %d\n", result);

    if (result == cudaSuccess) {
        dim3 gridDim(64, 64, 1);
        dim3 blockDim(16, 16, 1);
        printf("Start calculate \n");
        MatMul <<< gridDim, blockDim >>> (devC, devA, devB, 1024);
        printf("End calculate \n");
        result = cudaMemcpy(c, devC, 1024 * 1024 * sizeof(int), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            printf("Get calculation result failed\n");
        }
    }

    result = cudaFree(devA);
    result = cudaFree(devB);
    result = cudaFree(devC);
    return result;
}