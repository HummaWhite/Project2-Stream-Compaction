#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace gal {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

    template<typename T>
    __host__ __device__ T lowBit(T x) {
        return x & ~x;
    }

    __device__ void bitonicSwapWarp(uint32_t* array, uint32_t baseIdx, uint32_t stride, uint32_t level) {
    }

    __device__ void bitonicSwapBlock(uint32_t* array, uint32_t idx, uint32_t stride, uint32_t level) {
        for (uint32_t subStride = stride, subLevel = level; subStride > 1; subStride >>= 1, subLevel--) {
            uint32_t baseIdx = (idx >> subLevel << subLevel) + ((idx & (subStride - 1)) >> 1);
            uint32_t dir = (idx >> level) & 1;
            uint32_t mappedIdx = baseIdx ^ ((stride - 1) * dir);
            uint32_t compareIdx = mappedIdx ^ (subStride >> 1);

            uint32_t a = array[mappedIdx];
            uint32_t b = array[compareIdx];

            array[mappedIdx] = min(a, b);
            array[compareIdx] = max(a, b);
            __syncthreads();
        }
    }

    __global__ void bitonicSort32uKernel(uint32_t* array, uint32_t n) {
        uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx >= n) {
            return;
        }
        idx <<= 1;

        for (uint32_t stride = 2, level = 1; stride <= n; stride <<= 1, level++) {
            for (uint32_t subStride = stride, subLevel = level; subStride > 1; subStride >>= 1, subLevel--) {
                uint32_t baseIdx = (idx >> subLevel << subLevel) + ((idx & (subStride - 1)) >> 1);
                uint32_t dir = (idx >> level) & 1;
                uint32_t mappedIdx = baseIdx ^ ((stride - 1) * dir);
                uint32_t compareIdx = mappedIdx ^ (subStride >> 1);

                uint32_t a = array[mappedIdx];
                uint32_t b = array[compareIdx];

                array[mappedIdx] = min(a, b);
                array[compareIdx] = max(a, b);
                __syncthreads();
            }
        }
    }

    __global__ void bitonicSort32uPartialKernel(uint32_t* array, uint32_t n, uint32_t level, uint32_t subLevel) {
        uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx >= n) {
            return;
        }
        idx <<= 1;

        uint32_t stride = 1 << level;
        uint32_t subStride = 1 << subLevel;

        uint32_t baseIdx = (idx >> subLevel << subLevel) + ((idx & (subStride - 1)) >> 1);
        uint32_t dir = (idx >> level) & 1;
        uint32_t mappedIdx = baseIdx ^ ((stride - 1) * dir);
        uint32_t compareIdx = mappedIdx ^ (subStride >> 1);

        uint32_t a = array[mappedIdx];
        uint32_t b = array[compareIdx];

        array[mappedIdx] = min(a, b);
        array[compareIdx] = max(a, b);
    }

    __global__ void bitonicSort32uSharedKernel(uint32_t* array, uint32_t n, uint32_t level) {
        extern __shared__ uint32_t shared[];

        uint32_t globalIdx = blockDim.x * blockIdx.x + threadIdx.x;

        if (globalIdx >= n) {
            return;
        }
        uint32_t idx = threadIdx.x << 1;

        shared[idx + 0] = array[globalIdx * 2 + 0];
        shared[idx + 1] = array[globalIdx * 2 + 1];
        __syncthreads();

        uint32_t stride = 1 << level;

        for (uint32_t subStride = stride, subLevel = level; subStride > 1; subStride >>= 1, subLevel--) {
            uint32_t baseIdx = (idx >> subLevel << subLevel) + ((idx & (subStride - 1)) >> 1);
            uint32_t dir = (idx >> level) & 1;
            uint32_t mappedIdx = baseIdx ^ ((stride - 1) * dir);
            uint32_t compareIdx = mappedIdx ^ (subStride >> 1);

            uint32_t a = shared[mappedIdx];
            uint32_t b = shared[compareIdx];

            array[mappedIdx] = min(a, b);
            array[compareIdx] = max(a, b);
            __syncthreads();
        }
        array[globalIdx * 2 + 0] = shared[idx + 0];
        array[globalIdx * 2 + 1] = shared[idx + 1];
    }

    void bitonicSort32u(uint32_t* out, uint32_t* in, uint32_t n, uint32_t blockSize) {
        auto size = sizeof(uint32_t) * n;

        uint32_t* devArray;
        cudaMalloc(&devArray, size);
        cudaMemcpy(devArray, in, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

        uint32_t blockNum = ceilDiv(n >> 1, blockSize);

        timer().startGpuTimer();

        for (uint32_t stride = 2, level = 1; stride <= n; stride <<= 1, level++) {
            if (stride <= blockSize) {
                LAUNCH_KERNEL(bitonicSort32uSharedKernel, blockNum, blockSize)(devArray, n, level);
            }
            else {
                for (uint32_t subStride = stride, subLevel = level; subStride > 1; subStride >>= 1, subLevel--) {
                    LAUNCH_KERNEL(bitonicSort32uPartialKernel, blockNum, blockSize)(devArray, n, level, subLevel);
                }
            }
        }

        /*
        for (uint32_t stride = 2, level = 1; stride <= n; stride <<= 1, level++) {
            for (uint32_t subStride = stride, subLevel = level; subStride > 1; subStride >>= 1, subLevel--) {
                LAUNCH_KERNEL(bitonicSort32uPartialKernel, blockNum, blockSize)(devArray, n, level, subLevel);
            }
        }
        */
        //LAUNCH_KERNEL(bitonicSort32uKernel, blockNum, blockSize)(devArray, n);

        timer().endGpuTimer();

        cudaMemcpy(out, devArray, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaFree(devArray);
    }
}
