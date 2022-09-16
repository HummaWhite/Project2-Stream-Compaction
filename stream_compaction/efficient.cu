#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        enum class ScanSource { Host, Device };

        template<typename T>
        struct DevMemRec {
            DevMemRec() = default;
            DevMemRec(T* ptr, int size) : ptr(ptr), size(size) {}
            DevMemRec(T* ptr, int size, int realSize) : ptr(ptr), size(size), realSize(realSize) {}
            T* ptr;
            int size;
            int realSize;
        };

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernPartialUpSweep(int* data, int n, int stride) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x + 1;
            if (idx > n) {
                return;
            }
            int mappedIdx = idx * stride - 1;
            data[mappedIdx] += data[mappedIdx - stride / 2];
        }

        __global__ void kernPartialDownSweep(int* data, int n, int stride) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x + 1;
            if (idx > n) {
                return;
            }
            int mappedIdx = idx * stride - 1;
            data[mappedIdx] += data[mappedIdx - stride / 2];
            data[mappedIdx - stride / 2] = data[mappedIdx] - data[mappedIdx - stride / 2];
        }

        __global__ void kernBlockScanShared(int* data, int* blockSum, int n) {
            extern __shared__ int shared[];

            int idx = threadIdx.x + 1;
            int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;

            if (globalIdx > n) {
                return;
            }

            shared[idx - 1] = data[globalIdx];
            if (idx == blockDim.x) {
                shared[idx] = shared[idx - 1];
            }
            __syncthreads();
#pragma unroll
            for (int stride = 1, active = blockDim.x >> 1; stride < blockDim.x; stride <<= 1, active >>= 1) {
                if (idx <= active) {
                    int mappedIdx = idx * stride * 2 - 1;
                    shared[mappedIdx] += shared[mappedIdx - stride];
                }
                __syncthreads();
            }

            if (idx == 1) {
                shared[blockDim.x - 1] = 0;
            }
            __syncthreads();
#pragma unroll
            for (int stride = blockDim.x >> 1, active = 1; stride >= 1; stride >>= 1, active <<= 1) {
                if (idx <= active) {
                    int mappedIdx = idx * stride * 2 - 1;
                    shared[mappedIdx] += shared[mappedIdx - stride];
                    shared[mappedIdx - stride] = shared[mappedIdx] - shared[mappedIdx - stride];
                }
                __syncthreads();
            }
            data[globalIdx] = shared[idx - 1];

            if (idx == 1) {
                blockSum[blockIdx.x] = shared[blockDim.x - 1] + shared[blockDim.x];
            }
        }

        __global__ void kernScannedBlockAdd(int* data, const int* blockSum, int n) {
            extern __shared__ int sum;
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) {
                return;
            }

            if (threadIdx.x == 0) {
                sum = blockSum[blockIdx.x];
            }
            __syncthreads();
            data[idx] += sum;
        }

        void devScanInPlace(int* devData, int size) {
            if (size != ceilPow2(size)) {
                throw std::runtime_error("devScanInPlace:: size not pow of 2");
            }

            for (int stride = 2; stride <= size; stride <<= 1) {
                int num = size / stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = ceilDiv(num, blockSize);
                kernPartialUpSweep<<<blockNum, blockSize>>>(devData, num, stride);
            }

            cudaMemset(devData + size - 1, 0, sizeof(int));
            for (int stride = size; stride >= 2; stride >>= 1) {
                int num = size / stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = ceilDiv(num, blockSize);
                kernPartialDownSweep<<<blockNum, blockSize>>>(devData, num, stride);
            }
        }

        void devBlockScanInPlaceShared(int* devData, int* devBlockSum, int size, int blockSize) {
            if (size % blockSize != 0) {
                throw std::runtime_error("devBlockScanInPlaceShared:: size not multiple of BlockSize");
            }
            kernBlockScanShared<<<size / blockSize, blockSize>>>(devData, devBlockSum, size);
        }

        void devScanInPlaceShared(int* devData, int size) {
            const int blockSize = 128;
            if (size % blockSize != 0 || size <= blockSize) {
                throw std::runtime_error("devScanInPlaceShared:: size not multiple of BlockSize");
            }

            std::vector<DevMemRec<int>> sums;
            for (int i = size; i >= 1; i = ceilDiv(i, blockSize)) {
                int size = ceilDiv(i, blockSize) * blockSize;
                int* sum;
                cudaMalloc(&sum, size * sizeof(int));
                sums.push_back({ sum, size, i });

                if (i == 1) {
                    break;
                }
            }
            cudaMemcpy(sums[0].ptr, devData, size * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

            timer().startGpuTimer();
            for (int i = 0; i + 1 < sums.size(); i++) {
                devBlockScanInPlaceShared(sums[i].ptr, sums[i + 1].ptr, sums[i].size, blockSize);
            }

            for (int i = sums.size() - 2; i > 0; i--) {
                kernScannedBlockAdd<<<sums[i].size, blockSize>>>(sums[i - 1].ptr, sums[i].ptr, sums[i - 1].size);
            }
            timer().endGpuTimer();

            cudaDeviceSynchronize();
            cudaMemcpy(devData, sums[0].ptr, size * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

            for (auto& sum : sums) {
                cudaFree(sum.ptr);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int size = ceilPow2(n);
            int* data;
            cudaMalloc(&data, size * sizeof(int));
            cudaMemcpy(data, idata, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            devScanInPlace(data, size);

            timer().endGpuTimer();
            
            cudaMemcpy(odata, data, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(data);
        }

        void scanWithSharedMemory(int* out, const int* in, int n, int blockSize) {
            // Just to keep the edge case correct
            // If n <= blockSize, there's no need to perform a GPU scan
            if (n <= blockSize) {
                out[0] = 0;
                for (int i = 1; i < n; i++) {
                    out[i] = out[i - 1] + in[i - 1];
                }
                return;
            }

            std::vector<DevMemRec<int>> sums;
            for (int i = n; i >= 1; i = ceilDiv(i, blockSize)) {
                int size = ceilDiv(i, blockSize) * blockSize;
                int* sum;
                cudaMalloc(&sum, size * sizeof(int));
                sums.push_back({ sum, size, i });

                if (i == 1) {
                    break;
                }
            }
            cudaMemcpy(sums[0].ptr, in, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int i = 0; i + 1 < sums.size(); i++) {
                //Common::printDeviceInGroup(sums[i].ptr, sums[i].realSize, blockSize, "before");
                devBlockScanInPlaceShared(sums[i].ptr, sums[i + 1].ptr, sums[i].size, blockSize);
                //Common::printDeviceInGroup(sums[i].ptr, sums[i].realSize, blockSize, "after");
                //Common::printDeviceInGroup(sums[i + 1].ptr, sums[i + 1].realSize, blockSize, "seg sum");
            }

            for (int i = sums.size() - 2; i > 0; i--) {
                //Common::printDeviceInGroup(sums[i - 1].ptr, sums[i - 1].realSize, blockSize, "#before");
                kernScannedBlockAdd<<<sums[i].size, blockSize>>>(sums[i - 1].ptr, sums[i].ptr, sums[i - 1].size);
                //Common::printDeviceInGroup(sums[i - 1].ptr, sums[i - 1].realSize, blockSize, "#after");
            }
            timer().endGpuTimer();

            cudaDeviceSynchronize();
            cudaMemcpy(out, sums[0].ptr, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            for (auto& sum : sums) {
                cudaFree(sum.ptr);
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO
            int* in, * out;
            int bytes = n * sizeof(int);
            cudaMalloc(&in, bytes);
            cudaMalloc(&out, bytes);
            cudaMemcpy(in, idata, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

            int size = ceilPow2(n);
            int* indices;
            cudaMalloc(&indices, size * sizeof(int));

            timer().startGpuTimer();

            int blockSize = Common::getDynamicBlockSizeEXT(n);
            int blockNum = ceilDiv(n, blockSize);

            Common::kernMapToBoolean<<<blockNum, blockSize>>>(n, indices, in);
            devScanInPlace(indices, size);
            Common::kernScatter<<<blockNum, blockSize>>>(n, out, in, in, indices);

            timer().endGpuTimer();

            int compactedSize;
            cudaMemcpy(&compactedSize, indices + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            compactedSize += (idata[n - 1] != 0);

            cudaMemcpy(odata, out, compactedSize * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            cudaFree(indices);
            cudaFree(in);
            cudaFree(out);

            return compactedSize;
        }
    }
}
