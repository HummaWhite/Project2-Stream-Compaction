#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

inline int lowBit(int x) {
    return x & -x;
}

inline int ceilPow2(int x) {
    while (x != lowBit(x)) {
        x += lowBit(x);
    }
    return x;
}

inline int floorPow2(int x) {
    return ceilPow2(x) >> 1;
}

namespace StreamCompaction {
    namespace Common {
        int numSM();
        int warpSize();
        int maxBlockSize();
        void initCudaProperties();

        inline int getDynamicBlockSizeEXT(int n) {
            return std::max(warpSize(), std::min(maxBlockSize(), floorPow2(n / numSM())));
        }

        __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices);

        template<typename T>
        void printDevice(const T* devData, size_t n, const std::string& msg = "") {
            T* data = new T[n];
            cudaMemcpy(data, devData, n * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            if (msg != "") {
                std::cout << msg << std::endl;
            }

            for (size_t i = 0; i < n; i++) {
                std::cout << data[i] << " ";
            }
            std::cout << std::endl;

            delete[] data;
        }

        /**
        * This class is used for timing the performance
        * Uncopyable and unmovable
        *
        * Adapted from WindyDarian(https://github.com/WindyDarian)
        */
        class PerformanceTimer
        {
        public:
            PerformanceTimer()
            {
                cudaEventCreate(&event_start);
                cudaEventCreate(&event_end);
            }

            ~PerformanceTimer()
            {
                cudaEventDestroy(event_start);
                cudaEventDestroy(event_end);
            }

            void startCpuTimer()
            {
                if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
                cpu_timer_started = true;

                time_start_cpu = std::chrono::high_resolution_clock::now();
            }

            void endCpuTimer()
            {
                time_end_cpu = std::chrono::high_resolution_clock::now();

                if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

                std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
                prev_elapsed_time_cpu_milliseconds =
                    static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

                cpu_timer_started = false;
            }

            void startGpuTimer()
            {
                if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
                gpu_timer_started = true;

                cudaEventRecord(event_start);
            }

            void endGpuTimer()
            {
                cudaEventRecord(event_end);
                cudaEventSynchronize(event_end);

                if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

                cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
                gpu_timer_started = false;
            }

            float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
            {
                return prev_elapsed_time_cpu_milliseconds;
            }

            float getGpuElapsedTimeForPreviousOperation() //noexcept
            {
                return prev_elapsed_time_gpu_milliseconds;
            }

            // remove copy and move functions
            PerformanceTimer(const PerformanceTimer&) = delete;
            PerformanceTimer(PerformanceTimer&&) = delete;
            PerformanceTimer& operator=(const PerformanceTimer&) = delete;
            PerformanceTimer& operator=(PerformanceTimer&&) = delete;

        private:
            cudaEvent_t event_start = nullptr;
            cudaEvent_t event_end = nullptr;

            using time_point_t = std::chrono::high_resolution_clock::time_point;
            time_point_t time_start_cpu;
            time_point_t time_end_cpu;

            bool cpu_timer_started = false;
            bool gpu_timer_started = false;

            float prev_elapsed_time_cpu_milliseconds = 0.f;
            float prev_elapsed_time_gpu_milliseconds = 0.f;
        };
    }
}
