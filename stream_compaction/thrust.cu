#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::host_vector<int> in(idata, idata + n);
            thrust::device_vector<int> devIn = in;
            thrust::device_vector<int> devOut(n);

            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            auto last = thrust::exclusive_scan(devIn.begin(), devIn.end(), devOut.begin());
            
            timer().endGpuTimer();

            int size = last - devOut.begin();
            thrust::host_vector<int> out = devOut;
            memcpy(odata, out.data(), size * sizeof(int));
        }
    }
}
