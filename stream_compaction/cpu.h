#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compactWithoutScan(int n, int *odata, const int *idata);

        int compactWithScan(int n, int *odata, const int *idata);

        void sort(int* out, const int* in, int n);

        void bitonicSort32u(uint32_t* out, uint32_t* in, uint32_t n);
    }
}
