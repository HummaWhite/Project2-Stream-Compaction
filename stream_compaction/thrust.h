#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Thrust {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void sort32u(uint32_t* out, uint32_t* in, uint32_t n);

        void stableSort32u(uint32_t* out, uint32_t* in, uint32_t n);
    }
}
