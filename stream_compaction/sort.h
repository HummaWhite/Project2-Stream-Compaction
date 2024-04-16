#pragma once

#include "common.h"

namespace gal {
    StreamCompaction::Common::PerformanceTimer& timer();

    void bitonicSort32u(uint32_t* out, uint32_t* in, uint32_t n, uint32_t blockSize);
}
