/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radixsort.h>
#include "testing_helpers.hpp"
/// <summary>
/// 
/// </summary>
const int SIZE = 1 << 28; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

#define PRINT_ARRAY 0

int main(int argc, char* argv[]) {
    StreamCompaction::Common::initCudaProperties();

    // Scan tests

    /*
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    */

    /*
    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);
    */

    /*
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
    */

    zeroArray(SIZE, c);
    printDesc("work-efficient scan with shared memory - 128, NPOT");
    StreamCompaction::Efficient::scanShared(c, a, NPOT, 128);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan with shared memory - 128x2, NPOT");
    StreamCompaction::Efficient::scanShared2(c, a, NPOT, 128);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan with shared memory - 128x4, NPOT");
    StreamCompaction::Efficient::scanShared4(c, a, NPOT, 128);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(NPOT, b, c);
    /*
    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    */
    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("****************\n");
    printf("** BLOCK SCAN 1X **\n");
    printf("****************\n");

    std::fill(a, a + SIZE, 1);
    
    zeroArray(SIZE, c);
    printDesc("block scan 32x1");
    StreamCompaction::Efficient::scanBlockTest(c, a, NPOT, 32);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("block scan 64x1");
    StreamCompaction::Efficient::scanBlockTest(c, a, NPOT, 64);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif
    
    zeroArray(SIZE, c);
    printDesc("block scan 128x1");
    StreamCompaction::Efficient::scanBlockTest(c, a, NPOT, 128);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif
    
    zeroArray(SIZE, c);
    printDesc("block scan 256x1");
    StreamCompaction::Efficient::scanBlockTest(c, a, NPOT, 256);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    printf("\n");
    printf("****************\n");
    printf("** BLOCK SCAN 2X **\n");
    printf("****************\n");

    zeroArray(SIZE, c);
    printDesc("block scan 32x2");
    StreamCompaction::Efficient::scanBlockTest2(c, a, NPOT, 32);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("block scan 64x2");
    StreamCompaction::Efficient::scanBlockTest2(c, a, NPOT, 64);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("block scan 128x2");
    StreamCompaction::Efficient::scanBlockTest2(c, a, NPOT, 128);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("block scan 256x2");
    StreamCompaction::Efficient::scanBlockTest2(c, a, NPOT, 256);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    printf("\n");
    printf("****************\n");
    printf("** BLOCK SCAN 4X **\n");
    printf("****************\n");

    zeroArray(SIZE, c);
    printDesc("block scan 32x4");
    StreamCompaction::Efficient::scanBlockTest4(c, a, NPOT, 32);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("block scan 64x4");
    StreamCompaction::Efficient::scanBlockTest4(c, a, NPOT, 64);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("block scan 128x4");
    StreamCompaction::Efficient::scanBlockTest4(c, a, NPOT, 128);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("block scan 256x4");
    StreamCompaction::Efficient::scanBlockTest4(c, a, NPOT, 256);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    printf("\n");
    printf("****************\n");
    printf("** WARP SCAN 1X **\n");
    printf("****************\n");

    zeroArray(SIZE, c);
    printDesc("warp scan 32x1");
    StreamCompaction::Efficient::scanWarpTest(c, a, NPOT, 32);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 64x1");
    StreamCompaction::Efficient::scanWarpTest(c, a, NPOT, 64);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 128x1");
    StreamCompaction::Efficient::scanWarpTest(c, a, NPOT, 128);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 256x1");
    StreamCompaction::Efficient::scanWarpTest(c, a, NPOT, 256);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 512x1");
    StreamCompaction::Efficient::scanWarpTest(c, a, NPOT, 512);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 1024x1");
    StreamCompaction::Efficient::scanWarpTest(c, a, NPOT, 1024);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    printf("\n");
    printf("****************\n");
    printf("** WARP SCAN 2X **\n");
    printf("****************\n");

    zeroArray(SIZE, c);
    printDesc("warp scan 32x2");
    StreamCompaction::Efficient::scanWarpTest2(c, a, NPOT, 32);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 64x2");
    StreamCompaction::Efficient::scanWarpTest2(c, a, NPOT, 64);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 128x2");
    StreamCompaction::Efficient::scanWarpTest2(c, a, NPOT, 128);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 256x2");
    StreamCompaction::Efficient::scanWarpTest2(c, a, NPOT, 256);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 512x2");
    StreamCompaction::Efficient::scanWarpTest2(c, a, NPOT, 512);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 1024x2");
    StreamCompaction::Efficient::scanWarpTest2(c, a, NPOT, 1024);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    printf("\n");
    printf("****************\n");
    printf("** WARP SCAN 4X **\n");
    printf("****************\n");

    zeroArray(SIZE, c);
    printDesc("warp scan 32x4");
    StreamCompaction::Efficient::scanWarpTest4(c, a, NPOT, 32);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 64x4");
    StreamCompaction::Efficient::scanWarpTest4(c, a, NPOT, 64);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 128x4");
    StreamCompaction::Efficient::scanWarpTest4(c, a, NPOT, 128);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 256x4");
    StreamCompaction::Efficient::scanWarpTest4(c, a, NPOT, 256);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 512x4");
    StreamCompaction::Efficient::scanWarpTest4(c, a, NPOT, 512);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif

    zeroArray(SIZE, c);
    printDesc("warp scan 1024x4");
    StreamCompaction::Efficient::scanWarpTest4(c, a, NPOT, 1024);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
#if PRINT_ARRAY
    printArray(1024, c);
#endif
    
    /*
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.

    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan, NPOT");
    count = StreamCompaction::CPU::compactWithScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(c, a, SIZE);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(c, a, NPOT);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact with shared memory, NPOT");
    count = StreamCompaction::Efficient::compactShared(c, a, NPOT);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("*****************************\n");

    genArray(SIZE - 1, a, INT32_MAX);
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    zeroArray(NPOT, b);
    printDesc("cpu std::sort, NPOT");
    StreamCompaction::CPU::sort(b, a, NPOT);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono)");
    printCmpResult(NPOT, b, b);

    zeroArray(NPOT, c);
    printDesc("gpu radix sort, NPOT");
    StreamCompaction::RadixSort::sort(c, a, NPOT);
    printElapsedTime(StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(NPOT, c, b);

    zeroArray(NPOT, c);
    printDesc("gpu radix sort with shared memory, NPOT");
    StreamCompaction::RadixSort::sortShared(c, a, NPOT);
    printElapsedTime(StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(NPOT, c, b);
    */

    delete[] a;
    delete[] b;
    delete[] c;
}
