
#include "utils.h"
#include "random.h"
#include <malloc.h>


#define L1_CACHE_LINE_SIZE 64


void latinHypercube(Random *random, Float *dest, size_t nSamples, size_t nDim) {
    Float delta = 1 / (Float) nSamples;
    for (size_t i = 0; i < nSamples; ++i)
        for (size_t j = 0; j < nDim; ++j)
            dest[nDim * i + j] = (i + random->nextFloat()) * delta;
    for (size_t i = 0; i < nDim; ++i) {
        for (size_t j = 0; j < nSamples; ++j) {
            size_t other = random->nextSize(nSamples);
            std::swap(dest[nDim * j + i], dest[nDim * other + i]);
        }
    }
}

void * __restrict allocAligned(size_t size) {
#if defined(__WINDOWS__)
    return _aligned_malloc(size, L1_CACHE_LINE_SIZE);
#elif defined(__OSX__)
    /* OSX malloc already returns 16-byte aligned data suitable
       for AltiVec and SSE computations */
    return malloc(size);
#else
    return memalign(L1_CACHE_LINE_SIZE, size);
#endif
}

void freeAligned(void *ptr) {
#if defined(__WINDOWS__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
