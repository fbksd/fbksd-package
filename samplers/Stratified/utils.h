#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

//Constants
#define ONE_MINUS_EPS_FLT 0x1.fffffep-1f
#define ONE_MINUS_EPS_DBL 0x1.fffffffffffff7p-1
#define RCPOVERFLOW_FLT   0x1p-128f
#define RCPOVERFLOW_DBL   0x1p-1024
#define ONE_MINUS_EPS     ONE_MINUS_EPS_FLT
#define RCPOVERFLOW       RCPOVERFLOW_FLT

class Random;
using Float = float;

template <typename T>
struct TPoint2
{
    TPoint2(): x(0), y(0)
    {}

    TPoint2(T v): x(v), y(v)
    {}

    TPoint2(T x, T y): x(x), y(y)
    {}

    T x, y;
};

using Point2 = TPoint2<Float>;
using Point2i = TPoint2<int>;
using Point2f = TPoint2<float>;
using Point2d = TPoint2<double>;
using Vector2i = TPoint2<int>;

namespace math
{

inline int log2i(uint32_t value) {
    int r = 0;
    while ((value >> r) != 0)
        r++;
    return r-1;
}

inline int log2i(uint64_t value) {
    int r = 0;
    while ((value >> r) != 0)
        r++;
    return r-1;
}

inline uint32_t roundToPowerOfTwo(uint32_t i) {
    i--;
    i |= i >> 1; i |= i >> 2;
    i |= i >> 4; i |= i >> 8;
    i |= i >> 16;
    return i+1;
}

inline uint64_t roundToPowerOfTwo(uint64_t i) {
    i--;
    i |= i >> 1;  i |= i >> 2;
    i |= i >> 4;  i |= i >> 8;
    i |= i >> 16; i |= i >> 32;
    return i+1;
}

inline double fastlog(double value) {
    return std::log(value);
}

}// math


void latinHypercube(Random *random, Float *dest, size_t nSamples, size_t nDim);

enum ELogLevel {
    ETrace = 0,   ///< Trace message, for extremely verbose debugging
    EDebug = 100, ///< Debug message, usually turned off
    EInfo = 200,  ///< More relevant debug / information message
    EWarn = 300,  ///< Warning message
    EError = 400  ///< Error message, causes an exception to be thrown
};

#define SIZE_T_FMT "%zd"
#include <cstdio>
#define Log(level, ...) \
    printf(__VA_ARGS__)

#define Assert(cond) ((void)0)
#define SAssert(cond) ((void)0)

#define FINLINE                inline __attribute__((always_inline))

void * __restrict allocAligned(size_t size);
void freeAligned(void *ptr);


#endif
