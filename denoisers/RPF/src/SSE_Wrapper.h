#ifndef	SSE_WRAPPER_H_INCLUDED
#define SSE_WRAPPER_H_INCLUDED

#include <stdio.h>
#include <xmmintrin.h>	// Need this for SSE compiler intrinsics
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include "Globals.h"

#if ENABLE_SSE

// Typedef used in SSE_Math.h 
#ifdef _WINDOWS
typedef __m128 v4sf;  // Vector of 4 float (sse1)
#else
typedef union V4sf{
    V4sf(__m128 v = _mm_set1_ps(0.f)) :
        v(v)
    {}

    operator __m128()
    { return v; }

    float m128_f32[4];  // scalar array of 4 floats

private:
    __m128 v;           // SSE 4 x float vector
} v4sf;
#endif

// Defined in SSE_Math.h 
v4sf log_ps(v4sf x);

// Use SSE to calculate mean of first length elements of source
float calculateMeanSSE(float* src, size_t length);

// Use SSE to calculate std of first length elements of source
float calculateStdSSE(float* src, float mean, size_t length);

// Use SSE to normalize first length elements of source and save them in
// result. Normalize by subtracting mean and dividing by the std
void normalizeSSE(float* src, float mean, float std, float* result, size_t length);

// Use SSE to subtract first length elemnts of data2 from data1 then square 
// and multiply by alpha. The result is then saved
void subtractSquareScaleSSE(float* data1, float* data2, float* result, float alpha, size_t length);

// Use SSE to do element-wise addition between first length elements of data1 
// and data2 and store in result.
void sumSSE(float* data1, float* data2, float* result, size_t length);

// Use SSE to multiply the first length elements of data by alpha and store
// in result
void multByValSSE(float* data, float* result, float alpha, size_t length);

// Use SSE to element-wise multiply the first length elements of data1 and
// data2 and store in result
void multSSE(float* data1, float* data2, float* result, size_t length);

// Use SSE to add up the first length elements of data
float sumAllElementsSSE(float* data, size_t length);

#endif

#endif
