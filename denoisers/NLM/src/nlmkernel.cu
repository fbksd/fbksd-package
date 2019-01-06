/*
 * File:   NlmeansKernel.cpp
 * Author: rousselle
 *
 * Created on March 19, 2012, 2:46 PM
 */

#include "nlmkernel.h"
#include <algorithm>
using std::fill;
#include <numeric>
using std::accumulate;
#include <cmath>
using std::exp;
#include "pbrt.h"
#include <iostream>
#include <bits/stl_vector.h>
using std::cout;
using std::endl;
#include <omp.h>
#include <bits/basic_string.h>

// The WGT_THRESHOLD value is used to round to zero small weights, which often
// correspond to neighbors with low confidence. A value of 0.05 works well in
// most cases, but a value of 0.00 (ie. no rounding) seems to work best when
// considering feature buffers.
const float WGT_THRESHOLD = 0.05f;

// Declaration of CUDA related stuff
#include <cuda.h>
static struct {
    float *_spp;
    float *_avg1, *_avg2;
    float *_var1, *_var2;
    float *_avgVar1, *_avgVar2;
    float *_avgVar, *_varVar;
    float *_d2avg1, *_d2avg2, *_d2avgS;
    // Misc
    float *_tmp;
    float *_wgt1, *_wgt2;
    float *_avgOut, *_sppOut, *_area;
} device;
// General
__global__ void conv_box_h(int width, int height, float * target, float const * source, int r);
__global__ void conv_box_v(int width, int height, float * target, float const * source, int r);
__global__ void sqr_diff_scale(int width, int height, float * target, float const * source1, float const * source2, float const scale, int nChannels);
__global__ void clamp_min(int width, int height, float * target, float const * source1, float const * source2, int nChannels);
__global__ void clamp_to_zero(int width, int height, float * target, float val, int nChannels);
__global__ void cumulate(int width, int height, float * target, float const * source);
__global__ void relax(int width, int height, float * tgt, float const * wgt, float const * src, int dx, int dy, int nChannels);
__global__ void normalize(int width, int height, float * target, float const * source, float const * area, int nChannels);
// Asymmetric
__global__ void weights(int width, int height, float * target, float const * d2, int dx, int dy);
// Symmetric
__global__ void distance(int width, int height, float * target1, float * target2, float * targetS, float const * src, float const * var, int dx1, int dy1, float scale, float gamma2, int nChannels);
__global__ void weights_sym(int width, int height, float * wgt1, float * wgt2, float const * d2avg1, float const * d2avg2, float const * d2avgS, int dx, int dy);
__global__ void relax_sym(int width, int height, float * tgt, float const * wgt1, float const * wgt2, float const * src, int dx, int dy, int nChannels);

inline
void CheckCudaError(cudaError_t error, const char* task = "") {
    switch (error) {
        case cudaSuccess:
            break;
        case cudaErrorInvalidValue:
            Severe("%s: cudaErrorInvalidValue", task);
            break;
        case cudaErrorInvalidDevicePointer:
            Severe("%s: cudaErrorInvalidDevicePointer", task);
            break;
        case cudaErrorInvalidMemcpyDirection:
            Severe("%s: cudaErrorInvalidDevicePointer", task);
            break;
        case cudaErrorMemoryAllocation:
            Severe("%s: cudaErrorMemoryAllocation", task);
            break;
        case cudaErrorInitializationError:
            Severe("%s: cudaErrorInitializationError", task);
            break;
        default:
            Severe("%s: cudaError unkwown", task);
            break;
    }
}


void NlmeansKernel::Init(int wnd_rad, int ptc_rad, float k, int xPixelCount, int yPixelCount) {
    // Params
    _wnd_rad = wnd_rad;
    _ptc_rad = ptc_rad;
    _k = k;

    // Image dims
    _xPixelCount = xPixelCount;
    _yPixelCount = yPixelCount;

    // Allocate memory on the device
    // Patch window radius
    int const size_img_bytes = _xPixelCount * _yPixelCount * sizeof(float);
    CheckCudaError(cudaMalloc((void **) &device._tmp,     1 * size_img_bytes), "alloc tmp");
    CheckCudaError(cudaMalloc((void **) &device._wgt1,    1 * size_img_bytes), "alloc wgt1");
    CheckCudaError(cudaMalloc((void **) &device._wgt2,    1 * size_img_bytes), "alloc wgt2");
    CheckCudaError(cudaMalloc((void **) &device._spp,     1 * size_img_bytes), "alloc spp");
    CheckCudaError(cudaMalloc((void **) &device._avg1,    3 * size_img_bytes), "alloc avg1");
    CheckCudaError(cudaMalloc((void **) &device._avg2,    3 * size_img_bytes), "alloc avg2");
    CheckCudaError(cudaMalloc((void **) &device._var1,    3 * size_img_bytes), "alloc var1");
    CheckCudaError(cudaMalloc((void **) &device._var2,    3 * size_img_bytes), "alloc var2");
    CheckCudaError(cudaMalloc((void **) &device._avgVar,  3 * size_img_bytes), "alloc avgVar");
    CheckCudaError(cudaMalloc((void **) &device._avgVar1, 3 * size_img_bytes), "alloc avgVar1");
    CheckCudaError(cudaMalloc((void **) &device._avgVar2, 3 * size_img_bytes), "alloc avgVar2");
    CheckCudaError(cudaMalloc((void **) &device._varVar,  3 * size_img_bytes), "alloc varVar");
    CheckCudaError(cudaMalloc((void **) &device._avgOut,  3 * size_img_bytes), "alloc avgOut");
    CheckCudaError(cudaMalloc((void **) &device._sppOut,  1 * size_img_bytes), "alloc sppOut");
    CheckCudaError(cudaMalloc((void **) &device._area,    1 * size_img_bytes), "alloc area");
    CheckCudaError(cudaMalloc((void **) &device._d2avg1,  1 * size_img_bytes), "alloc d2avg1");
    CheckCudaError(cudaMalloc((void **) &device._d2avg2,  1 * size_img_bytes), "alloc d2avg2");
    CheckCudaError(cudaMalloc((void **) &device._d2avgS,  1 * size_img_bytes), "alloc d2avgS");
}


void NlmeansKernel::Apply(NlmeansData dataType,
    const ImageBuffer &spp,
    const ImageBuffer &avg1, const ImageBuffer &var1, ImageBuffer &avgVar1,
    const ImageBuffer &avg2, const ImageBuffer &var2, ImageBuffer &avgVar2,
    ImageBuffer &avgOut1, ImageBuffer &sppOut1,
    ImageBuffer &avgOut2, ImageBuffer &sppOut2)
{
    int size_img_bytes = _xPixelCount * _yPixelCount * sizeof(float);

    dim3 block(16, 16, 1);
    dim3 grid((_xPixelCount + block.x - 1) / block.x, (_yPixelCount + block.y - 1) / block.y, 1);

    // Push SPP
    CheckCudaError(cudaMemcpy(device._spp, &spp[0], 1 * size_img_bytes, cudaMemcpyHostToDevice));

    // Push IMAGE data
    CheckCudaError(cudaMemcpy(device._avg1, &avg1[0], 3 * size_img_bytes, cudaMemcpyHostToDevice));
    CheckCudaError(cudaMemcpy(device._avg2, &avg2[0], 3 * size_img_bytes, cudaMemcpyHostToDevice));
    CheckCudaError(cudaMemcpy(device._var1, &var1[0], 3 * size_img_bytes, cudaMemcpyHostToDevice));
    CheckCudaError(cudaMemcpy(device._var2, &var2[0], 3 * size_img_bytes, cudaMemcpyHostToDevice));
    CheckCudaError(cudaMemcpy(device._avgVar1, &avgVar1[0], 3 * size_img_bytes, cudaMemcpyHostToDevice));
    CheckCudaError(cudaMemcpy(device._avgVar2, &avgVar2[0], 3 * size_img_bytes, cudaMemcpyHostToDevice));
    sqr_diff_scale<<<grid, block>>>(_xPixelCount, _yPixelCount, device._varVar, device._var1, device._var2, 1.f/2.f, 3);

    // Compute the buffer variance
    sqr_diff_scale<<<grid, block>>>(_xPixelCount, _yPixelCount, device._avgVar, device._avg1, device._avg2, 1.f/2.f, 3);

    // Get the filtered buffer mean variance
    ApplyIntVar(3, device._avgVar, device._var1, device._varVar, 1, 4.f);
    CheckCudaError(cudaMemcpy(device._avgVar, device._avgOut, 3 * size_img_bytes, cudaMemcpyDeviceToDevice));
    clamp_min<<<grid, block>>>(_xPixelCount, _yPixelCount, device._avgVar, device._avgVar1, device._avgVar2, 3);

    // Filter first buffer
#ifndef LD_SAMPLING
    CheckCudaError(cudaMemcpy(device._avgVar, &avgVar2[0], 3 * size_img_bytes, cudaMemcpyHostToDevice));
#endif
    ApplyInt(device._avg1, device._spp, device._avg2, device._avgVar,
        dataType, _wnd_rad, _k, 1.f);
    // Extract back the data
    CheckCudaError(cudaMemcpy(&avgOut1[0], device._avgOut, 3 * size_img_bytes, cudaMemcpyDeviceToHost));
    CheckCudaError(cudaMemcpy(&sppOut1[0], device._sppOut, 1 * size_img_bytes, cudaMemcpyDeviceToHost));
    
    // Filter second buffer
#ifndef LD_SAMPLING
    CheckCudaError(cudaMemcpy(device._avgVar, &avgVar1[0], 3 * size_img_bytes, cudaMemcpyHostToDevice));
#endif
    ApplyInt(device._avg2, device._spp, device._avg1, device._avgVar,
        dataType, _wnd_rad, _k, 1.f);
    
    // Extract back the data
    CheckCudaError(cudaMemcpy(&avgOut2[0], device._avgOut, 3 * size_img_bytes, cudaMemcpyDeviceToHost));
    CheckCudaError(cudaMemcpy(&sppOut2[0], device._sppOut, 1 * size_img_bytes, cudaMemcpyDeviceToHost));
}


void NlmeansKernel::ApplyIntVar(int nChannels, const float *avg1,
    const float *avg2, const float *avgVar2, int rad, float vScale) {

    int const size_img_bytes = _xPixelCount * _yPixelCount * sizeof(float);

    dim3 block(16, 16, 1);
    dim3 grid((_xPixelCount + block.x - 1) / block.x, (_yPixelCount + block.y - 1) / block.y, 1);

    // Initialize accumulators
    CheckCudaError(cudaMemset(device._area,   0, 1 * size_img_bytes));
    CheckCudaError(cudaMemset(device._avgOut, 0, nChannels * size_img_bytes));

    // Filter again, using patch-based filtering
    for (int dy = -rad; dy <= 0; ++ dy) {
        int dx_max = (dy == 0) ? -1 : +rad;
        for (int dx = -rad; dx <= dx_max; ++ dx) {
            // The relative inter-pixel mean squared distance
            distance  <<<grid, block>>>(_xPixelCount, _yPixelCount, device._d2avg1, device._d2avg2, device._d2avgS, avg2, avgVar2, dx, dy, vScale, sqr(_k), nChannels);
            conv_box_h<<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._d2avg1, _ptc_rad);
            conv_box_v<<<grid, block>>>(_xPixelCount, _yPixelCount, device._d2avg1, device._tmp, _ptc_rad);
            conv_box_h<<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._d2avg2, _ptc_rad);
            conv_box_v<<<grid, block>>>(_xPixelCount, _yPixelCount, device._d2avg2, device._tmp, _ptc_rad);
            conv_box_h<<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._d2avgS, _ptc_rad);
            conv_box_v<<<grid, block>>>(_xPixelCount, _yPixelCount, device._d2avgS, device._tmp, _ptc_rad);
            // Compute the weights
            weights_sym<<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt1, device._wgt2, device._d2avg1, device._d2avg2, device._d2avgS, dx, dy);
            conv_box_h <<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._wgt1, _ptc_rad);
            conv_box_v <<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt1, device._tmp, _ptc_rad);
            conv_box_h <<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._wgt2, _ptc_rad);
            conv_box_v <<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt2, device._tmp, _ptc_rad);
            clamp_to_zero<<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt1, WGT_THRESHOLD, 1); // ensure we only have reliable weights
            clamp_to_zero<<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt2, WGT_THRESHOLD, 1);
            cumulate  <<<grid, block>>>(_xPixelCount, _yPixelCount, device._area, device._wgt1);
            cumulate  <<<grid, block>>>(_xPixelCount, _yPixelCount, device._area, device._wgt2);
            relax_sym <<<grid, block>>>(_xPixelCount, _yPixelCount, device._avgOut, device._wgt1, device._wgt2, avg1, dx, dy, nChannels);
        }
    }

    // Add contribution of center pixel
    CheckCudaError(cudaMemset(device._d2avg1, 0, 1 * size_img_bytes));
    weights <<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt1, device._d2avg1, 0, 0);
    cumulate<<<grid, block>>>(_xPixelCount, _yPixelCount, device._area, device._wgt1);
    relax   <<<grid, block>>>(_xPixelCount, _yPixelCount, device._avgOut, device._wgt1, avg1, 0, 0, nChannels);

    normalize<<<grid, block>>>(_xPixelCount, _yPixelCount, device._avgOut, device._avgOut, device._area, nChannels);
}


void NlmeansKernel::ApplyInt(const float *avg1, const float *spp,
    const float *avg2, const float *avgVar,
    NlmeansData dataType, int wnd_rad, float gamma, float vScale) {

    int const size_img_bytes = _xPixelCount * _yPixelCount * sizeof(float);

    dim3 block(16, 16, 1);
    dim3 grid((_xPixelCount + block.x - 1) / block.x, (_yPixelCount + block.y - 1) / block.y, 1);

    vScale *= (dataType == NLM_DATA_FINAL) ? 1.f : .5f;

    // Initialize accumulators
    CheckCudaError(cudaMemset(device._area,   0, 1 * size_img_bytes));
    CheckCudaError(cudaMemset(device._sppOut, 0, 1 * size_img_bytes));
    CheckCudaError(cudaMemset(device._avgOut, 0, 3 * size_img_bytes));

    wnd_rad = (dataType == NLM_DATA_FINAL) ? wnd_rad : 7;
    // Filter again, using patch-based filtering
    for (int dy = -wnd_rad; dy <= 0; ++ dy) {
        int dx_max = (dy == 0) ? -1 : +wnd_rad;
        for (int dx = -wnd_rad; dx <= dx_max; ++ dx) {
            // The relative inter-pixel mean squared distance
            distance  <<<grid, block>>>(_xPixelCount, _yPixelCount, device._d2avg1, device._d2avg2, device._d2avgS, avg2, avgVar, dx, dy, vScale * 1.f, sqr(gamma), 3);
            conv_box_h<<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._d2avg1, _ptc_rad);
            conv_box_v<<<grid, block>>>(_xPixelCount, _yPixelCount, device._d2avg1, device._tmp, _ptc_rad);
            conv_box_h<<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._d2avg2, _ptc_rad);
            conv_box_v<<<grid, block>>>(_xPixelCount, _yPixelCount, device._d2avg2, device._tmp, _ptc_rad);
            conv_box_h<<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._d2avgS, _ptc_rad);
            conv_box_v<<<grid, block>>>(_xPixelCount, _yPixelCount, device._d2avgS, device._tmp, _ptc_rad);
            // Compute the weights
            weights_sym<<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt1, device._wgt2, device._d2avg1, device._d2avg2, device._d2avgS, dx, dy);
            conv_box_h <<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._wgt1, _ptc_rad);
            conv_box_v <<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt1, device._tmp, _ptc_rad);
            conv_box_h <<<grid, block>>>(_xPixelCount, _yPixelCount, device._tmp, device._wgt2, _ptc_rad);
            conv_box_v <<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt2, device._tmp, _ptc_rad);
            clamp_to_zero<<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt1, WGT_THRESHOLD, 1); // ensure we only have reliable weights
            clamp_to_zero<<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt2, WGT_THRESHOLD, 1);
            cumulate  <<<grid, block>>>(_xPixelCount, _yPixelCount, device._area, device._wgt1);
            cumulate  <<<grid, block>>>(_xPixelCount, _yPixelCount, device._area, device._wgt2);
            relax_sym <<<grid, block>>>(_xPixelCount, _yPixelCount, device._sppOut, device._wgt1, device._wgt2, spp,  dx, dy, 1);
            relax_sym <<<grid, block>>>(_xPixelCount, _yPixelCount, device._avgOut, device._wgt1, device._wgt2, avg1, dx, dy, 3);
        }
    }
    // Add contribution of center pixel
    CheckCudaError(cudaMemset(device._d2avg1, 0, 1 * size_img_bytes));
    weights <<<grid, block>>>(_xPixelCount, _yPixelCount, device._wgt1, device._d2avg1, 0, 0);
    cumulate<<<grid, block>>>(_xPixelCount, _yPixelCount, device._area, device._wgt1);
    relax   <<<grid, block>>>(_xPixelCount, _yPixelCount, device._sppOut, device._wgt1, spp,  0, 0, 1);
    relax   <<<grid, block>>>(_xPixelCount, _yPixelCount, device._avgOut, device._wgt1, avg1, 0, 0, 3);
    
    normalize<<<grid, block>>>(_xPixelCount, _yPixelCount, device._sppOut, device._sppOut, device._area, 1);
    normalize<<<grid, block>>>(_xPixelCount, _yPixelCount, device._avgOut, device._avgOut, device._area, 3);
}


#define CLAMP_MIRROR(pos, pos_max) \
    ((pos) < 0) ? -(pos) : ((pos) >= (pos_max)) ? 2*(pos_max) - (pos) - 2 : (pos)
#define CALCULATE_INDEX(width, height) \
    int x = blockIdx.x * blockDim.x + threadIdx.x; \
    int y = blockIdx.y * blockDim.y + threadIdx.y; \
    if (x >= (width) || y >= (height)) return; \
    int index = (y) * (width) + x
#define CALCULATE_INDICES(width, height, dx, dy) \
    int x = blockIdx.x * blockDim.x + threadIdx.x; \
    int y = blockIdx.y * blockDim.y + threadIdx.y; \
    if (x >= (width) || y >= (height)) return; \
    int indexC = y * (width) + x; \
    int x1 = CLAMP_MIRROR(x+(dx), (width)); \
    int y1 = CLAMP_MIRROR(y+(dy), (height)); \
    int indexN = x1 + y1 * (width)
#define CALCULATE_INDICES_SYM(width, height, dx, dy) \
    int x = blockIdx.x * blockDim.x + threadIdx.x; \
    int y = blockIdx.y * blockDim.y + threadIdx.y; \
    if (x >= (width) || y >= (height)) return; \
    int indexC = y * (width) + x; \
    int x1 = CLAMP_MIRROR(x+(dx), (width)); \
    int y1 = CLAMP_MIRROR(y+(dy), (height)); \
    int indexN1 = x1 + y1 * (width); \
    int x2 = CLAMP_MIRROR(x-(dx), (width)); \
    int y2 = CLAMP_MIRROR(y-(dy), (height)); \
    int indexN2 = x2 + y2 * (width)

__global__ void distance(int width, int height, float * tgt1, float * tgt2, float * tgtS, float const * src, float const * var, int dx, int dy, float scale, float k2, int nChannels)
{
    CALCULATE_INDICES_SYM(width, height, dx, dy);

    float d, d2;
    tgt1[indexC] = 0;
    tgt2[indexC] = 0;
    tgtS[indexC] = 0;
    for (int c = 0; c < nChannels; c++) {
        // The center and neighbor channel indices
        int idxC  = nChannels * indexC  + c;
        int idxN1 = nChannels * indexN1 + c;
        int idxN2 = nChannels * indexN2 + c;

        // FIRST NEIGHBOR
        float varN1 = min(var[idxC], var[idxN1]);
        // Compute squared difference
        d = src[idxC] - src[idxN1];
        d2 = d * d - scale * (var[idxC] + varN1);
        d2 /= 1e-10f + k2 * (var[idxC] + var[idxN1]);
        tgt1[indexC] += d2;
        // SECOND NEIGHBOR
        float varN2 = min(var[idxC], var[idxN2]);
        // Compute squared difference
        d = src[idxC] - src[idxN2];
        d2 = d * d - scale * (var[idxC] + varN2);
        d2 /= 1e-10f + k2 * (var[idxC] + var[idxN2]);
        tgt2[indexC] += d2;

        // SYMMETRIC NEIGHBORS
        // Compute squared difference
        d = src[idxC] - (src[idxN1]+src[idxN2])/2.f;
        d2 = d * d - scale * (var[idxC] + (varN1+varN2)/4.f);
        d2 /= 1e-10f + k2 * (var[idxC] + (var[idxN1] + var[idxN2]) / 4.f);

        tgtS[indexC] += d2;
    }

    tgt1[indexC] /= nChannels;
    tgt2[indexC] /= nChannels;
    tgtS[indexC] /= nChannels;
}

__global__ void conv_box_h(int width, int height, float * target, float const * source, int r)
{
    CALCULATE_INDEX(width, height);

    int r1 = min(r, x);
    int r2 = min(r, width-1-x);

    int l = 2 * r + 1;
    float acc = 0;
    for (int j = -r1; j <= r2; ++ j)
    {
        acc += source[index + j];
    }
    target[index] = acc / l;
}

__global__ void conv_box_v(int width, int height, float * target, float const * source, int r)
{
    CALCULATE_INDEX(width, height);

    int r1 = min(r, y);
    int r2 = min(r, height-1-y);

    int l = (r1+r2) + 1;
    float acc = 0;
    for (int j = -r1; j <= r2; ++ j)
    {
        acc += source[index + j * width];
    }
    target[index] = acc / l;
}

__global__ void weights(int width, int height, float * wgt, float const * d2, int dx, int dy)
{
    CALCULATE_INDEX(width, height);

    wgt[index] = exp(- max(d2[index], 0.f));
}

__global__ void weights_sym(int width, int height, float * wgt1, float * wgt2, float const * d2avg1, float const * d2avg2, float const * d2avgS, int dx, int dy)
{
    CALCULATE_INDEX(width, height);

    float w1 = exp(- max(0.f, d2avg1[index]));
    float w2 = exp(- max(0.f, d2avg2[index]));
    float wS = exp(- max(0.f, d2avgS[index]));

    float wA = w1 + w2;
    float r = min(1.f, max(0.f, wS / wA - 1));
    wgt1[index] = r * wS + (1-r) * w1;
    wgt2[index] = r * wS + (1-r) * w2;
}

__global__ void cumulate(int width, int height, float * target, float const * source)
{
    CALCULATE_INDEX(width, height);

    target[index] += source[index];
}

__global__ void relax(int width, int height, float * tgt, float const * wgt, float const * src, int dx, int dy, int nChannels)
{
    CALCULATE_INDICES(width, height, dx, dy);

    for (int c = 0; c < nChannels; c++)
        tgt[nChannels*indexC+c] += wgt[indexC] * src[nChannels*indexN+c];
}

__global__ void relax_sym(int width, int height, float * tgt, float const * wgt1, float const * wgt2, float const * src, int dx, int dy, int nChannels)
{
    CALCULATE_INDICES_SYM(width, height, dx, dy);

    for (int c = 0; c < nChannels; c++) {
        tgt[nChannels*indexC+c] += wgt1[indexC] * src[nChannels*indexN1+c];
        tgt[nChannels*indexC+c] += wgt2[indexC] * src[nChannels*indexN2+c];
    }
}

__global__ void normalize(int width, int height, float * target, float const * source, float const * area, int nChannels)
{
    CALCULATE_INDEX(width, height);

    for (int c = 0; c < nChannels; c++) {
        target[nChannels*index+c] = source[nChannels*index+c] / area[index];
    }
}


__global__ void sqr_diff_scale(int width, int height, float * target, float const * source1, float const * source2, float const scale, int nChannels)
{
    CALCULATE_INDEX(width, height);

    for (int c = 0; c < nChannels; c++) {
        float d = source1[nChannels*index+c] - source2[nChannels*index+c];
        target[nChannels*index+c] = d * d * scale;
    }
}


__global__ void clamp_min(int width, int height, float * target, float const * source1, float const * source2, int nChannels)
{
    CALCULATE_INDEX(width, height);

    for (int c = 0; c < nChannels; c++) {
        target[nChannels*index+c] = min(target[nChannels*index+c], max(source1[nChannels*index+c], source2[nChannels*index+c]));
    }
}

__global__ void clamp_to_zero(int width, int height, float * target, float val, int nChannels)
{
    CALCULATE_INDEX(width, height);

    for (int c = 0; c < nChannels; c++) {
        if (target[nChannels*index+c] < val)
            target[nChannels*index+c] = 0.f;
    }
}

