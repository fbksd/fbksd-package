/*
 *  Copyright(c) 2011 Fabrice Rousselle.
 *
 *  You can redistribute and/or modify this file under the terms of the GNU
 *  General Public License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 */


#include "kernel2d.h"
#include <limits>
#include <thread>


Kernel2D::Kernel2D(const Filter* filter, KernelNorm norm, int resSub) {
    _resSub = resSub;

    // We assume that the input filter is separable and isotropic!
    float width = filter->xWidth;
    int pad = Floor2Int(width);
    int nTaps = _resSub * (1+2*pad);

    // Set X taps
    _k.resize(nTaps);
    float offset = _resSub % 2 == 0 ? .5f : 0.f;
    for (int i = 0; i < nTaps; i++) {
        float dist = (offset + i - nTaps/2) / _resSub;
        float val = filter->Evaluate(dist, 0.f);
        _k[i] = val;
    }

    Normalize(norm);

    BuildCDF();
}


void Kernel2D::ConvolveWith(const Filter *filter, KernelNorm norm) {
    // We assume that the input filter is separable and isotropic!
    float width = filter->xWidth;
    int pad = Floor2Int(_resSub * width);

    // Tabulate and normalize the filter
    vector<float> ker(1+2*pad);
    for (int tap = -pad; tap <= pad; tap++) {
        float dist = float(tap) / _resSub;
        float val = filter->Evaluate(dist, 0.f);
        ker[tap+pad] = val;
    }

    // Convolve with local
    vector<float> tmp;
    Convolve(_k, ker, tmp); swap(_k, tmp);
    Normalize(norm);

    BuildCDF();
}


float Kernel2D::Convolve(const vector<float>& in1, const vector<float>& in2,
    vector<float>& out) {
    out.clear();
    out.resize(in1.size() + in2.size() - 1, 0.f);

    for (uint32_t i = 0; i < in1.size(); i++) {
        for (uint32_t j = 0; j < in2.size(); j++) {
            out[i+j] += in1[i] * in2[j];
        }
    }

    return std::accumulate(out.begin(), out.end(), 0.f);
}


void Kernel2D::Apply1C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out,
    KernelSkip skip) const {

    int nPix = xPixelCount * yPixelCount;
    int nTaps = _k.size();
    int pad = nTaps/2;

    // Horizontal filtering
#pragma omp parallel for num_threads(std::thread::hardware_concurrency())
    for (int pix = 0; pix < nPix; pix++) {
        int y = pix / xPixelCount, x = pix - y * xPixelCount;
        tmp[pix] = 0.f;
        // Restrict the window
        float tmin = max(0, pad-x), tmax = min(nTaps, xPixelCount+pad-x);
        // Do the filtering
        float sum = 0.f;
        for (int tap = tmin; tap < tmax; tap++) {
            int spix = pix+tap-pad;
            tmp[pix] += _k[tap] * in[spix];
            sum += _k[tap];
        }
        tmp[pix] *= _kNorm/sum;
    }

    // Vertical filtering
#pragma omp parallel for num_threads(std::thread::hardware_concurrency())
    for (int pix = 0; pix < nPix; pix++) {
        int y = pix / xPixelCount;
        out[pix] = 0.f;
        // Restrict the window
        float tmin = max(0, pad-y), tmax = min(nTaps, yPixelCount+pad-y);
        // Do the filtering
        float sum = 0.f;
        for (int tap = tmin; tap < tmax; tap++) {
            int spix = pix + (tap-pad) * xPixelCount;
            out[pix] += _k[tap] * tmp[spix];
            sum += _k[tap];
        }
        out[pix] *= _kNorm/sum;

        // decrease contribution of center pixel to, at most, th%, and reweight the rest
        if (skip == KERNEL_SKIP_CENTER) {
            float norm = _kNorm*_kNorm;
            float th = 0.01f; // small residual contribution to tip the balance if needed
            float cwt = _k[pad] * _k[pad], diff = max(0.f, cwt-th);
            float scale = norm / (norm - diff);
            out[pix] = (out[pix]-diff*in[pix]) * scale;

        }
    }
}


void Kernel2D::Apply3C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out,
    KernelSkip skip) const {

    int nPix = xPixelCount * yPixelCount;
    int nTaps = _k.size();
    int pad = nTaps/2;

    // Horizontal filtering
#pragma omp parallel for num_threads(std::thread::hardware_concurrency())
    for (int pix = 0; pix < nPix; pix++) {
        int y = pix / xPixelCount, x = pix - y * xPixelCount;
        tmp[3*pix+0] = tmp[3*pix+1] = tmp[3*pix+2] = 0.f;
        // Restrict the window
        float tmin = max(0, pad-x), tmax = min(nTaps, xPixelCount+pad-x);
        // Do the filtering
        float sum = 0.f;
        for (int tap = tmin; tap < tmax; tap++) {
            int spix = pix+tap-pad;
            tmp[3*pix+0] += _k[tap] * in[3*spix+0];
            tmp[3*pix+1] += _k[tap] * in[3*spix+1];
            tmp[3*pix+2] += _k[tap] * in[3*spix+2];
            sum += _k[tap];
        }
        tmp[3*pix+0] *= _kNorm/sum;
        tmp[3*pix+1] *= _kNorm/sum;
        tmp[3*pix+2] *= _kNorm/sum;
    }

    // Vertical filtering
#pragma omp parallel for num_threads(std::thread::hardware_concurrency())
    for (int pix = 0; pix < nPix; pix++) {
        int y = pix / xPixelCount;
        out[3*pix+0] = out[3*pix+1] = out[3*pix+2] = 0.f;
        // Restrict the window
        float tmin = max(0, pad-y), tmax = min(nTaps, yPixelCount+pad-y);
        // Do the filtering
        float sum = 0.f;
        for (int tap = tmin; tap < tmax; tap++) {
            int spix = pix + (tap-pad) * xPixelCount;
            out[3*pix+0] += _k[tap] * tmp[3*spix+0];
            out[3*pix+1] += _k[tap] * tmp[3*spix+1];
            out[3*pix+2] += _k[tap] * tmp[3*spix+2];
            sum += _k[tap];
        }
        out[3*pix+0] *= _kNorm/sum;
        out[3*pix+1] *= _kNorm/sum;
        out[3*pix+2] *= _kNorm/sum;

        // decrease contribution of center pixel to, at most, th%, and reweight the rest
        if (skip == KERNEL_SKIP_CENTER) {
            float norm = _kNorm*_kNorm;
            float th = 0.01f; // small residual contribution to tip the balance if needed
            float cwt = _k[pad] * _k[pad], diff = max(0.f, cwt-th);
            float scale = norm / (norm - diff);
            out[3*pix+0] = (out[3*pix+0]-diff*in[3*pix+0]) * scale;
            out[3*pix+1] = (out[3*pix+1]-diff*in[3*pix+1]) * scale;
            out[3*pix+2] = (out[3*pix+2]-diff*in[3*pix+2]) * scale;

        }
    }
}


void Kernel2D::ApplySub1C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out) const {

    int nPix = xPixelCount * yPixelCount;
    int xSubPixelCount = xPixelCount * _resSub;
    int ySubPixelCount = yPixelCount * _resSub;
    int nTaps = int(_k.size());
    int pad = nTaps/2;

    // Horizontal filtering
    int nTmpPix = xPixelCount * ySubPixelCount;
#pragma omp parallel for num_threads(std::thread::hardware_concurrency())
    for (int tmpPix = 0; tmpPix < nTmpPix; tmpPix++) {
        int y = tmpPix / xPixelCount, x = tmpPix - y * xPixelCount;
        x = _resSub*x + _resSub/2;
        tmp[tmpPix] = 0.f;
        // Restrict the window
        float tmin = max(0, pad-x), tmax = min(nTaps, xSubPixelCount+pad-x);
        // Do the filtering
        float sum = 0.f;
        int subPix = x + y * xSubPixelCount;
        for (int tap = tmin; tap < tmax; tap++) {
            int subOffset = tap-pad;
            int spix = subPix+subOffset;
            tmp[tmpPix] += _k[tap] * in[spix];
            sum += _k[tap];
        }
        tmp[tmpPix] *= _kNorm/sum;
    }

    // Vertical filtering
#pragma omp parallel for num_threads(std::thread::hardware_concurrency())
    for (int pix = 0; pix < nPix; pix++) {
        int y = pix / xPixelCount, x = pix - y * xPixelCount;
        y = _resSub*y + _resSub/2;
        out[pix] = 0.f;
        // Restrict the window
        float tmin = max(0, pad -y), tmax = min(nTaps, ySubPixelCount+pad -y);
        // Do the filtering
        float sum = 0.f;
        int tmpPix = x + y * xPixelCount;
        for (int tap = tmin; tap < tmax; tap++) {
            int subOffset = (tap-pad ) * xPixelCount;
            int spix = tmpPix+subOffset;
            out[pix] += _k[tap] * tmp[spix];
            sum += _k[tap];
        }
        out[pix] *= _kNorm/sum;
    }
}


void Kernel2D::ApplySub3C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out) const {

    int nPix = xPixelCount * yPixelCount;
    int xSubPixelCount = xPixelCount * _resSub;
    int ySubPixelCount = yPixelCount * _resSub;
    int nTaps = int(_k.size());
    int pad = nTaps/2;

    // Horizontal filtering
    int nTmpPix = xPixelCount * ySubPixelCount;
#pragma omp parallel for num_threads(std::thread::hardware_concurrency())
    for (int tmpPix = 0; tmpPix < nTmpPix; tmpPix++) {
        int y = tmpPix / xPixelCount, x = tmpPix - y * xPixelCount;
        x = _resSub*x + _resSub/2;
        tmp[3*tmpPix+0] = tmp[3*tmpPix+1] = tmp[3*tmpPix+2] = 0.f;
        // Restrict the window
        float tmin = max(0, pad-x), tmax = min(nTaps, xSubPixelCount+pad-x);
        // Do the filtering
        float sum = 0.f;
        int subPix = x + y * xSubPixelCount;
        for (int tap = tmin; tap < tmax; tap++) {
            int subOffset = tap-pad;
            int spix = subPix+subOffset;
            tmp[3*tmpPix+0] += _k[tap] * in[3*spix+0];
            tmp[3*tmpPix+1] += _k[tap] * in[3*spix+1];
            tmp[3*tmpPix+2] += _k[tap] * in[3*spix+2];
            sum += _k[tap];
        }
        tmp[3*tmpPix+0] *= _kNorm/sum;
        tmp[3*tmpPix+1] *= _kNorm/sum;
        tmp[3*tmpPix+2] *= _kNorm/sum;
    }

    // Vertical filtering
#pragma omp parallel for num_threads(std::thread::hardware_concurrency())
    for (int pix = 0; pix < nPix; pix++) {
        int y = pix / xPixelCount, x = pix - y * xPixelCount;
        y = _resSub*y + _resSub/2;
        out[3*pix+0] = out[3*pix+1] = out[3*pix+2] = 0.f;
        // Restrict the window
        float tmin = max(0, pad -y), tmax = min(nTaps, ySubPixelCount+pad -y);
        // Do the filtering
        float sum = 0.f;
        int tmpPix = x + y * xPixelCount;
        for (int tap = tmin; tap < tmax; tap++) {
            int subOffset = (tap-pad ) * xPixelCount;
            int spix = tmpPix+subOffset;
            out[3*pix+0] += _k[tap] * tmp[3*spix+0];
            out[3*pix+1] += _k[tap] * tmp[3*spix+1];
            out[3*pix+2] += _k[tap] * tmp[3*spix+2];
            sum += _k[tap];
        }
        out[3*pix+0] *= _kNorm/sum;
        out[3*pix+1] *= _kNorm/sum;
        out[3*pix+2] *= _kNorm/sum;
    }
}


void Kernel2D::BuildCDF() {
    // Accumulate the weights
    _cdf.resize(_k.size());
    _cdf[0] = fabs(_k[0]);
    for (uint32_t i = 1; i < _cdf.size(); i++)
        _cdf[i] = fabs(_k[i]) + _cdf[i-1];
    // Normalize
    for(uint32_t i = 0; i < _cdf.size(); i++)
        _cdf[i] /= _cdf.back();
}


void Kernel2D::Warp(float &rnd, const vector<float> &cdf) const {
    float offset =  -float(cdf.size()/2) / _resSub;
    if (cdf.size() % 2 == 0) offset += 1.f / (2 * _resSub);

    float prev = 0.f;
    for (uint32_t i = 0; i < cdf.size(); i++) {
        if (rnd <= cdf[i]) {
            // Skip taps with 0 weight
            float diff = cdf[i] - prev;
            if (diff == 0.f) continue;
            // Check were we fell
            float ratio = (rnd - prev) / diff;
            // Warp the sample
            rnd = offset + i + ratio;
            return;
        }
        prev = cdf[i];
    }
    // We should never reach this point
    return;
}



