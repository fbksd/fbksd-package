/*
 *  Copyright(c) 2011 Fabrice Rousselle.
 *
 *  You can redistribute and/or modify this file under the terms of the GNU
 *  General Public License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 */

#ifndef KERNEL2D_H
#define	KERNEL2D_H


#include <numeric>
using std::accumulate;
#include <algorithm>
using std::max_element;
#include "pbrt.h"
#include "filter.h"


enum KernelNorm {
    KERNEL_NORM_STD,
    KERNEL_NORM_UNIT
};


enum KernelSkip {
    KERNEL_SKIP_NONE,
    KERNEL_SKIP_CENTER
};


class Kernel2D {
public:
    // Methods
    Kernel2D() {}
    Kernel2D(const Filter *filter, KernelNorm norm, int resSub = 1);

    void ConvolveWith(const Filter *filter, KernelNorm norm);

    void Apply(int xPixelCount, int yPixelCount, const vector<float> &in,
        vector<float> &tmp, vector<float> &out,
        KernelSkip skip = KERNEL_SKIP_NONE) const {
        int nPix = xPixelCount * yPixelCount, nChannels = out.size() / nPix;
        if (nChannels == 1)
            Apply1C(xPixelCount, yPixelCount, in, tmp, out, skip);
        else
            Apply3C(xPixelCount, yPixelCount, in, tmp, out, skip);
    }
    void ApplySub(int xPixelCount, int yPixelCount, const vector<float> &in,
        vector<float> &tmp, vector<float> &out) const {
        int nPix = xPixelCount * yPixelCount, nChannels = out.size() / nPix;
        if (nChannels == 1)
            ApplySub1C(xPixelCount, yPixelCount, in, tmp, out);
        else
            ApplySub3C(xPixelCount, yPixelCount, in, tmp, out);
    }
    static Kernel2D * Sub2Var(const Kernel2D &sub) {
        Kernel2D *kernel = new Kernel2D();
        kernel->_resSub = 1;
        kernel->_kNorm = Kernel2D::Sub2Var(sub._k, sub._resSub, kernel->_k);

        kernel->BuildCDF();
        return kernel;
    }
    static Kernel2D * Sub2Pix(const Kernel2D &sub, KernelNorm norm) {
        Kernel2D *kernel = new Kernel2D();
        kernel->_resSub = 1;
        kernel->_kNorm = Kernel2D::Sub2Pix(sub._k, sub._resSub, kernel->_k);

        kernel->Normalize(norm);

        kernel->BuildCDF();
        return kernel;
    }
    float Norm() const { return _kNorm * _kNorm; }
    float SqrNorm() const { return _kNorm * _kNorm * _kNorm * _kNorm; }
    void WarpSampleToPixelOffset(float &x, float &y) const {
        // We assume x and y are in the range [0, 1)
        Warp(x, _cdf);
        Warp(y, _cdf);
    }

protected:
    // Attributes
    float _kNorm;       // the kernel norm
    vector<float> _k;   // the actual (1d) kernel
    // The subpixel grid resolution (1 if working on pixel grid)
    int _resSub;
    // CDF
    vector<float> _cdf;

    // Methods
    // Pixel resolution
    void Apply1C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out, KernelSkip skip) const;
    void Apply3C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out, KernelSkip skip) const;
    // Subpixel resolution with implicit downsampling
    void ApplySub1C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out) const;
    void ApplySub3C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out) const;
    //
    static float Sub2Var(const vector<float> &subKernel, int resSub, vector<float> &varKernel)  {
        // Define a variance filter matching the subpixel filter. Since the
        // variance is uniform for all subpixels of a given pixel, we simply
        // accumulate the squared weights over each tap at the pixel resolution.
        int pad = Ceil2Int(float(subKernel.size()/2-(resSub/2))/resSub);
        int nTaps = 2 * pad + 1;
        varKernel.clear(); varKernel.resize(nTaps, 0.f);
        // Accumulate all squared subpixel filter entries for each pixel.
        // Since the subpixel filter is not necessarily aligned with the pixel
        // filter, we compute the offset between the two.
        int offset = (resSub * nTaps - subKernel.size()) / 2;
        for (uint32_t i = 0; i < subKernel.size(); i++) {
            int idx = (i+offset) / resSub;
            float sqr_val = subKernel[i] * subKernel[i];
            varKernel[idx] += sqr_val * resSub;
        }
        return accumulate(varKernel.begin(), varKernel.end(), 0.f);
    }
    static float Sub2Pix(const vector<float> &subKernel, int resSub, vector<float> &pixKernel)  {
        // Simply accumulate the weights over each tap at the pixel resolution.
        int pad = Ceil2Int(float(subKernel.size()/2-(resSub/2))/resSub);
        int nTaps = 2 * pad + 1;
        pixKernel.clear(); pixKernel.resize(nTaps, 0.f);
        // Accumulate all subpixel filter entries for each pixel.
        // Since the subpixel filter is not necessarily aligned with the pixel
        // filter, we compute the offset between the two.
        int offset = (resSub * nTaps - subKernel.size()) / 2;
        for (uint32_t i = 0; i < subKernel.size(); i++) {
            int idx = (i+offset) / resSub;
            pixKernel[idx] += subKernel[i];
        }
        return accumulate(pixKernel.begin(), pixKernel.end(), 0.f);
    }
    float Convolve(const vector<float>& in1, const vector<float>& in2,
        vector<float>& out);

    void Normalize(KernelNorm norm) {
        if (norm == KERNEL_NORM_UNIT) {
            float sum = accumulate(_k.begin(), _k.end(), 0.f);
            for (uint32_t i = 0; i < _k.size(); i++) _k[i] /= sum;
        }
        else {
            float mval = *max_element(_k.begin(), _k.end());
            for (uint32_t i = 0; i < _k.size(); i++) _k[i] /= mval;
        }
        _kNorm = accumulate(_k.begin(), _k.end(), 0.f);
    }

    // Importance Sampling related
    void BuildCDF();
    void Warp(float &rnd, const vector<float> &cdf) const;
};


class Gauss2D : public Kernel2D {
public:
    Gauss2D(float sigma, KernelNorm norm, int resSub = 1) {
        Init(sigma, norm, resSub);
    }

private:
    void Init(float sigma, KernelNorm norm, int resSub) {
        _resSub = resSub;

        _k.clear();

        // Set kernel parameter
        int pad = Floor2Int(3.f * sigma);
        int nTaps = _resSub * (1+2*pad);
        float alpha = -.5f/sigma/sigma;
        // Set filter values
        _k.resize(nTaps);
        float offset = _resSub % 2 == 0 ? .5f : 0.f;
        for (int i = 0; i < nTaps; i++) {
            float dist = (offset + i - nTaps/2) / _resSub;
            float val = expf(alpha*dist*dist);
            _k[i] = val;
        }

        Normalize(norm);
        BuildCDF();
    }
};


#endif	/* KERNEL2D_H */

