
/*
 * File:   kernel2d.cpp
 * Author: rousselle
 *
 * Created on February 21, 2011, 10:05 AM
 */

#ifndef KERNEL2D_H
#define	KERNEL2D_H


#include <numeric>
using std::accumulate;
#include <algorithm>
using std::max_element;
#include <limits>
using std::numeric_limits;
#include "pbrt.h"
#include "filter.h"


enum KernelNorm {
    KERNEL_NORM_STD,
    KERNEL_NORM_UNIT,
    KERNEL_NORM_NONE
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

    void Apply(int xPixelCount, int yPixelCount, const vector<float> &in,
        vector<float> &tmp, vector<float> &out,
        KernelSkip skip = KERNEL_SKIP_NONE) const {
        int nPix = xPixelCount * yPixelCount, nChannels = out.size() / nPix;
        if (_isSeparable) {
            if (nChannels == 1)
                ApplySep1C(xPixelCount, yPixelCount, in, tmp, out, skip);
            else
                ApplySep3C(xPixelCount, yPixelCount, in, tmp, out, skip);
        }
        else {
            if (nChannels == 1)
                ApplyNonSep1C(xPixelCount, yPixelCount, in, tmp, out, skip);
            else
                ApplyNonSep3C(xPixelCount, yPixelCount, in, tmp, out, skip);
        }
    }
    void ApplySub(int xPixelCount, int yPixelCount, const vector<float> &in,
        vector<float> &tmp, vector<float> &out) const {
        int nPix = xPixelCount * yPixelCount, nChannels = out.size() / nPix;
        if (_isSeparable) {
            if (nChannels == 1)
                ApplySepSub1C(xPixelCount, yPixelCount, in, tmp, out);
            else
                ApplySepSub3C(xPixelCount, yPixelCount, in, tmp, out);
        }
        else {
            if (nChannels == 1)
                ApplyNonSepSub1C(xPixelCount, yPixelCount, in, tmp, out);
            else
                ApplyNonSepSub3C(xPixelCount, yPixelCount, in, tmp, out);
        }
    }
    static Kernel2D * Sub2Var(const Kernel2D &sub) {
        Kernel2D *kernel = new Kernel2D();
        kernel->_isSeparable = sub._isSeparable;
        if (sub._isSeparable) {
            kernel->_resSub = 1;
            kernel->_kNorm = Kernel2D::Sub2Var1D(sub._k, sub._resSub, kernel->_k);
            kernel->_k2Norm = Kernel2D::Sub2Var1D(sub._k2, sub._resSub, kernel->_k2);
            kernel->_length = kernel->_k.size();
        }
        else {
            kernel->_resSub = 1;
            kernel->_kNorm = Kernel2D::Sub2Var2D(sub._length, sub._k, sub._resSub, kernel->_k, kernel->_length);
            kernel->_k2Norm = 1.f; kernel->_k2.clear();
        }

        return kernel;
    }

protected:
    // Attributes
    // If (_k2.empty()), the kernel is non-separable and '_length' corresponds
    // to the xPixelCount of the kernel. The yPixelCount is given by _k.size() / _length.
    int _length;
    // For non-separable kernels, _k2 is empty.
    float _kNorm, _k2Norm;
    vector<float> _k, _k2; // _k2 only defined if kernel is separable
    // A kernel can be defined over a subpixel grid, applying it performs
    // implicit downsampling.
    int _resSub;
    //
    bool _isSeparable;

    // Methods
    // Pixel resolution
    void ApplySep1C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out, KernelSkip skip) const;
    void ApplySep3C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out, KernelSkip skip) const;
    void ApplyNonSep1C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out, KernelSkip skip) const;
    void ApplyNonSep3C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out, KernelSkip skip) const;
    // Subpixel resolution with implicit downsampling
    void ApplySepSub1C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out) const;
    void ApplySepSub3C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out) const;
    void ApplyNonSepSub1C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out) const;
    void ApplyNonSepSub3C(int xPixelCount, int yPixelCount, const vector<float>& in, vector<float>& tmp, vector<float>& out) const;
    //
    static float Sub2Var1D(const vector<float> &subKernel, int resSub, vector<float> &varKernel)  {
        // Define a variance filter matching the subpixel filter. Since the
        // variance is uniform for all subpixels of a given pixel, we simply
        // accumulate the squared weights over each tap at the pixel resolution.
        int pad = Ceil2Int(float(subKernel.size()/2-(resSub/2))/resSub);
        int nTaps = 2 * pad + 1;
        varKernel.clear(); varKernel.resize(nTaps, 0.f);
        // Accumulate all squared subpixel filter entries for each pixel.
        // Since the subpixel filter does not necessarily match the pixel
        // filter, we compute the offset between the two.
        int offset = (resSub * nTaps - subKernel.size()) / 2;
        for (uint32_t i = 0; i < subKernel.size(); i++) {
            int idx = (i+offset) / resSub;
            float sqr_val = subKernel[i] * subKernel[i];
            varKernel[idx] += sqr_val * resSub;
        }
        return accumulate(varKernel.begin(), varKernel.end(), 0.f);
    }
    static float Sub2Var2D(int subLength, const vector<float> &subKernel, int resSub, vector<float> &varKernel, int &length)  {
        // Define a variance filter matching the subpixel filter. Since the
        // variance is uniform for all subpixels of a given pixel, we simply
        // accumulate the squared weights over each tap at the pixel resolution.
        int xTapsSub = subLength, yTapsSub = subKernel.size() / subLength;
        int xPad = Ceil2Int(float(xTapsSub/2-(resSub/2))/resSub);
        int yPad = Ceil2Int(float(yTapsSub/2-(resSub/2))/resSub);
        int xTaps = 2 * xPad + 1, yTaps = 2 * yPad + 1;
        varKernel.clear(); varKernel.resize(xTaps*yTaps, 0.f);
        // Accumulate all squared subpixel filter entries for each pixel.
        // Since the subpixel filter does not necessarily match the pixel
        // filter, we compute the offset between the two.
        int xTapOffset = (resSub * xTaps - xTapsSub) / 2;
        int yTapOffset = (resSub * yTaps - yTapsSub) / 2;
        for (int iySub = 0; iySub < yTapsSub; iySub++) {
            for (int ixSub = 0; ixSub < xTapsSub; ixSub++) {
                int tapSub = ixSub + iySub * xTapsSub;
                float sqr_val = subKernel[tapSub] * subKernel[tapSub];

                int tap = (ixSub+xTapOffset) / resSub + (iySub+yTapOffset) / resSub * xTaps;
                varKernel[tap] += sqr_val * resSub * resSub;
            }
        }
        length = xTaps;
        return accumulate(varKernel.begin(), varKernel.end(), 0.f);
    }

    void Normalize(KernelNorm norm) {
        if (norm == KERNEL_NORM_UNIT) {
            float sum = accumulate(_k.begin(), _k.end(), 0.f);
            for (uint32_t i = 0; i < _k.size(); i++) _k[i] /= sum;
            sum = accumulate(_k2.begin(), _k2.end(), 0.f);
            for (uint32_t i = 0; i < _k2.size(); i++) _k2[i] /= sum;
        }
        else if (norm == KERNEL_NORM_STD) {
            float mval = *max_element(_k.begin(), _k.end());
            for (uint32_t i = 0; i < _k.size(); i++) _k[i] /= mval;
            if (!_k2.empty()) {
                mval = *max_element(_k2.begin(), _k2.end());
                for (uint32_t i = 0; i < _k2.size(); i++) _k2[i] /= mval;
            }
        }
        _kNorm = accumulate(_k.begin(), _k.end(), 0.f);
        _k2Norm = _k2.empty() ? 1.f : accumulate(_k2.begin(), _k2.end(), 0.f);
    }

};


class Gauss2D : public Kernel2D {
public:
    Gauss2D(float s, KernelNorm norm, int resSub = 1) {
        Init(s, s, 0.f, norm, resSub);
    }

private:
    void Init(float sx, float sy, float angle, KernelNorm norm, int resSub) {
        _isSeparable = true;
        _resSub = resSub;
        
        // When rotating by PI/2 a separable kernel, swap sx and sy
        if (fabs(M_PI/2.f-angle < 1e-3f)) swap(sx, sy);
        
        // Simply process as 2 1D kernels
        Init1D(sx, norm, _k);
        Init1D(sy, norm, _k2);

        _length = _k.size();
        
        Normalize(norm);
    }

    void Init1D(float sigma, KernelNorm norm, vector<float> &kernel) const {
        kernel.clear();
        
        // For null sigma, there is no filtering to do
        if (sigma == 0.f) return;
        
        // Set kernel parameter
        int pad = Floor2Int(3.f * sigma);
        int nTaps = _resSub * (1+2*pad);
        float alpha = -.5f/sigma/sigma;
        // Set filter values
        kernel.resize(nTaps);
        float offset = _resSub % 2 == 0 ? .5f : 0.f;
        for (int i = 0; i < nTaps; i++) {
            float dist = (offset + i - nTaps/2) / _resSub;
            float val = expf(alpha*dist*dist);
            kernel[i] = val;
        }
    }
};


#endif	/* KERNEL2D_H */

