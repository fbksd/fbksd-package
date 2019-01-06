
/*
 * File:   kernel2d.cpp
 * Author: rousselle
 *
 * Created on February 21, 2011, 10:05 AM
 */


#include "kernel2d.h"
#include "rng.h"


Kernel2D::Kernel2D(const Filter* filter, KernelNorm norm, int resSub) {
    _resSub = resSub;
    _isSeparable = true;

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

    // Mirror for _k2
    _k2 = _k;

    _length = _k.size();
    
    Normalize(norm);
}


void Kernel2D::ApplySep1C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out,
    KernelSkip skip) const {

    int nPix = xPixelCount * yPixelCount;

    // Horizontal filtering
    float kxPad = 1.f;
    if (_k.empty())
        copy(in.begin(), in.end(), tmp.begin());
    else {
        int xTaps = _k.size();
        int xPad = xTaps/2;
        kxPad = _k[xPad];
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int pix = 0; pix < nPix; pix++) {
            int y = pix / xPixelCount, x = pix - y * xPixelCount;
            tmp[pix] = 0.f;
            // Restrict the window
            float tmin = max(0, xPad-x), tmax = min(xTaps, xPixelCount+xPad-x);
            // Do the filtering
            float sum = 0.f;
            for (int tap = tmin; tap < tmax; tap++) {
                int spix = pix+tap-xPad;
                tmp[pix] += _k[tap] * in[spix];
                sum += _k[tap];
            }
            tmp[pix] *= _kNorm/sum;

            // decrease contribution of center pixel to, at most, th%, and reweight the rest
            if (_k2.empty() && skip == KERNEL_SKIP_CENTER) {
                float th = 0.01f; // small residual contribution to tip the balance if needed
                float cwt = kxPad, diff = max(0.f, cwt-th);
                float scale = _kNorm / (_kNorm - diff);
                tmp[pix] = (tmp[pix]-diff*in[pix]) * scale;

            }
        }
    }

    // Vertical filtering
    if (_k2.empty())
        copy(tmp.begin(), tmp.end(), out.begin());
    else {
        int yTaps = _k2.size();
        int yPad = yTaps/2;
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int pix = 0; pix < nPix; pix++) {
            int y = pix / xPixelCount;
            out[pix] = 0.f;
            // Restrict the window
            float tmin = max(0, yPad-y), tmax = min(yTaps, yPixelCount+yPad-y);
            // Do the filtering
            float sum = 0.f;
            for (int tap = tmin; tap < tmax; tap++) {
                int spix = pix + (tap-yPad) * xPixelCount;
                out[pix] += _k2[tap] * tmp[spix];
                sum += _k2[tap];
            }
            out[pix] *= _k2Norm/sum;

            // decrease contribution of center pixel to, at most, th%, and reweight the rest
            if (skip == KERNEL_SKIP_CENTER) {
                float norm = _kNorm * _k2Norm;
                float th = 0.01f; // small residual contribution to tip the balance if needed
                float cwt = kxPad * _k2[yPad], diff = max(0.f, cwt-th);
                float scale = norm / (norm - diff);
                out[pix] = (out[pix]-diff*in[pix]) * scale;

            }
        }
    }
}


void Kernel2D::ApplySep3C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out,
    KernelSkip skip) const {

    int nPix = xPixelCount * yPixelCount;

    // Horizontal filtering
    float kxPad = 1.f;
    if (_k.empty())
        copy(in.begin(), in.end(), tmp.begin());
    else {
        int xTaps = _k.size();
        int xPad = xTaps/2;
        kxPad = _k[xPad];
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int pix = 0; pix < nPix; pix++) {
            int y = pix / xPixelCount, x = pix - y * xPixelCount;
            tmp[3*pix+0] = tmp[3*pix+1] = tmp[3*pix+2] = 0.f;
            // Restrict the window
            float tmin = max(0, xPad-x), tmax = min(xTaps, xPixelCount+xPad-x);
            // Do the filtering
            float sum = 0.f;
            for (int tap = tmin; tap < tmax; tap++) {
                int spix = pix+tap-xPad;
                tmp[3*pix+0] += _k[tap] * in[3*spix+0];
                tmp[3*pix+1] += _k[tap] * in[3*spix+1];
                tmp[3*pix+2] += _k[tap] * in[3*spix+2];
                sum += _k[tap];
            }
            tmp[3*pix+0] *= _kNorm/sum;
            tmp[3*pix+1] *= _kNorm/sum;
            tmp[3*pix+2] *= _kNorm/sum;

            // decrease contribution of center pixel to, at most, th%, and reweight the rest
            if (_k2.empty() && skip == KERNEL_SKIP_CENTER) {
                float th = 0.01f; // small residual contribution to tip the balance if needed
                float cwt = kxPad, diff = max(0.f, cwt-th);
                float scale = _kNorm / (_kNorm - diff);
                tmp[3*pix+0] = (tmp[3*pix+0]-diff*in[3*pix+0]) * scale;
                tmp[3*pix+1] = (tmp[3*pix+1]-diff*in[3*pix+1]) * scale;
                tmp[3*pix+2] = (tmp[3*pix+2]-diff*in[3*pix+2]) * scale;
            }
        }
    }

    // Vertical filtering
    if (_k2.empty())
        copy(tmp.begin(), tmp.end(), out.begin());
    else {
        int yTaps = _k2.size();
        int yPad = yTaps/2;
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int pix = 0; pix < nPix; pix++) {
            int y = pix / xPixelCount;
            out[3*pix+0] = out[3*pix+1] = out[3*pix+2] = 0.f;
            // Restrict the window
            float tmin = max(0, yPad-y), tmax = min(yTaps, yPixelCount+yPad-y);
            // Do the filtering
            float sum = 0.f;
            for (int tap = tmin; tap < tmax; tap++) {
                int spix = pix + (tap-yPad) * xPixelCount;
                out[3*pix+0] += _k2[tap] * tmp[3*spix+0];
                out[3*pix+1] += _k2[tap] * tmp[3*spix+1];
                out[3*pix+2] += _k2[tap] * tmp[3*spix+2];
                sum += _k2[tap];
            }
            out[3*pix+0] *= _k2Norm/sum;
            out[3*pix+1] *= _k2Norm/sum;
            out[3*pix+2] *= _k2Norm/sum;

            // decrease contribution of center pixel to, at most, th%, and reweight the rest
            if (skip == KERNEL_SKIP_CENTER) {
                float norm = _kNorm * _k2Norm;
                float th = 0.01f; // small residual contribution to tip the balance if needed
                float cwt = kxPad * _k2[yPad], diff = max(0.f, cwt-th);
                float scale = norm / (norm - diff);
                out[3*pix+0] = (out[3*pix+0]-diff*in[3*pix+0]) * scale;
                out[3*pix+1] = (out[3*pix+1]-diff*in[3*pix+1]) * scale;
                out[3*pix+2] = (out[3*pix+2]-diff*in[3*pix+2]) * scale;

            }
        }
    }
}


void Kernel2D::ApplySepSub1C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out) const {

    int nPix = xPixelCount * yPixelCount;
    int xSubPixelCount = xPixelCount * _resSub;
    int ySubPixelCount = yPixelCount * _resSub;

    // Horizontal filtering
    if (_k.empty()) {
        int nTmpPix = xPixelCount * ySubPixelCount;
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int tmpPix = 0; tmpPix < nTmpPix; tmpPix++) {
            int y = tmpPix / xPixelCount, x = tmpPix - y * xPixelCount;
            x = _resSub*x + _resSub/2;
            int subPix = x + y * xSubPixelCount;
            tmp[tmpPix] = in[subPix];
        }
    }
    else {
        int xTaps = int(_k.size());
        int xPad = xTaps/2;
        int nTmpPix = xPixelCount * ySubPixelCount;
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int tmpPix = 0; tmpPix < nTmpPix; tmpPix++) {
            int y = tmpPix / xPixelCount, x = tmpPix - y * xPixelCount;
            x = _resSub*x + _resSub/2;
            tmp[tmpPix] = 0.f;
            // Restrict the window
            float tmin = max(0, xPad-x), tmax = min(xTaps, xSubPixelCount+xPad-x);
            // Do the filtering
            float sum = 0.f;
            int subPix = x + y * xSubPixelCount;
            for (int tap = tmin; tap < tmax; tap++) {
                int subOffset = tap-xPad;
                int spix = subPix+subOffset;
                tmp[tmpPix] += _k[tap] * in[spix];
                sum += _k[tap];
            }
            tmp[tmpPix] *= _kNorm/sum;
        }
    }

    // Vertical filtering
    if (_k2.empty()) {
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int pix = 0; pix < nPix; pix++) {
            int y = pix / xPixelCount, x = pix - y * xPixelCount;
            y = _resSub*y + _resSub/2;
            int tmpPix = x + y * xPixelCount;
            out[pix] = tmp[tmpPix];
        }
    }
    else {
        int yTaps = _k2.size();
        int yPad = yTaps/2;
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int pix = 0; pix < nPix; pix++) {
            int y = pix / xPixelCount, x = pix - y * xPixelCount;
            y = _resSub*y + _resSub/2;
            out[pix] = 0.f;
            // Restrict the window
            float tmin = max(0, yPad -y), tmax = min(yTaps, ySubPixelCount+yPad -y);
            // Do the filtering
            float sum = 0.f;
            int tmpPix = x + y * xPixelCount;
            for (int tap = tmin; tap < tmax; tap++) {
                int subOffset = (tap-yPad ) * xPixelCount;
                int spix = tmpPix+subOffset;
                out[pix] += _k2[tap] * tmp[spix];
                sum += _k2[tap];
            }
            out[pix] *= _k2Norm/sum;
        }
    }
}


void Kernel2D::ApplySepSub3C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out) const {

    int nPix = xPixelCount * yPixelCount;
    int xSubPixelCount = xPixelCount * _resSub;
    int ySubPixelCount = yPixelCount * _resSub;

    // Horizontal filtering
    if (_k.empty()) {
        int nTmpPix = xPixelCount * ySubPixelCount;
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int tmpPix = 0; tmpPix < nTmpPix; tmpPix++) {
            int y = tmpPix / xPixelCount, x = tmpPix - y * xPixelCount;
            x = _resSub*x + _resSub/2;
            int subPix = x + y * xSubPixelCount;
            tmp[3*tmpPix+0] = in[3*subPix+0];
            tmp[3*tmpPix+1] = in[3*subPix+1];
            tmp[3*tmpPix+2] = in[3*subPix+2];
        }
    }
    else {
        int xTaps = int(_k.size());
        int xPad = xTaps/2;
        int nTmpPix = xPixelCount * ySubPixelCount;
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int tmpPix = 0; tmpPix < nTmpPix; tmpPix++) {
            int y = tmpPix / xPixelCount, x = tmpPix - y * xPixelCount;
            x = _resSub*x + _resSub/2;
            tmp[3*tmpPix+0] = tmp[3*tmpPix+1] = tmp[3*tmpPix+2] = 0.f;
            // Restrict the window
            float tmin = max(0, xPad-x), tmax = min(xTaps, xSubPixelCount+xPad-x);
            // Do the filtering
            float sum = 0.f;
            int subPix = x + y * xSubPixelCount;
            for (int tap = tmin; tap < tmax; tap++) {
                int subOffset = tap-xPad;
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
    }

    // Vertical filtering
    if (_k2.empty()) {
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int pix = 0; pix < nPix; pix++) {
            int y = pix / xPixelCount, x = pix - y * xPixelCount;
            y = _resSub*y + _resSub/2;
            int tmpPix = x + y * xPixelCount;
            out[3*pix+0] = tmp[3*tmpPix+0];
            out[3*pix+1] = tmp[3*tmpPix+1];
            out[3*pix+2] = tmp[3*tmpPix+2];
        }
    }
    else {
        int yTaps = _k2.size();
        int yPad = yTaps/2;
#pragma omp parallel for num_threads(PbrtOptions.nCores)
        for (int pix = 0; pix < nPix; pix++) {
            int y = pix / xPixelCount, x = pix - y * xPixelCount;
            y = _resSub*y + _resSub/2;
            out[3*pix+0] = out[3*pix+1] = out[3*pix+2] = 0.f;
            // Restrict the window
            float tmin = max(0, yPad -y), tmax = min(yTaps, ySubPixelCount+yPad -y);
            // Do the filtering
            float sum = 0.f;
            int tmpPix = x + y * xPixelCount;
            for (int tap = tmin; tap < tmax; tap++) {
                int subOffset = (tap-yPad ) * xPixelCount;
                int spix = tmpPix+subOffset;
                out[3*pix+0] += _k2[tap] * tmp[3*spix+0];
                out[3*pix+1] += _k2[tap] * tmp[3*spix+1];
                out[3*pix+2] += _k2[tap] * tmp[3*spix+2];
                sum += _k2[tap];
            }
            out[3*pix+0] *= _k2Norm/sum;
            out[3*pix+1] *= _k2Norm/sum;
            out[3*pix+2] *= _k2Norm/sum;
        }
    }
}


void Kernel2D::ApplyNonSep1C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out,
    KernelSkip skip) const {
    int xPad = _length/2, yPad = _k.size()/_length/2;
    int xTaps = _length, yTaps = _k.size()/_length;

    // Go over every pixel of the output buffer
#pragma omp parallel for num_threads(PbrtOptions.nCores)
    for (int y = 0; y < yPixelCount; y++) {
        for (int x = 0; x < xPixelCount; x++) {
            int pix = x + y * xPixelCount;
            out[pix] = 0.f;

            // Crop filtering window
            int xTapMin = max(0, xPad-x), xTapMax = min(xTaps, xPixelCount+xPad-x);
            int yTapMin = max(0, yPad-y), yTapMax = min(yTaps, yPixelCount+yPad-y);

            // Perform the filtering
            float sum = 0.f;
            for (int yTap = yTapMin; yTap < yTapMax; yTap++) {
                int ypix = pix + (yTap-yPad) * xPixelCount;

                for (int xTap = xTapMin; xTap < xTapMax; xTap++) {
                    int i = xTap + yTap * xTaps;
                    float wt = _k[i];

                    int spix = ypix + xTap-xPad;
                    out[pix] += wt * in[spix];
                    sum += wt;
                }
            }
            out[pix] *= _kNorm/sum;

            // decrease contribution of center pixel to, at most, th%, and reweight the rest
            if (skip == KERNEL_SKIP_CENTER) {
                float th = 0.01f; // small residual contribution to tip the balance if needed
                float cwt = _k[xPad+yPad*_length], diff = max(0.f, cwt-th);
                out[pix] = (out[pix]-diff*in[pix]) / (1.f-diff);

            }
        }
    }
}


void Kernel2D::ApplyNonSep3C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out,
    KernelSkip skip) const {
    int xPad = _length/2, yPad = _k.size()/_length/2;
    int xTaps = _length, yTaps = _k.size()/_length;

    // Go over every pixel of the output buffer
#pragma omp parallel for num_threads(PbrtOptions.nCores)
    for (int y = 0; y < yPixelCount; y++) {
        for (int x = 0; x < xPixelCount; x++) {
            int pix = x + y * xPixelCount;
            out[3*pix+0] = out[3*pix+1] = out[3*pix+2] = 0.f;

            // Crop filtering window
            int xTapMin = max(0, xPad-x), xTapMax = min(xTaps, xPixelCount+xPad-x);
            int yTapMin = max(0, yPad-y), yTapMax = min(yTaps, yPixelCount+yPad-y);

            // Perform the filtering
            float sum = 0.f;
            for (int yTap = yTapMin; yTap < yTapMax; yTap++) {
                int ypix = pix + (yTap-yPad) * xPixelCount;

                for (int xTap = xTapMin; xTap < xTapMax; xTap++) {
                    int i = xTap + yTap * xTaps;
                    float wt = _k[i];

                    int spix = ypix + xTap-xPad;
                    out[3*pix+0] += wt * in[3*spix+0];
                    out[3*pix+1] += wt * in[3*spix+1];
                    out[3*pix+2] += wt * in[3*spix+2];
                    sum += wt;
                }
            }
            out[3*pix+0] *= _kNorm/sum;
            out[3*pix+1] *= _kNorm/sum;
            out[3*pix+2] *= _kNorm/sum;

            // decrease contribution of center pixel to, at most, th%, and reweight the rest
            if (skip == KERNEL_SKIP_CENTER) {
                float th = 0.01f; // small residual contribution to tip the balance if needed
                float cwt = _k[xPad+yPad*_length], diff = max(0.f, cwt-th);
                out[3*pix+0] = (out[3*pix+0]-diff*in[3*pix+0]) / (1.f-diff);
                out[3*pix+1] = (out[3*pix+1]-diff*in[3*pix+1]) / (1.f-diff);
                out[3*pix+2] = (out[3*pix+2]-diff*in[3*pix+2]) / (1.f-diff);

            }
        }
    }
}


void Kernel2D::ApplyNonSepSub1C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out) const {

    int xSubPixelCount = xPixelCount * _resSub;
    int ySubPixelCount = yPixelCount * _resSub;

    int xPad = _length/2, yPad = _k.size()/_length/2;
    int xTaps = _length, yTaps = _k.size()/_length;

    // Go over every pixel of the output buffer
#pragma omp parallel for num_threads(PbrtOptions.nCores)
    for (int y = 0; y < yPixelCount; y++) {
        for (int x = 0; x < xPixelCount; x++) {
            // Initialize output pixel
            int pix = x + y * xPixelCount;
            out[pix] = 0.f;

            // Compute corresponding subpixel position
            int xSub = _resSub*x + _resSub/2, ySub = _resSub*y + _resSub/2;
            int pixSub = xSub + ySub * xSubPixelCount;

            // Crop filtering window
            int xTapMin = max(0, xPad-xSub), xTapMax = min(xTaps, xSubPixelCount+xPad-xSub);
            int yTapMin = max(0, yPad-ySub), yTapMax = min(yTaps, ySubPixelCount+yPad-ySub);

            // Perform the filtering
            float sum = 0.f;
            for (int yTap = yTapMin; yTap < yTapMax; yTap++) {
                int ypix = pixSub + (yTap-yPad) * xSubPixelCount;

                for (int xTap = xTapMin; xTap < xTapMax; xTap++) {
                    int i = xTap + yTap * xTaps;
                    float wt = _k[i];

                    int spix = ypix + xTap-xPad;
                    out[pix] += wt * in[spix];
                    sum += wt;
                }
            }
            out[pix] *= _kNorm/sum;
        }
    }
}


void Kernel2D::ApplyNonSepSub3C(int xPixelCount, int yPixelCount,
    const vector<float>& in, vector<float>& tmp, vector<float>& out) const {

    int xSubPixelCount = xPixelCount * _resSub;
    int ySubPixelCount = yPixelCount * _resSub;

    int xPad = _length/2, yPad = _k.size()/_length/2;
    int xTaps = _length, yTaps = _k.size()/_length;

    // Go over every pixel of the output buffer
#pragma omp parallel for num_threads(PbrtOptions.nCores)
    for (int y = 0; y < yPixelCount; y++) {
        for (int x = 0; x < xPixelCount; x++) {
            // Initialize output pixel
            int pix = x + y * xPixelCount;
            out[3*pix+0] = out[3*pix+1] = out[3*pix+2] = 0.f;

            // Compute corresponding subpixel position
            int xSub = _resSub*x + _resSub/2, ySub = _resSub*y + _resSub/2;
            int pixSub = xSub + ySub * xSubPixelCount;

            // Crop filtering window
            int xTapMin = max(0, xPad-xSub), xTapMax = min(xTaps, xSubPixelCount+xPad-xSub);
            int yTapMin = max(0, yPad-ySub), yTapMax = min(yTaps, ySubPixelCount+yPad-ySub);

            // Perform the filtering
            float sum = 0.f;
            for (int yTap = yTapMin; yTap < yTapMax; yTap++) {
                int ypix = pixSub + (yTap-yPad) * xSubPixelCount;

                for (int xTap = xTapMin; xTap < xTapMax; xTap++) {
                    int i = xTap + yTap * xTaps;
                    float wt = _k[i];

                    int spix = ypix + xTap-xPad;
                    out[3*pix+0] += wt * in[3*spix+0];
                    out[3*pix+1] += wt * in[3*spix+1];
                    out[3*pix+2] += wt * in[3*spix+2];
                    sum += wt;
                }
            }
            out[3*pix+0] *= _kNorm/sum;
            out[3*pix+1] *= _kNorm/sum;
            out[3*pix+2] *= _kNorm/sum;
        }
    }
}


