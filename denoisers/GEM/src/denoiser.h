/*
 *  Copyright(c) 2011 Fabrice Rousselle.
 * 
 *  You can redistribute and/or modify this file under the terms of the GNU
 *  General Public License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 */

#ifndef DENOISER_H
#define	DENOISER_H

#include "pbrt.h"
#include <vector>
#include <limits>
using std::numeric_limits;
#include "rng.h"
#include "kernel2d.h"


typedef vector<Kernel2D*> FilterBank;
typedef vector<vector<float> > Decomposition;


struct Pixel {
    Pixel() {
        for (int i = 0; i < 3; ++i) {
            Lxyz[i] = splatXYZ[i] = 0.f;
        }
        weightSum = 0.f;
    }
    float Lxyz[3];
    float weightSum;
    float splatXYZ[3];
    float pad;
};

struct SubPixel {
    SubPixel() {
        for (int i = 0; i < 3; ++i)
            Lxyz[i] = 0.f;
        weightSum = 0.f;
    }
    float Lxyz[3];
    float weightSum;
};

struct PixelError {
    PixelError() { }
    PixelError(int pix, float error) {
        _pix = pix;
        _error = error;
    }
    int _pix;
    float _error;
};
typedef vector<PixelError> PixelErrorVec;

struct PixelArea {
    int   _pix;
    float _xPos, _yPos;
    int _scale;
};
typedef vector<PixelArea> PixelAreaVec;

struct BoxVariance {
    BoxVariance() {
        for (int i = 0; i < 3; ++i) LrgbSum[i] = LrgbSumSqr[i] = 0.f;
        nSamples = 0;
    }
    float LrgbSum[3];
    float LrgbSumSqr[3];
    int nSamples;
};

class Denoiser {
public:
    Denoiser(int nScales, int xPixelCount, int yPixelCount,
        const Filter *filter, const string &filename, float gamma, int resSub);
    virtual ~Denoiser();

    // Subpixel resolution
    virtual void SetPixelMeanAndVariance(const vector<SubPixel> &subPixels,
         BlockedArray<BoxVariance> *variance);
    virtual void SetPixelMeanAndVarianceFinal(const vector<SubPixel> &subPixels,
         BlockedArray<BoxVariance> *variance);
    void WriteImage(float* img);
    virtual void GetWorstPixels(int nPixels, PixelAreaVec &pixelAreas, int spp);

    bool IsAdaptive() const { return _itrCount > 0; }

    void GetFilterbank(vector<const Kernel2D*> &filters) const {
        filters.clear();
        filters.push_back(_imgSmpFilter);
        for (int s = 0; s < _nSteps; s++)
            filters.push_back(_decSmpFilters[s]);
    }

protected:
    int _itrCount;  // used to monitor the successive calls to GetWorstPixels()
    int _nVarZero;  // count of pixels with estimated variance of zero

    const Filter *_imgFilter;

    // Denoiser attributes
    int _nSteps, _nOri;
    float _scaleFactor;
    vector<float> _aniFactors;
    vector<float> _scales;
    vector<vector<float> > _scalesMinor, _scalesMajor;
    float _gamma;

    // Attributes of the output image
    int _nSamples; // average per-pixel sample count
    string _filename;
    int _nPix, _xPixelCount, _yPixelCount;

    // Data buffers
    vector<float> _tmpImg, _tmpMap, _tmpSel;
    // The 'box' variance: variance of the pixel mean using a box filter
    vector<float> _boxVar;
    // Base: the original image and all related data
    vector<float> _imgAvg, _refAvg;  // pixel mean
    vector<float> _imgVar, _refVar;  // variance of the pixel mean
    vector<float> _boxSmp, _imgSmp;  // number of samples in each pixel
    vector<float> _imgFlt;
    // Decompositions
    Decomposition _decAvg;  // pixel mean
    Decomposition _decVar;  // variance of the pixel mean
    Decomposition _decSmp;  // number of samples in each pixel
    // Selection map
    vector<int> _imgSel;  // scale selection map
    Decomposition _decSel; // per-scale binary stop map
    // MSE (Mean Squared Error)
    Decomposition _decMse;

    // Filterbanks
    Kernel2D *_subAvgFilter, *_imgVarFilter, *_imgSmpFilter;
    FilterBank _decAvgFilters; // -> decomposition of pixel mean
    FilterBank _decVarFilters; // -> decomposition of variance of the pixel mean
    FilterBank _decSmpFilters; // -> decomposition of count of sample per pixel
    FilterBank _decSelFilters; // -> decomposition of count of sample per pixel
    void ResetFilterbank(FilterBank &bank) {
        for (uint32_t s = 0; s < bank.size(); s++)
            delete bank[s];
        bank.resize(_nSteps);
    }

    PixelErrorVec _pixelErrors; // used to sort pixels

    // Subpixel related stuff
    int _resSub;
    vector<float> _boxSub;  // the "filled" subpixel grid
    vector<float> _tmpSub;  // used to store intermediate gaussian filtering step

    // Methods

    // Generation of the filterbanks
    enum FilterRes {
        FLT_PIX,
        FLT_SUB
    };
    void GenerateImgFilters(const Filter* filter);
    void GenerateDecFilterBanks(int nSteps, float scaleFactor, float scaleMin, float alpha);

    // Computation of the base image data
    void UpdatePixelMean(const vector<SubPixel> &subPixels, BlockedArray<BoxVariance> *variance);
    void UpdatePixelMeanVariance(BlockedArray<BoxVariance> *variance);

    // Convolutions
    void Decompose(const vector<float> &in, const FilterBank &filters, Decomposition &dec) {
        for (uint32_t s = 0; s < filters.size(); s++)
           filters[s]->Apply(_xPixelCount, _yPixelCount, in, _tmpImg, dec[s]);
    }
    void DecomposeMap(const vector<float> &in, const FilterBank &filters, Decomposition &dec) {
        for (uint32_t s = 0; s < filters.size(); s++)
           filters[s]->Apply(_xPixelCount, _yPixelCount, in, _tmpMap, dec[s]);
    }

    // Computation of the filtering map
    void UpdateSelectionMap();
    void UpdatePixelCosts(int spp);

    virtual void GetFilteringMap(int spp);
    virtual void GetFilteringMapFinal();
    void GetMse(float param);
    void GetFilteringMapInc();
    // The loss: bias increase
    void GetLoss(const float *avg1, const float *avg2, float beta, float *loss) {
        loss[0] = avg1[0] - avg2[0]; loss[0] *= beta*loss[0];
        loss[1] = avg1[1] - avg2[1]; loss[1] *= beta*loss[1];
        loss[2] = avg1[2] - avg2[2]; loss[2] *= beta*loss[2];
    }
    void GetGain(const float *var1, const float *var2, float *gain) {
        gain[0] = var1[0] - var2[0];
        gain[1] = var1[1] - var2[1];
        gain[2] = var1[2] - var2[2];
    }
    // Computation of the filtered image
    virtual void GetFilteredImage();

    static bool SortPixelError(const PixelError &e1, const PixelError &e2) {
        // We sort pixel error from largest to smallest
        return e1._error > e2._error;
    }
    float GetPixelError(int pix, int scale, float mse, int spp) {
        // We take a relative mse, with damping to limit impact of near-zero
        float val = (scale == 0) ? rgb2avg(&_imgAvg[3*pix]) : rgb2avg(&_decAvg[scale-1][3*pix]);
        mse /= 1e-3f + val * val;
        // Estimate the potential variance reduction. Since the variance is
        // proportional to the number of samples, the relative gain is inversely
        // proportional to the actual number of samples. We scale by 10000 to
        // prevent near-zero values.
        if (scale > 0)
            mse *= 10000.f * spp / (spp + _decSmp[scale-1][pix]);
        else
            mse *= 10000.f * spp / (spp + _imgSmp[pix]);

        return mse;
    }

    void FilterSelectionMaps();
    void FilterSelectionMapsFinal();

    // Utility functions used to dump buffers into EXR images
    enum DumpType {
        DUMP_FINAL,
        DUMP_ITERATION
    };
    void DumpDecomposition(Decomposition &dec, const string &tag, DumpType dumpType);
    void DumpImageRGB(vector<float> &img, const string &tag, DumpType dumpType);
    template <typename T>
    void DumpMap(vector<T> &img, const string &tag, DumpType dumpType, float scale);
    void DumpMapInv(vector<float> &img, const string &tag, DumpType dumpType);
    void DumpErrorMap(const string &tag, DumpType dumpType);
    void DumpErrorMap(DumpType dumpType);

    // Functions related to filtering of the subpixel grid
    void FillSubHoles(const vector<SubPixel> &subPixels, BlockedArray<BoxVariance> *variance);

    // Various utility functions
    float rgb2lum(const float *rgb) const {
        return .299f * rgb[0] + .587f * rgb[1] + .114f * rgb[2];
    }
    float rgb2avg(const float *rgb) const {
        return (rgb[0] + rgb[1] + rgb[2]) / 3.f;
    }

};

template <typename T>
void Denoiser::DumpMap(vector<T> &map, const string &tag, DumpType dumpType,
    float scale) {
    // Retrieve "base" name
    string base(_filename.begin(), _filename.begin() + _filename.find_last_of("."));

    // Generate output filename
    char name[256];
    if (dumpType == DUMP_FINAL)
        sprintf(name, "%s_%s.exr", base.c_str(), tag.c_str());
    else // (dumpType == DUMP_ITERATION)
        sprintf(name, "%s_%s_itr%03d.exr", base.c_str(), tag.c_str(), _itrCount);

    // Duplicate over all three channels
    for (int pix = 0; pix < _nPix; pix++)
        _tmpImg[3*pix+0] = _tmpImg[3*pix+1] = _tmpImg[3*pix+2] = scale * map[pix];

    // Write to disk
    //::WriteImage(name, &_tmpImg[0], NULL, _xPixelCount, _yPixelCount, _xPixelCount, _yPixelCount, 0, 0);
}

#endif	/* DENOISER_H */
