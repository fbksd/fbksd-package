/* 
 * File:   nlmdenoiser.h
 * Author: rousselle
 *
 * Created on March 14, 2012, 10:04 AM
 */

#ifndef NLMEANSDENOISER_H
#define	NLMEANSDENOISER_H

#include "pbrt.h"
#include <vector>
#include <limits>
using std::numeric_limits;
#include "rng.h"
#include "kernel2d.h"
#include "nlmkernel.h"

// Define the pixel data
struct NlmeansPixel {
    NlmeansPixel() {
        _weightSum = 0.f;
        _nSamplesBox = 0.f;
        for (int i = 0; i < 3; ++i) {
            _Lrgb[i] = 0.f;
            _LrgbSumBox[i] = 0.f;
            _LrgbSumSqrBox[i] = 0.f;
            _varSumBox[i] = 0.f;
            _varSumSqrBox[i] = 0.f;
        }
    }
    // These store the sample values, as well as the cumulative weights, as
    // defined by the filter.
    float _Lrgb[3];
    float _weightSum;
    // The 'box' data is used to compute the variance of samples falling within
    // the boundary of a pixel. It is not affected by the reconstruction filter.
    int _nSamplesBox;
    float _LrgbSumBox[3];
    float _LrgbSumSqrBox[3];
    // Buffer variance
    float _tmp[3];
    float _varSumBox[3];
    float _varSumSqrBox[3];
};
typedef vector<NlmeansPixel> NlmPixelVec;

struct NlmeansFeatures {
    NlmeansFeatures() {
        _nSamplesBox = 0;
        _dSumBox = _dSumSqrBox = 0.f;
        _vSumBox = _vSumSqrBox = 0.f;
        for (int i = 0; i < 3; ++i) {
            _nSumBox[i] = _nSumSqrBox[i] = 0.f;
            _rSumBox[i] = _rSumSqrBox[i] = 0.f;
        }
    }
    // The 'box' data is used to compute the variance of samples falling within
    // the boundary of a pixel. It is not affected by the reconstruction filter.
    int _nSamplesBox;
    float _nSumBox[3], _nSumSqrBox[3]; // normal
    float _dSumBox,    _dSumSqrBox;    // depth
    float _vSumBox,    _vSumSqrBox;    // visibility
    float _rSumBox[3], _rSumSqrBox[3]; // reflectance
};

struct NlmeansSubPixel {
    NlmeansSubPixel() {
        _nSamplesBox = 0;
        for (int i = 0; i < 3; ++i)
            _LrgbSumBox[i] = 0.f;
    }
    int _nSamplesBox;
    float _LrgbSumBox[3];
};
typedef vector<NlmeansSubPixel> NlmSubPixelVec;

struct NlmeansSubFeatures {
    NlmeansSubFeatures() {
        _nSamplesBox = 0;
        _dSumBox = 0.f;
        _vSumBox = 0.f;
        for (int i = 0; i < 3; ++i) {
            _nSumBox[i] = 0.f;
            _rSumBox[i] = 0.f;
        }
    }
    int _nSamplesBox;
    float _nSumBox[3]; // normal
    float _dSumBox;    // depth
    float _vSumBox;    // visibility
    float _rSumBox[3]; // reflectance
};

struct NlmeansPixelError {
    NlmeansPixelError() { }
    NlmeansPixelError(int pix, float error) {
        _pix = pix;
        _error = error;
    }
    int _pix;
    float _error;
};

typedef vector<NlmeansPixelError> NlmPixelErrorVec;

class NlmeansDenoiser {
public:
    NlmeansDenoiser(int xPixelCount, int yPixelCount, const Filter *filter, int wnd_rad, float k,
        int ptc_rad, int resSub);
    virtual ~NlmeansDenoiser();

    // Set pixel data
    void UpdatePixelData(
        const vector<NlmeansPixel>    &pixelsA,
        const vector<NlmeansPixel>    &pixelsB,
        const vector<NlmeansSubPixel> &subPixelsA,
        const vector<NlmeansSubPixel> &subPixelsB,
        NlmeansData dataType);
    
    void GetSamplingMaps(int spp, int nSamples, ImageBuffer &mapA,
        ImageBuffer &mapB);

    void WriteImage(float* img);
    
    bool IsReady() const { return _itrCount > 0; }
    
protected:
    // Image buffers. All data is split over two buffers, to prevent correlation
    // between the data noise and the filter weights.
    ImageBuffer _imgAvgA, _imgAvgB; // pixel mean
    ImageBuffer _imgVarA, _imgVarB; // pixel variance
    ImageBuffer _imgAvgVarA, _imgAvgVarB; // variance of the pixel mean
    ImageBuffer _imgSppA, _imgSppB; // number of samples per pixel
    ImageBuffer _fltAvgA, _fltAvgB; // reconstructed images
    ImageBuffer _fltRelVarA, _fltRelVarB; // reconstructed images variance
    ImageBuffer _fltSppA, _fltSppB; // reconstructed images effective sample rate
    
    // Array holding the per-pixel error along with its index
    NlmPixelErrorVec _fltErrA, _fltErrB;
    
    int _itrCount;  // used to monitor the successive calls to GetWorstPixels()
    int _nVarZero;  // count of pixels with estimated variance of zero

    // Image filters
    const Filter *_imgFilter;
    Kernel2D *_imgAvgFilter;
    Kernel2D *_subAvgFilter;
    Kernel2D *_imgVarFilter;
    Kernel2D *_imgSmpFilter;
    
    // Nlmeans filter
    NlmeansKernel _nlmFilter;

    // Attributes of the output image
    int _nSamples; // average per-pixel sample count
    int _nPix, _xPixelCount, _yPixelCount;
    
    void CombineData(const ImageBuffer &datA, const ImageBuffer &sppA,
        const ImageBuffer &datB, const ImageBuffer &sppB, ImageBuffer &out);

    // Data buffers
    vector<float> _tmpImg, _tmpMap;
    // The 'box' variance: variance of the pixel mean using a box filter
    vector<float> _boxVarA, _boxVarB;
    vector<float> _boxSppA, _boxSppB;
    
    // Subpixel related stuff
    int _resSub;
    ImageBuffer _boxSubA, _boxSubB, _boxSubVar;  // the "filled" subpixel grid
    ImageBuffer _tmpSub;  // used to store intermediate gaussian filtering step

    // Methods
    
    // Computation of the base image data
    void GenerateImgFilters(const Filter* filter);
    void UpdatePixelMean(const NlmPixelVec &pixels,
        const NlmSubPixelVec &subPixels, ImageBuffer &imgAvg,
        ImageBuffer &boxSub);
    void UpdatePixelMeanVariance(const NlmPixelVec &pixels, ImageBuffer &boxSpp,
        ImageBuffer &imgSpp, ImageBuffer &boxVar, ImageBuffer &imgVar,
        ImageBuffer &imgAvgVar);
    void FillSubHoles(const NlmPixelVec &pixels, 
        const NlmSubPixelVec &subPixels, ImageBuffer &boxSub);
    
    // Computation of the filtering map
    void UpdatePixelCosts(const ImageBuffer &fltRelVar,
        const ImageBuffer &fltSpp, NlmPixelErrorVec &fltErr);

    static bool SortPixelError(const NlmeansPixelError &e1, const NlmeansPixelError &e2) {
        // We sort pixel error from largest to smallest
        return e1._error > e2._error;
    }
    

    // Utility functions used to dump buffers into EXR images
    enum DumpType {
        DUMP_FINAL,
        DUMP_ITERATION
    };
    void DumpRGB(vector<float> &img, const string &tag, DumpType dumpType);
    template <typename T>
    void DumpMap(vector<T> &img, const string &tag, DumpType dumpType, float scale = 1.f);
    void DumpMapInv(vector<float> &img, const string &tag, DumpType dumpType);

    // Various utility functions
    float sqr(float v) const {
        return v*v;
    }
    float rgb2avg(const float *rgb) const {
        return (rgb[0] + rgb[1] + rgb[2]) / 3.f;
    }
};

template <typename T>
void NlmeansDenoiser::DumpMap(vector<T> &map, const string &tag, DumpType dumpType,
    float scale) {
//    // Retrieve "base" name
//    string base(_filename.begin(), _filename.begin() + _filename.find_last_of("."));

//    // Generate output filename
//    char name[256];
//    if (dumpType == DUMP_FINAL)
//        sprintf(name, "%s_%s.exr", base.c_str(), tag.c_str());
//    else // (dumpType == DUMP_ITERATION)
//        sprintf(name, "%s_%s_itr%03d.exr", base.c_str(), tag.c_str(), _itrCount);

//    // Duplicate over all three channels
//    for (int pix = 0; pix < _nPix; pix++)
//        _tmpImg[3*pix+0] = _tmpImg[3*pix+1] = _tmpImg[3*pix+2] = scale * map[pix];

    // Write to disk
//    ::WriteImage(name, &_tmpImg[0], NULL, _xPixelCount, _yPixelCount, _xPixelCount, _yPixelCount, 0, 0);
}

#endif	/* NLMEANSDENOISER_H */

