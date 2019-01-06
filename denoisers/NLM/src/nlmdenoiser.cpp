 /* 
 * File:   NlmeansNlmeansDenoiser.cpp
 * Author: rousselle
 * 
 * Created on March 14, 2012, 10:04 AM
 */

#include "nlmdenoiser.h"

#include <algorithm>
using std::swap;
using std::min_element;
#include <numeric>
using std::accumulate;
#include <cmath>
using std::pow;

#include "memory.h"
#include "box.h"
#include "gaussian.h"
#include "rng.h"


NlmeansDenoiser::NlmeansDenoiser(int xPixelCount, int yPixelCount, const Filter *filter, int wnd_rad, float k,
    int ptc_rad, int resSub) {
    _itrCount = 0;
    
    // NlmeansDenoiser attributes
    _nlmFilter.Init(wnd_rad, ptc_rad, k, xPixelCount, yPixelCount);

    // Attributes of the output image
    _xPixelCount = xPixelCount;
    _yPixelCount = yPixelCount;
    _nPix =_xPixelCount * _yPixelCount;

    // Buffers related to the "base" image
    _imgAvgA.resize(3*_nPix);  _imgAvgB.resize(3*_nPix);
    _imgVarA.resize(3*_nPix);  _imgVarB.resize(3*_nPix);
    _fltAvgA.resize(3*_nPix);  _fltAvgB.resize(3*_nPix);
    _fltRelVarA.resize(3*_nPix);  _fltRelVarB.resize(3*_nPix);
    _fltSppA.resize(_nPix);    _fltSppB.resize(_nPix);
    _imgSppA.resize(_nPix);    _imgSppB.resize(_nPix);
    _boxSppA.resize(_nPix);    _boxSppB.resize(_nPix);
    _boxVarA.resize(3*_nPix);  _boxVarB.resize(3*_nPix);
    _imgAvgVarA.resize(3*_nPix); _imgAvgVarB.resize(3*_nPix);
    _tmpImg.resize(3*_nPix);
    _tmpMap.resize(_nPix);

    // Sub-pixel
    _resSub = resSub;
    _boxSubA.resize(3*_resSub*_resSub*_nPix);
    _boxSubB.resize(3*_resSub*_resSub*_nPix);
    _boxSubVar.resize(3*_resSub*_resSub*_nPix);
    _tmpSub.resize(3*_resSub*_nPix);
    
    // Array to hold pixel errors
    _fltErrA.resize(_nPix); _fltErrB.resize(_nPix);
    
    // Generate all filters needed
    GenerateImgFilters(filter);
}


NlmeansDenoiser::~NlmeansDenoiser() {
    if (_imgAvgFilter != NULL) delete _imgAvgFilter;
    if (_subAvgFilter != NULL) delete _subAvgFilter;
    if (_imgVarFilter != NULL) delete _imgVarFilter;
    if (_imgSmpFilter != NULL) delete _imgSmpFilter;
}


void NlmeansDenoiser::UpdatePixelMean(const NlmPixelVec &pixels, 
    const NlmSubPixelVec &subPixels, ImageBuffer &imgAvg, ImageBuffer &boxSub) {
    // Compute pixel
    FillSubHoles(pixels, subPixels, boxSub);
    // Generate "base" image
    _subAvgFilter->ApplySub(_xPixelCount, _yPixelCount, boxSub, _tmpSub, imgAvg);
}


void NlmeansDenoiser::UpdatePixelMeanVariance(const NlmPixelVec &pixels,
    ImageBuffer &boxSpp, ImageBuffer &imgSpp, ImageBuffer &boxVar,
    ImageBuffer &imgVar, ImageBuffer &imgAvgVar) {
    _nVarZero = 0;
    
    // Obtain pixel variance
    float mean[3];
    _nSamples = 0;
    for (int y = 0, pix = 0; y < _yPixelCount; ++y) {
        for (int x = 0; x < _xPixelCount; ++x, ++pix) {
            const NlmeansPixel &pixel = pixels[pix];
            int n = pixel._nSamplesBox;
            if (n > 1) {
                // Get pixel mean
                mean[0] = pixel._LrgbSumBox[0] / n;
                mean[1] = pixel._LrgbSumBox[1] / n;
                mean[2] = pixel._LrgbSumBox[2] / n;
                // Get unbiased variance estimate: (Sum_sqr - Sum*mean)/(n - 1)
                boxVar[3*pix+0] = max(0.f, (pixel._LrgbSumSqrBox[0] - pixel._LrgbSumBox[0]*mean[0])) / (n-1);
                boxVar[3*pix+1] = max(0.f, (pixel._LrgbSumSqrBox[1] - pixel._LrgbSumBox[1]*mean[1])) / (n-1);
                boxVar[3*pix+2] = max(0.f, (pixel._LrgbSumSqrBox[2] - pixel._LrgbSumBox[2]*mean[2])) / (n-1);
            }
            else {
                // Without sufficient data, we directly use the sample value
                // as a variance estimate
                _nVarZero++;
                boxVar[3*pix+0] = 0.f;//pixvar.LrgbSum[0];
                boxVar[3*pix+1] = 0.f;//pixvar.LrgbSum[1];
                boxVar[3*pix+2] = 0.f;//pixvar.LrgbSum[2];
            }
            // Update sample count
            _nSamples += n;
            boxSpp[pix] = n;
        }
    }
    _imgSmpFilter->Apply(_xPixelCount, _yPixelCount, boxSpp, _tmpMap, imgSpp);
    _imgVarFilter->Apply(_xPixelCount, _yPixelCount, boxVar, _tmpImg, imgVar);
    
    // Convert from pixel variance to pixel mean variance
    for (int pix = 0; pix < _nPix; pix++) {
        boxVar[3*pix+0] /= pixels[pix]._nSamplesBox;
        boxVar[3*pix+1] /= pixels[pix]._nSamplesBox;
        boxVar[3*pix+2] /= pixels[pix]._nSamplesBox;
    }
    _imgVarFilter->Apply(_xPixelCount, _yPixelCount, boxVar, _tmpImg, imgAvgVar);
}


void NlmeansDenoiser::FillSubHoles(const NlmPixelVec &pixels,
    const NlmSubPixelVec &subPixels, ImageBuffer &boxSub) {
    int xSubPixelCount = _resSub * _xPixelCount;

    // We go over every pixel
#pragma omp parallel for
    for (int y = 0; y < _yPixelCount; y++) {
        for (int x = 0; x < _xPixelCount; x++) {
            // Get pixel mean
            int pix = x + _xPixelCount * y;
            const NlmeansPixel &pixel = pixels[pix];
            float mean[3];
            int n = pixel._nSamplesBox;
            if (n == 0)
                mean[0] = mean[1] = mean[2] = 0;
            else {
                mean[0] = pixel._LrgbSumBox[0] / n;
                mean[1] = pixel._LrgbSumBox[1] / n;
                mean[2] = pixel._LrgbSumBox[2] / n;
            }
            // We go over every subpixel
            int xSubMin = x * _resSub, xSubMax = xSubMin + _resSub;
            int ySubMin = y * _resSub, ySubMax = ySubMin + _resSub;
            for (int yy = ySubMin; yy < ySubMax; yy++) {
                for (int xx = xSubMin; xx < xSubMax; xx++) {
                    int subPix = xx + yy * xSubPixelCount;
                    float weight = subPixels[subPix]._nSamplesBox;

                    if (weight == 0.f) {
                        // Fill holes using mean pixel value
                        boxSub[3*subPix+0] = mean[0];
                        boxSub[3*subPix+1] = mean[1];
                        boxSub[3*subPix+2] = mean[2];
                    }
                    else {
                        // Compute this subpixel value
                        float invW = 1.f / weight;
                        boxSub[3*subPix+0] = subPixels[subPix]._LrgbSumBox[0] * invW;
                        boxSub[3*subPix+1] = subPixels[subPix]._LrgbSumBox[1] * invW;
                        boxSub[3*subPix+2] = subPixels[subPix]._LrgbSumBox[2] * invW;
                    }
                }
            }
        }
    }
}


void NlmeansDenoiser::UpdatePixelData(
    const vector<NlmeansPixel>    &pixelsA,
    const vector<NlmeansPixel>    &pixelsB,
    const vector<NlmeansSubPixel> &subPixelsA,
    const vector<NlmeansSubPixel> &subPixelsB,
    NlmeansData dataType) {

    // Pixel color mean and variance
    UpdatePixelMean(pixelsA, subPixelsA, _imgAvgA, _boxSubA);
    UpdatePixelMean(pixelsB, subPixelsB, _imgAvgB, _boxSubB);
    UpdatePixelMeanVariance(pixelsA, _boxSppA, _imgSppA, _boxVarA, _imgVarA, _imgAvgVarA);
    UpdatePixelMeanVariance(pixelsB, _boxSppB, _imgSppB, _boxVarB, _imgVarB, _imgAvgVarB);

    // Filter the data
    _nlmFilter.Apply(dataType, _imgSppA,
        _imgAvgA, _imgVarA, _imgAvgVarA, _imgAvgB, _imgVarB, _imgAvgVarB,
        _fltAvgA, _fltSppA, _fltAvgB, _fltSppB);
    
    // Compute observed variance
    for (size_t i = 0; i < _fltRelVarA.size(); i++) {
        float var = sqr(_fltAvgA[i] - _fltAvgB[i]);
        _fltRelVarA[i] = 2.f * var / (1e-3f + sqr(_fltAvgA[i]));
        _fltRelVarB[i] = 2.f * var / (1e-3f + sqr(_fltAvgB[i]));
    }
    
    // Update the pixel costs
    UpdatePixelCosts(_fltRelVarA, _fltSppA, _fltErrA);
    UpdatePixelCosts(_fltRelVarB, _fltSppB, _fltErrB);
    
    _itrCount++;
}


void NlmeansDenoiser::WriteImage(float* img) {
    // Compute the aggregate noisy image and filtered image
    ImageBuffer imgAvg(3*_nPix), fltAvg(3*_nPix), boxSpp(_nPix);
    CombineData(_imgAvgA, _imgSppA, _imgAvgB, _imgSppB, imgAvg);
    CombineData(_fltAvgA, _imgSppA, _fltAvgB, _imgSppB, fltAvg);
    for (int pix = 0 ; pix < _nPix; pix++) {
        boxSpp[pix] = _boxSppA[pix] + _boxSppB[pix];
    }
    
    memcpy(img, fltAvg.data(), fltAvg.size()*sizeof(float));
    // Dump everything
//    DumpRGB(imgAvg, "img", DUMP_FINAL);
//    DumpRGB(fltAvg, "flt", DUMP_FINAL);
//    DumpMap(boxSpp, "bspp", DUMP_FINAL);
    
    // Compute mean sample rate for each buffer
    float sppA = accumulate(_boxSppA.begin(), _boxSppA.end(), 0.f) / _nPix;
    float sppB = accumulate(_boxSppB.begin(), _boxSppB.end(), 0.f) / _nPix;

    printf("Effective spp: (%.3f, %.3f)\n", sppA, sppB);
}


void NlmeansDenoiser::CombineData(const ImageBuffer &datA, 
    const ImageBuffer &sppA, const ImageBuffer &datB, const ImageBuffer &sppB,
    ImageBuffer &out) {
    // Combine the data using the spp as weights.
    for (int i = 0; i < _nPix; i++) {
        int r = 3*i+0, g = 3*i+1, b = 3*i+2;
        out[r] = (datA[r] * sppA[i] + datB[r] * sppB[i]) / (sppA[i] + sppB[i]);
        out[g] = (datA[g] * sppA[i] + datB[g] * sppB[i]) / (sppA[i] + sppB[i]);
        out[b] = (datA[b] * sppA[i] + datB[b] * sppB[i]) / (sppA[i] + sppB[i]);
    }
}

void NlmeansDenoiser::DumpRGB(vector<float> &img, const string &tag, DumpType dumpType) {
//    // Retrieve "base" name
//    string base(_filename.begin(), _filename.begin() + _filename.find_last_of("."));

//    // Generate output filename
//    char name[256];
//    if (dumpType == DUMP_FINAL)
//        sprintf(name, "%s_%s.exr", base.c_str(), tag.c_str());
//    else // (dumpType == DUMP_ITERATION)
//        sprintf(name, "%s_%s_itr%03d.exr", base.c_str(), tag.c_str(), _itrCount);

//    // Write to disk
//    printf("filename: %s\n", name);
//    ::WriteImage(name, &img[0], NULL, _xPixelCount, _yPixelCount, _xPixelCount, _yPixelCount, 0, 0);
}


void NlmeansDenoiser::DumpMapInv(vector<float> &map, const string &tag, DumpType dumpType) {
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
//        _tmpImg[3*pix+0] = _tmpImg[3*pix+1] = _tmpImg[3*pix+2] = 1.f - map[pix];

//    // Write to disk
//    ::WriteImage(name, &_tmpImg[0], NULL, _xPixelCount, _yPixelCount, _xPixelCount, _yPixelCount, 0, 0);
}


void NlmeansDenoiser::GenerateImgFilters(const Filter* filter) {
    // The mean and filters
    _imgFilter = filter;
    _imgAvgFilter = new Kernel2D(filter, KERNEL_NORM_UNIT);
    _subAvgFilter = new Kernel2D(filter, KERNEL_NORM_UNIT, _resSub);
    _imgVarFilter = Kernel2D::Sub2Var(*_subAvgFilter);
    _imgSmpFilter = new Kernel2D(filter, KERNEL_NORM_STD);
}


void NlmeansDenoiser::UpdatePixelCosts(const ImageBuffer &fltRelVar,
    const ImageBuffer &fltSpp, NlmPixelErrorVec &fltErr) {
    for (int pix = 0; pix < _nPix; pix++) {
        fltErr[pix]._pix = pix;
        
        // The pixel error
        float pixel_error = rgb2avg(&fltRelVar[3*pix]);
        
        // Set the pixel error info
        fltErr[pix]._pix   = pix;
        fltErr[pix]._error = pixel_error / fltSpp[pix];
    }
}


void NlmeansDenoiser::GetSamplingMaps(int spp, int nSamples, ImageBuffer &mapA,
    ImageBuffer &mapB) {
    // Initialize the sampling maps using the pixel costs
    mapA.resize(_nPix); mapB.resize(_nPix);
    for (int pix = 0; pix < _nPix; pix++) {
        mapA[pix] = _fltErrA[pix]._error + _fltErrB[pix]._error;
    }
    
    // Blur the maps a little
    Gauss2D gauss(.8f, KERNEL_NORM_UNIT);
    gauss.Apply(_xPixelCount, _yPixelCount, mapA, _tmpMap, mapA);
    
    // Normalize them to sum up to nSamples/2 each
    float sumA = accumulate(mapA.begin(), mapA.end(), 0.f);
    float nSamplesA = nSamples / 2.f;
    for (int pix = 0; pix < _nPix; pix++) {
        mapA[pix] *= nSamplesA / sumA;
    }
    
    // Clamp the map to "lim" samples per pixel max. "lim" is set to spp-1, so
    // that, even with error propagation, no more than spp can be picked
    int nPixOver1, nPixOver2;
    int lim = spp;
    do {
        nPixOver1 = 0;
        for (int pix = 0; pix < _nPix; pix++) {
            if (mapA[pix] > lim) {
                mapA[pix] = lim;
                nPixOver1 += 1;
            }
        }
        // Redistribute the remaining budget over the map
        nPixOver2 = 0;
        float distributed = accumulate(mapA.begin(), mapA.end(), 0.f);
        float scale = (nSamplesA-nPixOver1*lim)/(distributed-nPixOver1*lim);
        if (scale < 0) {
            Severe("Negative scale in sample redistribution!");
        }
        for (int pix = 0; pix < _nPix; pix++) {
            if (mapA[pix] < lim)
                mapA[pix] *= scale;
            
            if (mapA[pix] > lim)
                nPixOver2 += 1;
        }
    } while(nPixOver2 > 0);
    
    copy(mapA.begin(), mapA.end(), mapB.begin());
//    if (PbrtOptions.verbose) {
//        DumpMap(mapA, "map", DUMP_ITERATION);
//    }
}
