/*
 *  Copyright(c) 2011 Fabrice Rousselle.
 * 
 *  You can redistribute and/or modify this file under the terms of the GNU
 *  General Public License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 */

#include <algorithm>
using std::swap;
#include <numeric>
#include <thread>
using std::accumulate;

#include "denoiser.h"
//#include "spectrum.h"
#include "memory.h"
//#include "box.h"
#include "gaussian.h"
#include "rng.h"

Denoiser::Denoiser(int nScales, int xPixelCount, int yPixelCount,
    const Filter *filter, const string &filename, float gamma, int resSub) {
    _itrCount = 0;

    // Denoiser attributes
    _gamma = gamma;

    // Attributes of the output image
    _filename = filename;
    _xPixelCount = xPixelCount;
    _yPixelCount = yPixelCount;
    _nPix =_xPixelCount * _yPixelCount;

    // Buffers related to the "base" image
    _imgAvg.resize(3*_nPix);
    _imgVar.resize(3*_nPix);
    _boxSmp.resize(_nPix);
    _imgSmp.resize(_nPix);
    _imgSel.resize(_nPix);
    _boxVar.resize(3*_nPix);
    _tmpImg.resize(3*_nPix);
    _imgFlt.resize(3*_nPix);

    // Allocate map arrays
    _tmpMap.resize(_nPix);
    _tmpSel.resize(_nPix);

    // Subpixel
    _resSub = resSub;
    _boxSub.resize(3*_resSub*_resSub*_nPix);
    _tmpSub.resize(3*_resSub*_nPix);

    // Array to hold pixel errors
    _pixelErrors.resize(_nPix);

    _subAvgFilter = _imgVarFilter = NULL;

    // Generate all filters needed
    GenerateImgFilters(filter);
    GenerateDecFilterBanks(4, 2.f, 1.f, 1.f);
}


Denoiser::~Denoiser() {
    if (_subAvgFilter != NULL) delete _subAvgFilter;
    if (_imgVarFilter != NULL) delete _imgVarFilter;
    ResetFilterbank(_decAvgFilters);
    ResetFilterbank(_decVarFilters);
    ResetFilterbank(_decSmpFilters);
    ResetFilterbank(_decSelFilters);
}


void Denoiser::UpdatePixelMean(const vector<SubPixel> &subPixels,
    BlockedArray<BoxVariance> *variance) {
    // Compute pixel
    FillSubHoles(subPixels, variance);

    // Generate "base" image
    _subAvgFilter->ApplySub(_xPixelCount, _yPixelCount, _boxSub, _tmpSub, _imgAvg);
}


void Denoiser::UpdatePixelMeanVariance(BlockedArray<BoxVariance> *variance) {
    _nVarZero = 0;
    // Obtain pixel variance
    float mean[3];
    _nSamples = 0;
    for (int y = 0, pix = 0; y < _yPixelCount; ++y) {
        for (int x = 0; x < _xPixelCount; ++x, ++pix) {
            const BoxVariance &pixvar = (*variance)(x, y);
            int n = pixvar.nSamples;
            if (n > 1) {
                // Get pixel mean
                mean[0] = pixvar.LrgbSum[0] / n;
                mean[1] = pixvar.LrgbSum[1] / n;
                mean[2] = pixvar.LrgbSum[2] / n;
                // Get unbiased variance estimate: (Sum_sqr - Sum*mean)/(n - 1)
                _boxVar[3*pix+0] = max(0.f,(pixvar.LrgbSumSqr[0] - pixvar.LrgbSum[0]*mean[0])) / (n-1);
                _boxVar[3*pix+1] = max(0.f,(pixvar.LrgbSumSqr[1] - pixvar.LrgbSum[1]*mean[1])) / (n-1);
                _boxVar[3*pix+2] = max(0.f,(pixvar.LrgbSumSqr[2] - pixvar.LrgbSum[2]*mean[2])) / (n-1);
                // To go from sample variance to pixel mean, divide again by n
                _boxVar[3*pix+0] /= n;
                _boxVar[3*pix+1] /= n;
                _boxVar[3*pix+2] /= n;
            }
            else {
                // Without sufficient data, we directly use the sample value
                // as a variance estimate
                _nVarZero++;
                _boxVar[3*pix+0] = 0;//pixvar.LrgbSum[0];
                _boxVar[3*pix+1] = 0;//pixvar.LrgbSum[1];
                _boxVar[3*pix+2] = 0;//pixvar.LrgbSum[2];
            }
            // Update sample count
            _nSamples += n;
            _boxSmp[pix] = n;
        }
    }

    _imgVarFilter->Apply(_xPixelCount, _yPixelCount, _boxVar, _tmpImg, _imgVar);
}


void Denoiser::SetPixelMeanAndVariance(const vector<SubPixel> &subPixels,
    BlockedArray<BoxVariance> *variance) {

    // Update "base" image data
    UpdatePixelMean(subPixels, variance);
    UpdatePixelMeanVariance(variance);

    // Decompose the base image data
    for (int s = 0; s < _nSteps; s++) {
        _decAvgFilters[s]->Apply(_xPixelCount, _yPixelCount, _imgAvg, _tmpImg, _decAvg[s]);
    }
    Decompose(_boxVar, _decVarFilters, _decVar);
    _imgSmpFilter->Apply(_xPixelCount, _yPixelCount, _boxSmp, _tmpMap, _imgSmp);
    DecomposeMap(_boxSmp, _decSmpFilters, _decSmp);

    _itrCount++;
}


void Denoiser::SetPixelMeanAndVarianceFinal(const vector<SubPixel> &subPixels,
    BlockedArray<BoxVariance> *variance) {
    
    // Increase the filterbank resolution for the final reconstruction
    GenerateDecFilterBanks(8, sqrtf(2.f), 1.f/sqrtf(2.f), 2.f);

    // Update all buffer and compute filtering map
    SetPixelMeanAndVariance(subPixels, variance);
    GetFilteringMapFinal();

    // Dump everything
//    DumpImageRGB(_imgAvg, "img", DUMP_FINAL);
    GetFilteredImage();
//    DumpImageRGB(_imgFlt, "flt", DUMP_FINAL);
//    DumpMap(_boxSmp, "smp", DUMP_FINAL, 1.f);
//    DumpMap(_imgSel, "map", DUMP_FINAL, 1.f/(_nSteps));

    // Print average sample count
    float nSamples = 0;
    for (int pix = 0; pix < _nPix; pix++)
        nSamples += _boxSmp[pix];
    printf("Average sample count: %.2f\n", nSamples / _nPix);
}


void Denoiser::WriteImage(float* img) {
//    ::WriteImage(_filename, &_imgAvg[0], NULL, _xPixelCount, _yPixelCount,
//                 _xPixelCount, _yPixelCount, 0, 0);

//    memcpy(img, _imgAvg.data(), _nPix*3*sizeof(float));
    memcpy(img, _imgFlt.data(), _nPix*3*sizeof(float));

//    Info("Effective spp: %.3f\n", float(_nSamples) / _xPixelCount / _yPixelCount);
}


void Denoiser::GetFilteredImage() {
    // For each pixel, pick the requested scale
    for (int pix = 0; pix < _nPix; pix++) {
        int scale = _imgSel[pix];

        // Update filtered image buffer
        float *avg = (scale == 0) ? &_imgAvg[3*pix] : &_decAvg[scale-1][3*pix];
        _imgFlt[3*pix+0] = avg[0];
        _imgFlt[3*pix+1] = avg[1];
        _imgFlt[3*pix+2] = avg[2];
    }
}


void Denoiser::FilterSelectionMaps() {
    // Filter and then round the maps
    for (int s = 0; s < _nSteps; s++) {
        for (int iter = 0; iter < 1; iter++) {
            _decSelFilters[s]->Apply(_xPixelCount, _yPixelCount, _decSel[s], _tmpMap, _tmpSel, KERNEL_SKIP_CENTER);
            for (int pix = 0; pix < _nPix; pix++)
                _decSel[s][pix] = floorf(_tmpSel[pix] + .5f);
        }
    }
}


void Denoiser::FilterSelectionMapsFinal() {
    // Filter and then round the maps
    for (int s = 0; s < _nSteps; s++) {
        for (int iter = 0; iter < 1; iter++) {
            _decSelFilters[s]->Apply(_xPixelCount, _yPixelCount, _decSel[s], _tmpMap, _tmpSel, KERNEL_SKIP_CENTER);
            for (int pix = 0; pix < _nPix; pix++)
                _decSel[s][pix] = min(_decSel[s][pix], floorf(_tmpSel[pix] + .5f));
        }
    }
}


void Denoiser::GetWorstPixels(int nPixels, PixelAreaVec &pixelAreas, int spp) {
    // Each pixel MSE is computed at the scale specified by the filtering map
    GetFilteringMap(spp);
    
    nPixels = min(nPixels, _nPix);

    // Randomize pixel order if data is insufficient for a meaningfull sort
    if (nPixels > _nPix - _nVarZero)
        std::random_shuffle(_pixelErrors.begin(), _pixelErrors.end());
    // Partially sort the pixels according to their error
    std::nth_element(_pixelErrors.begin(), _pixelErrors.begin() + nPixels,
        _pixelErrors.end(), Denoiser::SortPixelError);

    // Return the sampling area of the worst pixels
    pixelAreas.resize(nPixels);
    for (int i = 0; i < nPixels; i++) {
        int pix = _pixelErrors[i]._pix;
        int scale = _imgSel[pix];
        // Compute pixel coordinates
        int yPos = pix / _xPixelCount;
        int xPos = pix - yPos * _xPixelCount;
        // Store sampling area
        pixelAreas[i]._pix = pix;
        pixelAreas[i]._xPos = xPos + .5f;
        pixelAreas[i]._yPos = yPos + .5f;
        pixelAreas[i]._scale = scale;
    }

//    if (PbrtOptions.verbose && !PbrtOptions.quiet) {
//        GetFilteredImage();
//        DumpImageRGB(_imgFlt, "flt", DUMP_ITERATION);
//        DumpMap(_boxSmp, "smp", DUMP_ITERATION, 1.f);
//        DumpImageRGB(_imgVar, "var", DUMP_ITERATION);
//        DumpDecomposition(_decVar, "var", DUMP_ITERATION);
//        DumpImageRGB(_imgAvg, "img", DUMP_ITERATION);
//        DumpDecomposition(_decAvg, "img", DUMP_ITERATION);
//        DumpMap(_imgSel, "map", DUMP_ITERATION, 1.f/(_nSteps));
//    }
}


void Denoiser::GetFilteringMapInc() {
    // We check the difference in variance between two scales and the squared
    // difference. As long as the variance difference is lower, we keep on
    // going to coarser scales.

    // Linearize parameter
    float param = -logf(1.f - powf(1.9f * _gamma, 1/sqrtf(2)));

    // Compute scaling factor to compute difference in variance based on the
    // coarser scale variance.
    vector<float> beta(_nSteps);
    for (int s = 0; s < _nSteps; s++) {
        if (s == 0) {
            beta[s] = 1.f;
        }
        else {
            float rcSqr = _scales[s] * _scales[s];
            float rfSqr = _scales[s-1] * _scales[s-1];
            beta[s] = (rcSqr+rfSqr) / (rcSqr-rfSqr);
        }
    }

    // Go through all pixels
    for (int pix = 0; pix < _nPix; pix++) {
        int r = 3*pix+0, g = 3*pix+1, b = 3*pix+2;

        // Initialize this pixel mse
        float mse[3];
        mse[0] = _imgVar[r]; mse[1] = _imgVar[g]; mse[2] = _imgVar[b];

        // Evaluate the validity of each transition
        for (int s = 0; s < _nSteps; s++) {
            // gain: difference of variance
            // loss: difference of squared bias
            // conf: lower for lower sample counts
            float conf, gain[3], loss[3];
            if (s == 0) {
                float nSamples = _imgSmp[pix];
                conf = 1.f - (1.f / nSamples);
                GetGain(&_imgVar[r], &_decVar[s][r], gain);
                GetLoss(&_imgAvg[r], &_decAvg[s][r], 1.f, loss);
            }
            else {
                float nSamples = _decSmp[s][pix] / _decSmpFilters[s]->Norm();
                conf = 1.f - (1.f / nSamples);
                GetGain(&_decVar[s-1][r], &_decVar[s][r], gain);
                GetLoss(&_decAvg[s-1][r], &_decAvg[s][r], beta[s], loss);
            }

            // Compute the overall gamma factor
            float z = param * conf;

            // Substract the scaled loss from the gain
            float diff[3];
            diff[0] = z*loss[0] - gain[0];
            diff[1] = z*loss[1] - gain[1];
            diff[2] = z*loss[2] - gain[2];

            // Flag this transition as invalid if the mse increased
            _decSel[s][pix] = (rgb2avg(diff) >= 0.f) ? 1.f : 0.f;

            // Update the cumulative mse estimate
            mse[0] += loss[0] - gain[0];//diff[0];
            mse[1] += loss[1] - gain[1];//diff[1];
            mse[2] += loss[2] - gain[2];//diff[2];
            _decMse[s][pix] = rgb2avg(mse);
        }
    }
}


void Denoiser::GetFilteringMap(int spp) {
    // Compute the filtering map
    GetFilteringMapInc();

//    if (PbrtOptions.verbose && !PbrtOptions.quiet) {
//        UpdateSelectionMap();
//        DumpDecomposition(_decSel, "sel1", DUMP_ITERATION);
//        DumpMap(_imgSel, "map", DUMP_ITERATION, 1.f/(_nSteps));
//        DumpMap(_boxSmp, "smp", DUMP_ITERATION, 1.f);
//    }

    FilterSelectionMaps();
    
    // Update scale selection map
    UpdateSelectionMap();
    UpdatePixelCosts(spp);

//    if (PbrtOptions.verbose && !PbrtOptions.quiet) {
//        DumpDecomposition(_decSel, "sel2", DUMP_ITERATION);
//        DumpMap(_imgSel, "map", DUMP_ITERATION, 1.f/(_nSteps));
//    }
}


void Denoiser::GetFilteringMapFinal() {
    // Compute the filtering map
    GetFilteringMapInc();

//    if (PbrtOptions.verbose && !PbrtOptions.quiet) {
//        UpdateSelectionMap();
//        DumpDecomposition(_decSel, "sel1", DUMP_FINAL);
//    }

    FilterSelectionMapsFinal();
    UpdateSelectionMap();

//    if (PbrtOptions.verbose && !PbrtOptions.quiet) {
//        DumpDecomposition(_decSel, "sel2", DUMP_FINAL);
//    }
}


void Denoiser::DumpDecomposition(Decomposition &dec, const string &tag, DumpType dumpType) {
//    string base(_filename.begin(), _filename.begin() + _filename.find_last_of("."));

//    char name[256];
//    for (uint32_t s = 0; s < dec.size(); s++) {
//        if (dumpType == DUMP_FINAL)
//            sprintf(name, "%s_%s%d.exr", base.c_str(), tag.c_str(), s);
//        else // (dumpType == DUMP_ITERATION)
//            sprintf(name, "%s_%s%d_itr%03d.exr", base.c_str(), tag.c_str(), s, _itrCount);
//        int nChannels = dec[s].size() / _nPix;
//        if (nChannels == 1) {
//            for (int pix = 0; pix < _nPix; pix++)
//                _tmpImg[3*pix+0] = _tmpImg[3*pix+1] = _tmpImg[3*pix+2] = dec[s][pix];
//            ::WriteImage(name, &_tmpImg[0], NULL, _xPixelCount, _yPixelCount, _xPixelCount, _yPixelCount, 0, 0);
//        }
//        else
//            ::WriteImage(name, &dec[s][0], NULL, _xPixelCount, _yPixelCount, _xPixelCount, _yPixelCount, 0, 0);
//    }
}


void Denoiser::DumpImageRGB(vector<float> &img, const string &tag, DumpType dumpType) {
//    // Retrieve "base" name
//    string base(_filename.begin(), _filename.begin() + _filename.find_last_of("."));

//    // Generate output filename
//    char name[256];
//    if (dumpType == DUMP_FINAL)
//        sprintf(name, "%s_%s.exr", base.c_str(), tag.c_str());
//    else // (dumpType == DUMP_ITERATION)
//        sprintf(name, "%s_%s_itr%03d.exr", base.c_str(), tag.c_str(), _itrCount);

//    // Write to disk
//    ::WriteImage(name, &img[0], NULL, _xPixelCount, _yPixelCount, _xPixelCount, _yPixelCount, 0, 0);
}


void Denoiser::DumpMapInv(vector<float> &map, const string &tag, DumpType dumpType) {
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


void Denoiser::DumpErrorMap(const string &tag, DumpType dumpType) {
    for (uint32_t i = 0; i < _pixelErrors.size(); i++) {
        int pix = _pixelErrors[i]._pix;
        int r = 3*pix+0, g = 3*pix+1, b = 3*pix+2;
        _tmpImg[r] = _tmpImg[g] = _tmpImg[b] = _pixelErrors[i]._error;
    }
    DumpImageRGB(_tmpImg, tag, dumpType);
}


void Denoiser::DumpErrorMap(DumpType dumpType) {
    for (int pix = 0; pix < _nPix; pix++)
        _tmpImg[3*pix+0] = _tmpImg[3*pix+1] = _tmpImg[3*pix+2] = _pixelErrors[pix]._error;
    DumpImageRGB(_tmpImg, "err", dumpType);
}

void Denoiser::FillSubHoles(const vector<SubPixel> &subPixels,
    BlockedArray<BoxVariance> *variance) {
    int xSubPixelCount = _resSub * _xPixelCount;

    // We go over every pixel
#pragma omp parallel for num_threads(std::thread::hardware_concurrency())
    for (int y = 0; y < _yPixelCount; y++) {
        for (int x = 0; x < _xPixelCount; x++) {
            // Get pixel mean
            const BoxVariance &pixvar = (*variance)(x, y);
            float mean[3];
            int n = pixvar.nSamples;
            if (n == 0)
                mean[0] = mean[1] = mean[2] = 0;
            else {
                mean[0] = pixvar.LrgbSum[0] / n;
                mean[1] = pixvar.LrgbSum[1] / n;
                mean[2] = pixvar.LrgbSum[2] / n;
            }
            // We go over every subpixel
            int xSubMin = x * _resSub, xSubMax = xSubMin + _resSub;
            int ySubMin = y * _resSub, ySubMax = ySubMin + _resSub;
            for (int yy = ySubMin; yy < ySubMax; yy++) {
                for (int xx = xSubMin; xx < xSubMax; xx++) {
                    int subPix = xx + yy * xSubPixelCount;
                    float weight = subPixels[subPix].weightSum;

                    if (weight == 0.f) {
                        // Fill holes using mean pixel value
                        _boxSub[3*subPix+0] = mean[0];
                        _boxSub[3*subPix+1] = mean[1];
                        _boxSub[3*subPix+2] = mean[2];
                    }
                    else {
                        // Compute this subpixel value
                        float invW = 1.f / weight;
                        _boxSub[3*subPix+0] = subPixels[subPix].Lxyz[0] * invW;
                        _boxSub[3*subPix+1] = subPixels[subPix].Lxyz[1] * invW;
                        _boxSub[3*subPix+2] = subPixels[subPix].Lxyz[2] * invW;
                    }
                }
            }
        }
    }
}


void Denoiser::GenerateImgFilters(const Filter* filter) {
    // The mean and filters
    _imgFilter = filter;
    _subAvgFilter = new Kernel2D(filter, KERNEL_NORM_UNIT, _resSub);
    _imgVarFilter = Kernel2D::Sub2Var(*_subAvgFilter);
    _imgSmpFilter = new Kernel2D(filter, KERNEL_NORM_STD);
}


void Denoiser::GenerateDecFilterBanks(int nSteps, float scaleFactor,
    float scaleMin, float alpha) {
    // Store parameters
    _nSteps = nSteps;
    _scaleFactor = scaleFactor;
    
    // Compute the scale of each filter
    _scales.clear();
    _scales.resize(_nSteps);
    for (int s = 0; s < _nSteps; s++) {
        _scales[s] = scaleMin * powf(scaleFactor, s);
    }

    // Allocate memory for all buffers
    _decAvg.resize(_nSteps);
    _decVar.resize(_nSteps);
    _decSmp.resize(_nSteps);
    _decMse.resize(_nSteps);
    _decSel.resize(_nSteps);
    for (int s = 0; s < _nSteps; s++) {
        _decAvg[s].resize(3*_nPix);
        _decVar[s].resize(3*_nPix);
        _decSmp[s].resize(_nPix);
        _decMse[s].resize(_nPix);
        _decSel[s].resize(_nPix);
    }

    ResetFilterbank(_decAvgFilters);
    ResetFilterbank(_decVarFilters);
    ResetFilterbank(_decSmpFilters);
    ResetFilterbank(_decSelFilters);

    for (int s = 0; s < _nSteps; s++) {
        // The 'avg' filter
        _decAvgFilters[s] = new Gauss2D(_scales[s], KERNEL_NORM_UNIT);
        // The subpixel scale filter. It is the convolution of the base image
        // filter and the current scale filter.
        Gauss2D subAvgFilters(_scales[s], KERNEL_NORM_UNIT, _resSub);
        subAvgFilters.ConvolveWith(_imgFilter, KERNEL_NORM_UNIT);
        // Corresponding 'var' filter
        _decVarFilters[s] = Kernel2D::Sub2Var(subAvgFilters);
        // Corresponding 'smp' filter, which counts how many samples
        // contributed to a pixel value at each scale
        _decSmpFilters[s] = Kernel2D::Sub2Pix(subAvgFilters, KERNEL_NORM_STD);
        // The 'sel' filter is used to post-process the binary stop maps
        Gauss2D subSelFilter(alpha * _scales[s], KERNEL_NORM_UNIT, _resSub);
        subSelFilter.ConvolveWith(_imgFilter, KERNEL_NORM_UNIT);
        _decSelFilters[s] = Kernel2D::Sub2Pix(subSelFilter, KERNEL_NORM_UNIT);
    }
}


void Denoiser::UpdateSelectionMap() {
    for (int pix = 0; pix < _nPix; pix++) {
        _imgSel[pix] = 0;
        for (int s = 0; s < _nSteps; s++) {
            // If this transition is invalid, we break out
            if (_decSel[s][pix] == 1.f) break;
            // Update the map
            _imgSel[pix] = s+1;
        }
    }
}


void Denoiser::UpdatePixelCosts(int spp) {
    for (int pix = 0; pix < _nPix; pix++) {
        int scale = _imgSel[pix];
        float mse = (scale > 0) ? _decMse[scale-1][pix] : rgb2avg(&_imgVar[3*pix]);
        float error = GetPixelError(pix, scale, mse, spp);
        _pixelErrors[pix] = PixelError(pix, error);
    }
}


