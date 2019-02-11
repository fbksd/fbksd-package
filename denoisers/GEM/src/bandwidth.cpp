
/*
 *  Copyright(c) 2011 Fabrice Rousselle.
 * 
 *  You can redistribute and/or modify this file under the terms of the GNU
 *  General Public License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 */

#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>

#include "bandwidth.h"
#include "montecarlo.h"
#include "denoiser.h"

BandwidthSampler::BandwidthSampler(int xstart, int xend, int ystart, int yend,
    int spp, float sopen, float sclose, float threshold, int nIterations,
    const SmoothFilm *film)
    : Sampler(xstart, xend, ystart, yend, spp, sopen, sclose),
      _xPixelCount(film->GetXPixelCount()),
      _yPixelCount(film->GetYPixelCount()),
      _nIterations(nIterations),
      _film(film) {

    initBase(NULL);
}


BandwidthSampler::BandwidthSampler(const BandwidthSampler *parent, int xstart,
    int xend, int ystart, int yend)
    : Sampler(parent->xPixelStart, parent->xPixelEnd, parent->yPixelStart,
      parent->yPixelEnd, parent->samplesPerPixel, parent->shutterOpen,
      parent->shutterClose),
      _xPixelCount(parent->_xPixelCount),
      _yPixelCount(parent->_yPixelCount),
      _nIterations(parent->_nIterations),
      _film(parent->_film) {

    _xPos = xstart;
    _yPos = ystart;
    _xStartSub = xstart; _xEndSub = xend;
    _yStartSub = ystart; _yEndSub = yend;
    initBase(parent);
}


BandwidthSampler::BandwidthSampler(const BandwidthSampler *parent,
    const PixelAreaVec &pixels, int taskNum, int nPixels)
    : Sampler(parent->xPixelStart, parent->xPixelEnd, parent->yPixelStart,
      parent->yPixelEnd, parent->samplesPerPixel, parent->shutterOpen,
      parent->shutterClose),
      _xPixelCount(parent->_xPixelCount),
      _yPixelCount(parent->_yPixelCount),
      _nIterations(parent->_nIterations),
      _film(parent->_film) {

    initAdapt(parent, pixels, taskNum, nPixels);
}


BandwidthSampler::~BandwidthSampler() {
    if (_samplerInit != NULL) delete _samplerInit;
    if (_isMainSampler) {
        delete [] _scrambling;
        int nPix = _xPixelCount * _yPixelCount;
        for (int pix = 0; pix < nPix; pix++)
            Mutex::Destroy(_pixelMutexes[pix]);
        delete [] _pixelMutexes;
    }
}


void BandwidthSampler::initBase(const BandwidthSampler * parent) {
    // Number of samples for the initialization phase
    if (_nIterations == 0) // non-adaptive
        _sppInit = samplesPerPixel;
    else // adaptive
        _sppInit = min(4, samplesPerPixel);

    _adaptive = false;

    // Construct the sampler for the initialization phase
//    _samplerInit = new RandomSampler(_xStartSub, _xEndSub, _yStartSub, _yEndSub,
//        _sppInit, shutterOpen, shutterClose);
    _samplerInit = nullptr;

    // Compute the total number of pixels to be generated
    int nPix = _xPixelCount * _yPixelCount;
    int nSamplesInit = _sppInit * nPix;
    int nSamplesAdapt = samplesPerPixel * nPix - nSamplesInit;
    _pixelsToSampleTotal = Ceil2Int(float(nSamplesAdapt) / samplesPerPixel);

    if (parent != NULL) {
        _xPos = _xStartSub;
        _yPos = _yStartSub;
        _isMainSampler = false;
        _scrambling = parent->_scrambling;
        _pixelMutexes = parent->_pixelMutexes;
    }
    else {
        _xPos = xPixelStart;
        _yPos = yPixelStart;
        _isMainSampler = true;
        _scrambling = new ScramblingInfo[nPix];
        _pixelMutexes = new Mutex*[nPix];
        for (int pix = 0; pix < nPix; pix++) {
            _pixelMutexes[pix] = Mutex::Create();
        }
    }

    _film->GetFilterbank(_filters);
}


void BandwidthSampler::initAdapt(const BandwidthSampler * parent,
                                 const PixelAreaVec &pixels, int taskNum,
                                 int nPixels) {
    _adaptive = true;
    _samplerInit = NULL;

    int first = taskNum * nPixels;
    int last = min(int(pixels.size()), first + nPixels);
    _pixels.resize(max(0, last-first));
    if (_pixels.size() > 0)
        copy(pixels.begin()+first, pixels.begin()+last, _pixels.begin());
    
    // Compute the total number of pixels to be generated
    int nPix = _xPixelCount * _yPixelCount;
    
    _xPos = xPixelStart;
    _yPos = yPixelStart;
    if (parent != NULL) {
        _isMainSampler = false;
        _scrambling = parent->_scrambling;
        _pixelMutexes = parent->_pixelMutexes;
    }
    else {
        _isMainSampler = true;
        _scrambling = new ScramblingInfo[nPix];
        _pixelMutexes = new Mutex*[nPix];
        for (int pix = 0; pix < nPix; pix++) {
            _pixelMutexes[pix] = Mutex::Create();
        }
    }
    
    _film->GetFilterbank(_filters);
}


Sampler *BandwidthSampler::GetSubSampler(int num, int count) {
    if (!_adaptive) {
        int x0, x1, y0, y1;
        ComputeSubWindow(num, count, &x0, &x1, &y0, &y1);
        if (x0 == x1 || y0 == y1) return NULL;
        return new BandwidthSampler(this, x0, x1, y0, y1);
    }
    else {
        // Compute the number of pixels for this sub-sampler
        int pixelsToSample = PixelsToSample(); // total job
        pixelsToSample = Ceil2Int(float(pixelsToSample) / count); // this job
        // Create sub-sampler
        return new BandwidthSampler(this, _pixels, num, pixelsToSample);
    }
}


int BandwidthSampler::GetMoreSamples(std::vector<float> *samples, RNG& rng) {
    // Check if we're done
    if (_pixels.empty()) return 0;

    // Get this pixel's sampling info and pop it
    float xPos = _pixels.back()._xPos;
    float yPos = _pixels.back()._yPos;
    float scale = _pixels.back()._scale;
    _pixels.pop_back();

    // Temporary buffer to hold low-discrepancy values
    float buffer[2]; // 2 floats per image sample

    // Get samples
    for (int i = 0; i < samplesPerPixel; i++) {
        // Importance sampling
        float xTmp = rng.RandomFloat(), yTmp = rng.RandomFloat();
        _filters[scale]->WarpSampleToPixelOffset(xTmp, yTmp);
        float sx = xPos + xTmp;
        float sy = yPos + yTmp;
        // Replace image samples with low-discrepancy samples inside film buffer
        int xPosSmp = Floor2Int(sx);
        int yPosSmp = Floor2Int(sy);
        if (xPosSmp >= 0 && yPosSmp >= 0 && xPosSmp < _xPixelCount && yPosSmp < _yPixelCount) {
            int pix = xPosSmp + yPosSmp * _xPixelCount; // pixel offset
            // Lock this pixel
            MutexLock lock(*(_pixelMutexes[pix]));
            // Draw the samples
            LDShuffleScrambled2D(1, 1, _scrambling[pix]._nGenerated, buffer, rng, _scrambling[pix]._image);
            sx = xPosSmp + buffer[0];
            sy = yPosSmp + buffer[1];
            _scrambling[pix]._nGenerated++;
            samples->push_back(sx);
            samples->push_back(sy);
        }
        else
        {
            // WORKAROUND: samples outside of film cause problems. Try another sample.
            --i;
        }
    }

    // DEBUG
//    for (int i = 0; i < samplesPerPixel; i++) {
//        if(samples[i*5] > _xPixelCount || samples[i*5 + 1] > _yPixelCount)
//            std::cout << "ERROR: out of range adaptive sample generated [" << samples[i*5] << ", " << samples[i*5 + 1] << "]" << std::endl;
//    }

    return samplesPerPixel;
}


//Sampler *CreateBandwidthSampler(const ParamSet &params,
//                                const Film *film, const Camera *camera) {
//    int ns = params.FindOneInt("pixelsamples", 32);
//    float th = params.FindOneFloat("threshold", std::numeric_limits<float>::infinity());
//    // By default we update 5% of the image on each iteration
//    int nIterations = params.FindOneInt("niterations", 8);

//    int xstart, xend, ystart, yend;
//    film->GetSampleExtent(&xstart, &xend, &ystart, &yend);

//    // Ensure we have a 'SmoothFilm'
//    const SmoothFilm *smoothFilm = dynamic_cast<const SmoothFilm *> (film);
//    if (smoothFilm == NULL) {
//        Error("CreateBandwidthSampler(): film is not of type 'SmoothFilm'");
//        return NULL;
//    }

//    // Output the sampler parameters
//    Info("CreateBandwidthSampler:\n");
//    Info("   pixelsamples.....: %d\n", ns);
//    Info("   niterations......: %d\n", nIterations);

//    return new BandwidthSampler(xstart, xend, ystart, yend, ns,
//                                camera->shutterOpen, camera->shutterClose, th,
//                                nIterations, smoothFilm);
//}


