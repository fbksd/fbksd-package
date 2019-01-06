/* 
 * File:   DualSampler.cpp
 * Author: rousselle
 * 
 * Created on March 16, 2012, 3:42 PM
 */

#include "dualsampler.h"

#include <vector>
#include <limits>
#include <algorithm>

#include "montecarlo.h"

DualSampler::DualSampler(int xstart, int xend, int ystart, int yend,
    int spp, float sopen, float sclose, float threshold, int nIterations,
    int sppInit, const DualFilm *film)
    : Sampler(xstart, xend, ystart, yend, spp, sopen, sclose),
      _xPixelCount(film->GetXPixelCount()),
      _yPixelCount(film->GetYPixelCount()),
      _nIterations(nIterations),
      _film(film) {
    _sppInitReq = sppInit;
    _samplesBuf = NULL;
    initBase(NULL);
}


// Constructor used to create sub-samplers during the initialization phase. The
// sub-samplers are the one that do the actual job, the 'main' sampler is only
// used to create these.
DualSampler::DualSampler(const DualSampler *parent, int xstart,
    int xend, int ystart, int yend, bool firstPass)
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
    _sppInitReq = parent->_sppInitReq;
    _samplesBuf = NULL;
    initBase(parent, firstPass);
}


// Constructor for sub-samplers during the adaptive phase. The sub-samplers do
// the actual jobs, while the "main" sampler only distributes the workload.
// This version of the sampler uses a sampling map to drive the sampling.
DualSampler::DualSampler(const DualSampler *parent, int xstart,
    int xend, int ystart, int yend, ImageBuffer &samplingMap,
    NlmeansScramblingInfo *scrambling)
    : Sampler(parent->xPixelStart, parent->xPixelEnd, parent->yPixelStart,
      parent->yPixelEnd, parent->samplesPerPixel, parent->shutterOpen,
      parent->shutterClose),
      _xPixelCount(parent->_xPixelCount),
      _yPixelCount(parent->_yPixelCount),
      _nIterations(parent->_nIterations),
      _film(parent->_film) {
    // Update the sampler state
    _adaptive = true;
    _samplerInit = NULL;
    _samplesBuf = NULL;
    _scramblingA = scrambling;
    
    // Each sub-sampler is tasked to sample a specific tile of the whole image.
    _xPos = xstart;
    _yPos = ystart;
    _xStartSub = xstart; _xEndSub = xend;
    _yStartSub = ystart; _yEndSub = yend;
    
    // We accumulate the sampling error: we can only pick an integer count of
    // samples, but the map is in floating points. By accumulating the rounding
    // error, we can propagate the error throughout each tile.
    _sppErr = 0.f;
    
    // Keep a pointer to the sampling map. It will be freed by the main sampler.
    _samplingMapA = &samplingMap;
    
    _isMainSampler = false;
    
    if (parent == NULL) {
        Severe("oups");
    }
}


DualSampler::~DualSampler() {
    if (_samplerInit != NULL) delete _samplerInit;
    if (_isMainSampler) {
        delete [] _scramblingA;
        delete [] _scramblingB;
        delete _samplingMapA;
        delete _samplingMapB;
    }
    if (_samplesBuf != NULL)
        delete [] _samplesBuf;
}


void DualSampler::initBase(const DualSampler * parent, bool firstPass) {
    // The DualSampler distributes the samples over two passes.
    int sppInitA = samplesPerPixel / 2;
    int sppInitB = samplesPerPixel - sppInitA;
    if (_nIterations > 0) {
        sppInitA = min(Floor2Int(.5f*_sppInitReq), sppInitA);
        sppInitB = min(Ceil2Int(.5f*_sppInitReq), sppInitB);
    }
    
    _sppInit = firstPass ? sppInitA : sppInitB;

    _adaptive = false;

//    // Construct the sampler for the initialization phase
#ifdef LD_SAMPLING
    _samplerInit = NULL;
#else
    _samplerInit = new RandomSampler(_xStartSub, _xEndSub, _yStartSub, _yEndSub,
        _sppInit, shutterOpen, shutterClose);
#endif

    // Compute the total number of pixels to be generated
    int nPix = _xPixelCount * _yPixelCount;
    int nSamplesInit = (sppInitA + sppInitB) * nPix;
    int nSamplesAdapt = samplesPerPixel * nPix - nSamplesInit;
    _pixelsToSampleTotal = Ceil2Int(float(nSamplesAdapt) / samplesPerPixel);

    if (parent != NULL) {
        _xPos = _xStartSub;
        _yPos = _yStartSub;
        _isMainSampler = false;
        _scramblingA = (firstPass) ? parent->_scramblingA : parent->_scramblingB;
    }
    else {
        _xPos = xPixelStart;
        _yPos = yPixelStart;
        _isMainSampler = true;
        int nPixInit = (xPixelEnd-xPixelStart) * (yPixelEnd-yPixelStart);
        _scramblingA = new NlmeansScramblingInfo[nPixInit];
        _scramblingB = new NlmeansScramblingInfo[nPixInit];
        _samplingMapA = new ImageBuffer();
        _samplingMapB = new ImageBuffer();
    }
}


Sampler *DualSampler::GetSubSampler(int num, int count) {
    // The DualSampler performs the sampling over two passes, one for each
    // destination buffer. The first half of the samplers will correspond to the
    // first pass, while the second half corresponds to the second pass. To
    // enable this, we simply modify the given 'num' and 'count' to cycle twice.
    count = count / 2;
    bool firstPass = num < count;
    num = num % count;
    if (!_adaptive) {
        int x0, x1, y0, y1;
        ComputeSubWindow(num, count, &x0, &x1, &y0, &y1);
        if (x0 == x1 || y0 == y1) return NULL;
        return new DualSampler(this, x0, x1, y0, y1, firstPass);
    }
    else {
        // Compute this job's tile
        int x0, x1, y0, y1;
        ComputeSubWindow(num, count, &x0, &x1, &y0, &y1);
        
        // Ensure we don't go outside the sampling map bounds
        x0 = max(x0, 0); x1 = min(x1, _xPixelCount);
        y0 = max(y0, 0); y1 = min(y1, _yPixelCount);
        
        // Use the appropriate sampling map
        if (firstPass)
            return new DualSampler(this, x0, x1, y0, y1, *_samplingMapA, _scramblingA);
        else
            return new DualSampler(this, x0, x1, y0, y1, *_samplingMapB, _scramblingB);
    }
}


int DualSampler::GetMoreSamplesMap(float *samples, RNG &rng) {
    // Nothing to do for degenerate patch
    if (_xStartSub == _xEndSub || _yStartSub == _yEndSub)
        return 0;
    
    // During the initialization phase, we generate samples in each pixel using
    // a standard stratified sampler.
    if (!_adaptive) {
        // Draw a set of random samples
        int nSamples = _samplerInit->GetMoreSamples(samples, rng);

        // Replace image samples with low-discrepancy samples inside film buffer
        for (int i = 0; i < nSamples; i++) {
            int xPos = samples[i*5];
            int yPos = samples[i*5 + 1];
            if (xPos >= 0 && yPos >= 0 && xPos < _xPixelCount && yPos < _yPixelCount) {
                int pix = (xPos-xPixelStart) + (yPos-yPixelStart) * (xPixelEnd-xPixelStart); // pixel offset
                // Seed low-discrepancy scrambling for this pixel
                if (_scramblingA[pix]._nGenerated == 0) {
                    int nSeeds = 2;
                    _scramblingA[pix]._seeds.resize(nSeeds);
                    for (int i = 0; i < nSeeds; i++)
                        _scramblingA[pix]._seeds[i] = rng.RandomUInt();
                    _scramblingA[pix]._image = &_scramblingA[pix]._seeds[0];
                }
                // Draw scrambled low-discrepancy samples
                float buffer[2]; // 2 floats per image sampleiv -
                MyLDShuffleScrambled2D(1, 1, _scramblingA[pix]._nGenerated, &buffer[0], rng, _scramblingA[pix]._image);
                _scramblingA[pix]._nGenerated++;
                samples[i*5] = xPos + buffer[0];
                samples[i*5 + 1] = yPos + buffer[1];
            }
        }
        return nSamples;
    }
    
    // Go over the tile until we get some samples
    for ( ; _yPos < _yEndSub; _yPos++) {
        for ( ; _xPos < _xEndSub; _xPos++) {
            // Get requested sample count for current pixel
            int pix = _xPos + _yPos * _xPixelCount;
            float req = (*_samplingMapA)[pix];

            // Factor in the half-toning error from the previous pixel
            req -= _sppErr;

            // Convert the float request into an integer sample count, we use the
            // residual as the probability of drawing an additional sample
            int nSamples = Float2Int(req);
            if (rng.RandomFloat() < req-nSamples)
                nSamples++;
            
            // Update the rounding error
            nSamples = min(samplesPerPixel, nSamples);
            _sppErr = nSamples - req;
            
            // Skip this pixel if there's no sample to draw
            if (nSamples <= 0)
                continue;

            // Temporary buffer to hold low-discrepancy values
            float buffer[2]; // 2 floats per image sample

            // Get samples
            for (int i = 0; i < nSamples; i++) {
                // Importance sampling
                float xTmp = rng.RandomFloat(), yTmp = rng.RandomFloat();
                samples[i*5] = _xPos + xTmp;
                samples[i*5 + 1] = _yPos + yTmp;
//                samples[i].imageX = _xPos + xTmp;
//                samples[i].imageY = _yPos + yTmp;
//                samples[i].lensU = rng.RandomFloat();
//                samples[i].lensV = rng.RandomFloat();
//                samples[i].time = Lerp(rng.RandomFloat(), shutterOpen, shutterClose);
//                // Generate random samples for integrators
//                for (uint32_t j = 0; j < samples[i].n1D.size(); ++j)
//                    for (uint32_t k = 0; k < samples[i].n1D[j]; ++k) {
//                        samples[i].oneD[j][k] = rng.RandomFloat();
//                    }
//                for (uint32_t j = 0; j < samples[i].n2D.size(); ++j)
//                    for (uint32_t k = 0; k < 2*samples[i].n2D[j]; ++k)
//                        samples[i].twoD[j][k] = rng.RandomFloat();

                // Replace image samples with low-discrepancy samples inside film buffer
                int xPosSmp = Floor2Int(samples[i*5]);
                int yPosSmp = Floor2Int(samples[i*5 + 1]);
                if (xPosSmp >= 0 && yPosSmp >= 0 && xPosSmp < _xPixelCount && yPosSmp < _yPixelCount) {
                    int pix = (xPosSmp-xPixelStart) + (yPosSmp-yPixelStart) * (xPixelEnd-xPixelStart); // pixel offset
                    if (true) {//_scrambling[pix]._nGenerated < 32) {
                        // Draw the samples
                        MyLDShuffleScrambled2D(1, 1, _scramblingA[pix]._nGenerated, buffer, rng, _scramblingA[pix]._image);
                        samples[i*5] = xPosSmp + buffer[0];
                        samples[i*5 + 1] = yPosSmp + buffer[1];
                        _scramblingA[pix]._nGenerated++;
                    }
                }
            }

            // Move to next pixel and return sample count
            _xPos++;
            return nSamples;
        }
        _xPos = _xStartSub;
    }
    
    return 0;
}


int DualSampler::GetMoreSamplesMapLD(float *samples, RNG &rng) {
    // Nothing to do for degenerate patch
    if (_xStartSub == _xEndSub || _yStartSub == _yEndSub)
        return 0;
    
    // During the initialization phase, we drawn samples one by one
    if (!_adaptive) {
        // Move to the next line of the tile if the current is done
        if (_xPos == _xEndSub) {
            _xPos = _xStartSub;
            _yPos++;
        }
        
        // Stop if we processed all lines of the tile
        if (_yPos == _yEndSub)
            return 0;
        
        // Allocate the "samples buffer" needed for low-discrepancy sampling
        if (_samplesBuf == NULL)
            _samplesBuf = new float[_sppInit*2];
//            _samplesBuf = new float[LDPixelSampleFloatsNeeded(samples, _sppInit)];
        
//        LDPixelSample(_xPos, _yPos, shutterOpen, shutterClose,
//                    _sppInit, samples, _samplesBuf, rng);
        // Draw the samples
        MyLDPixelSample(_xPos, _yPos, shutterOpen, shutterClose, _sppInit, samples, rng, _scramblingA);
        _xPos++;
        
        return _sppInit;
    }
    
    // Allocate the "samples buffer" needed for low-discrepancy sampling
    if (_samplesBuf == NULL)
        _samplesBuf = new float[samplesPerPixel*2];
//        _samplesBuf = new float[LDPixelSampleFloatsNeeded(samples, samplesPerPixel)];
    
    // Go over the tile until we get some samples
    for ( ; _yPos < _yEndSub; _yPos++) {
        for ( ; _xPos < _xEndSub; _xPos++) {
            // Get requested sample count for current pixel
            int pix = _xPos + _yPos * _xPixelCount;
            float req = (*_samplingMapA)[pix];

            // Factor in the half-toning error from the previous pixel
            req -= _sppErr;

            // Convert the float request into an integer sample count, we use the
            // residual as the probability of drawing an additional sample
            int nSamples = Float2Int(req);
            if (rng.RandomFloat() < req-nSamples)
                nSamples++;
            
            // Update the rounding error
            nSamples = min(samplesPerPixel, nSamples);
            _sppErr = nSamples - req;
            
            // Skip this pixel if there's no sample to draw
            if (nSamples <= 0)
                continue;
            
            // Draw the samples
            MyLDPixelSample(_xPos, _yPos, shutterOpen, shutterClose, nSamples, samples, rng, _scramblingA);

            // Move to next pixel and return sample count
            _xPos++;
            return nSamples;
        }
        _xPos = _xStartSub;
    }
    
    return 0;
}


//Sampler *CreateDualSampler(const ParamSet &params,
//                                const Film *film, const Camera *camera) {
//    int spp = params.FindOneInt("pixelsamples", 32);
//    int sppInit = params.FindOneInt("pixelsamplesinit", spp/2);
//    float th = params.FindOneFloat("threshold", std::numeric_limits<float>::infinity());
//    // By default we update 5% of the image on each iteration
//    int nIterations = params.FindOneInt("niterations", 8);

//    int xstart, xend, ystart, yend;
//    film->GetSampleExtent(&xstart, &xend, &ystart, &yend);

//    // Ensure we have a 'SmoothFilm'
//    const DualFilm *dualFilm = dynamic_cast<const DualFilm *> (film);
//    if (dualFilm == NULL) {
//        Error("CreateDualSampler(): film is not of type 'DualFilm'");
//        return NULL;
//    }
    
//    // Output the sampler parameters
//    Info("CreateDualSampler:\n");
//    Info("   pixelsamples.....: %d\n", spp);
//    Info("   pixelsamplesinit.: %d\n", sppInit);
//    Info("   niterations......: %d\n", nIterations);

//    return new DualSampler(xstart, xend, ystart, yend, spp, camera->shutterOpen,
//        camera->shutterClose, th, nIterations, sppInit, dualFilm);
//}


void DualSampler::MyLDPixelSample(int xPos, int yPos, float shutterOpen,
    float shutterClose, int nPixelSamples, float *samples, RNG &rng,
    NlmeansScramblingInfo *scramblingArray) {
    // Prepare temporary array pointers for low-discrepancy camera samples
    float *buf = _samplesBuf;
    float *imageSamples = buf; buf += 2 * nPixelSamples;
    float *lensSamples = buf;  buf += 2 * nPixelSamples;
    float *timeSamples = buf;  buf += nPixelSamples;

    // Prepare temporary array pointers for low-discrepancy integrator samples
//    uint32_t count1D = samples[0].n1D.size();
//    uint32_t count2D = samples[0].n2D.size();
//    const uint32_t *n1D = count1D > 0 ? &samples[0].n1D[0] : NULL;
//    const uint32_t *n2D = count2D > 0 ? &samples[0].n2D[0] : NULL;
//    float **oneDSamples = ALLOCA(float *, count1D);
//    float **twoDSamples = ALLOCA(float *, count2D);
//    for (uint32_t i = 0; i < count1D; ++i) {
//        oneDSamples[i] = buf;
//        buf += n1D[i] * nPixelSamples;
//    }
//    for (uint32_t i = 0; i < count2D; ++i) {
//        twoDSamples[i] = buf;
//        buf += 2 * n2D[i] * nPixelSamples;
//    }
    
    // Get reference to the current pixel's scrambling info
    int pix = (xPos-xPixelStart) + (yPos-yPixelStart) * (xPixelEnd-xPixelStart);
    NlmeansScramblingInfo &scrambling = scramblingArray[pix];
    
    // Define scrambling seeds on first call
    if (scrambling._nGenerated == 0) {
//        int nSeeds = 5 + count1D + 2 * count2D;
        int nSeeds = 2;
        scrambling._seeds.resize(nSeeds);
        for (int i = 0; i < nSeeds; i++)
            scrambling._seeds[i] = rng.RandomUInt();
        scrambling._image = &scrambling._seeds[0];
//        scrambling._lens = scrambling._image + 2;
//        scrambling._time = scrambling._lens  + 2;
//        scrambling._oneD = scrambling._time  + 1;
//        scrambling._twoD = scrambling._oneD  + count1D;
    }
    
    // Generate low-discrepancy pixel samples
    MyLDShuffleScrambled2D(1, nPixelSamples, scrambling._nGenerated, imageSamples, rng, scrambling._image);
//    MyLDShuffleScrambled2D(1, nPixelSamples, scrambling._nGenerated, lensSamples,  rng, scrambling._lens);
//    MyLDShuffleScrambled1D(1, nPixelSamples, scrambling._nGenerated, timeSamples,  rng, *scrambling._time);
//    for (uint32_t i = 0; i < count1D; ++i)
//        MyLDShuffleScrambled1D(n1D[i], nPixelSamples, scrambling._nGenerated, oneDSamples[i], rng, scrambling._oneD[i]);
//    for (uint32_t i = 0; i < count2D; ++i)
//        MyLDShuffleScrambled2D(n2D[i], nPixelSamples, scrambling._nGenerated, twoDSamples[i], rng, &scrambling._twoD[2*i]);

    // Initialize _samples_ with computed sample values
    for (int i = 0; i < nPixelSamples; ++i) {
        samples[i*5] = xPos + imageSamples[2*i];
        samples[i*5 + 1] = yPos + imageSamples[2*i+1];
//        samples[i].time = Lerp(timeSamples[i], shutterOpen, shutterClose);
//        samples[i].lensU = lensSamples[2*i];
//        samples[i].lensV = lensSamples[2*i+1];
//        // Copy integrator samples into _samples[i]_
//        for (uint32_t j = 0; j < count1D; ++j) {
//            int startSamp = n1D[j] * i;
//            for (uint32_t k = 0; k < n1D[j]; ++k)
//                samples[i].oneD[j][k] = oneDSamples[j][startSamp+k];
//        }
//        for (uint32_t j = 0; j < count2D; ++j) {
//            int startSamp = 2 * n2D[j] * i;
//            for (uint32_t k = 0; k < 2*n2D[j]; ++k)
//                samples[i].twoD[j][k] = twoDSamples[j][startSamp+k];
//        }
    }
    
    scrambling._nGenerated += nPixelSamples;
}


void DualSampler::MyLDPixelSampleInterleaved(int xPos, int yPos, float shutterOpen,
    float shutterClose, int nPixelSamples, float *samples, RNG &rng) {
    // Prepare temporary array pointers for low-discrepancy camera samples
    float *buf = _samplesBuf;
    float *imageSamples = buf; buf += 2 * nPixelSamples;
    float *lensSamples = buf;  buf += 2 * nPixelSamples;
    float *timeSamples = buf;  buf += nPixelSamples;

    // Prepare temporary array pointers for low-discrepancy integrator samples
//    uint32_t count1D = samples[0].n1D.size();
//    uint32_t count2D = samples[0].n2D.size();
//    const uint32_t *n1D = count1D > 0 ? &samples[0].n1D[0] : NULL;
//    const uint32_t *n2D = count2D > 0 ? &samples[0].n2D[0] : NULL;
//    float **oneDSamples = ALLOCA(float *, count1D);
//    float **twoDSamples = ALLOCA(float *, count2D);
//    for (uint32_t i = 0; i < count1D; ++i) {
//        oneDSamples[i] = buf;
//        buf += n1D[i] * nPixelSamples;
//    }
//    for (uint32_t i = 0; i < count2D; ++i) {
//        twoDSamples[i] = buf;
//        buf += 2 * n2D[i] * nPixelSamples;
//    }
    
    // Get reference to the current pixel's scrambling info
    int pix = (xPos-xPixelStart) + (yPos-yPixelStart) * (xPixelEnd-xPixelStart);
    
    // Define scrambling seeds on first call
    if (_scramblingA[pix]._nGenerated == 0) {
//        int nSeeds = 5 + count1D + 2 * count2D;
        int nSeeds = 2;
        _scramblingA[pix]._seeds.resize(nSeeds);
        _scramblingB[pix]._seeds.resize(nSeeds);
        for (int i = 0; i < nSeeds; i++) {
            _scramblingA[pix]._seeds[i] = rng.RandomUInt();
            _scramblingB[pix]._seeds[i] = rng.RandomUInt();
        }
        _scramblingA[pix]._image = &_scramblingA[pix]._seeds[0];
        _scramblingB[pix]._image = &_scramblingB[pix]._seeds[0];
//        _scramblingA[pix]._lens = _scramblingA[pix]._image + 2;
//        _scramblingB[pix]._lens = _scramblingB[pix]._image + 2;
//        _scramblingA[pix]._time = _scramblingA[pix]._lens  + 2;
//        _scramblingB[pix]._time = _scramblingB[pix]._lens  + 2;
//        _scramblingA[pix]._oneD = _scramblingA[pix]._time  + 1;
//        _scramblingB[pix]._oneD = _scramblingB[pix]._time  + 1;
//        _scramblingA[pix]._twoD = _scramblingA[pix]._oneD  + count1D;
//        _scramblingB[pix]._twoD = _scramblingB[pix]._oneD  + count1D;
    }
    
    // Initialize _samples_ with computed sample values
    for (int i = 0; i < nPixelSamples; ++i) {
        NlmeansScramblingInfo &scrambling = (i%2 == 0) ? _scramblingA[pix] : _scramblingB[pix];
        
        // Generate a low-discrepancy pixel sample for buffer A
        MyLDShuffleScrambled2D(1, 1, scrambling._nGenerated, &imageSamples[2*i], rng, scrambling._image);
//        MyLDShuffleScrambled2D(1, 1, scrambling._nGenerated, &lensSamples[2*i],  rng, scrambling._lens);
//        MyLDShuffleScrambled1D(1, 1, scrambling._nGenerated, &timeSamples[2*i],  rng, *scrambling._time);
//        for (uint32_t j = 0; j < count1D; ++j)
//            MyLDShuffleScrambled1D(n1D[j], 1, scrambling._nGenerated, oneDSamples[j], rng, scrambling._oneD[j]);
//        for (uint32_t j = 0; j < count2D; ++j)
//            MyLDShuffleScrambled2D(n2D[j], 1, scrambling._nGenerated, twoDSamples[j], rng, &scrambling._twoD[2*j]);
        scrambling._nGenerated ++;
        
        // Add current pixel offset
        samples[i*5] = xPos + imageSamples[2*i];
        samples[i*5 + 1] = yPos + imageSamples[2*i+1];
//        samples[i].imageX = xPos + imageSamples[2*i];
//        samples[i].imageY = yPos + imageSamples[2*i+1];
//        samples[i].time = Lerp(timeSamples[i], shutterOpen, shutterClose);
//        samples[i].lensU = lensSamples[2*i];
//        samples[i].lensV = lensSamples[2*i+1];
//        // Copy integrator samples into _samples[i]_
//        for (uint32_t j = 0; j < count1D; ++j) {
//            int startSamp = n1D[j] * i;
//            for (uint32_t k = 0; k < n1D[j]; ++k)
//                samples[i].oneD[j][k] = oneDSamples[j][startSamp+k];
//        }
//        for (uint32_t j = 0; j < count2D; ++j) {
//            int startSamp = 2 * n2D[j] * i;
//            for (uint32_t k = 0; k < 2*n2D[j]; ++k)
//                samples[i].twoD[j][k] = twoDSamples[j][startSamp+k];
//        }
    }
}


