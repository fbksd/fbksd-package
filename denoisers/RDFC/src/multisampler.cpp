/* 
 * File:   multisampler.h
 * Author: Fabrice Rousselle
 *
 * Created on 31. mars 2013, 18:48
 */

#include "multisampler.h"

#include <vector>
#include <limits>
#include <algorithm>

#include "montecarlo.h"


MultiSampler::MultiSampler(int xstart, int xend, int ystart, int yend,
    int spp, float sopen, float sclose, float threshold, int nIterations,
    int sppInit, const MultiFilm *film, bool finalize, bool use_ld_samples)
    : Sampler(xstart, xend, ystart, yend, spp, sopen, sclose),
      use_ld_samples(use_ld_samples),
      xPixelCount(film->GetXPixelCount()),
      yPixelCount(film->GetYPixelCount()),
      _nIterations(nIterations),
      film(const_cast<MultiFilm*> (film)) {
    sppInitReq = sppInit;
    samplesBuf = NULL;
    (this->film)->SetFlagLD(use_ld_samples);
    scrambling.resize(film->GetBufferCount());
    initBase(NULL);
    // Hack variable to skip filtering when we only care about the noisy image
    this->finalize = finalize;
}


// Constructor used to create sub-samplers during the initialization phase. The
// sub-samplers are the one that do the actual job, the 'main' sampler is only
// used to create these.
MultiSampler::MultiSampler(const MultiSampler *parent, int xstart,
    int xend, int ystart, int yend, int pass)
    : Sampler(parent->xPixelStart, parent->xPixelEnd, parent->yPixelStart,
      parent->yPixelEnd, parent->samplesPerPixel, parent->shutterOpen,
      parent->shutterClose),
      use_ld_samples(parent->use_ld_samples),
      xPixelCount(parent->xPixelCount),
      yPixelCount(parent->yPixelCount),
      _nIterations(parent->_nIterations),
      film(parent->film) {

    _xPos = xstart;
    _yPos = ystart;
    _xStartSub = xstart; _xEndSub = xend;
    _yStartSub = ystart; _yEndSub = yend;
    sppInitReq = parent->sppInitReq;
    samplesBuf = NULL;
    film->SetFlagLD(parent->use_ld_samples);
    scrambling.resize(film->GetBufferCount());
    initBase(parent, pass);
}


// Constructor for sub-samplers during the adaptive phase. The sub-samplers do
// the actual jobs, while the "main" sampler only distributes the workload.
// This version of the sampler uses a sampling map to drive the sampling.
MultiSampler::MultiSampler(const MultiSampler *parent, int xstart,
    int xend, int ystart, int yend, ImageBuffer &samplingmap,
    MultiScramblingInfo *scrambling)
    : Sampler(parent->xPixelStart, parent->xPixelEnd, parent->yPixelStart,
      parent->yPixelEnd, parent->samplesPerPixel, parent->shutterOpen,
      parent->shutterClose),
      use_ld_samples(parent->use_ld_samples),
      xPixelCount(parent->xPixelCount),
      yPixelCount(parent->yPixelCount),
      _nIterations(parent->_nIterations),
      film(parent->film) {
    // Update the sampler state
    adaptive = true;
    samplerInit = NULL;
    samplesBuf = NULL;
    film->SetFlagLD(parent->use_ld_samples);
    this->scrambling.resize(film->GetBufferCount());
    this->scrambling[0] = scrambling;
    
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
    this->samplingmap = &samplingmap;
    
    _isMainSampler = false;
    finalize = parent->finalize;
    
    if (parent == NULL) {
        Severe("oups");
    }
}


MultiSampler::~MultiSampler() {
    if (samplerInit != NULL) delete samplerInit;
    if (_isMainSampler) {
        for (size_t i = 0; i < scrambling.size(); i++) {
            delete [] scrambling[i];
        }
        delete samplingmap;
    }
    if (samplesBuf != NULL)
        delete [] samplesBuf;
}


void MultiSampler::initBase(const MultiSampler * parent, int pass) {
    // The MultiSampler distributes the samples over two passes.
    sppInit = samplesPerPixel / scrambling.size();
    if (_nIterations > 0) {
        sppInit = min(Floor2Int(sppInitReq / scrambling.size()), sppInit);
    }
    
    adaptive = false;

    // Construct the sampler for the initialization phase
    if (use_ld_samples) {
        samplerInit = NULL;
    }
    else {
        samplerInit = new RandomSampler(_xStartSub, _xEndSub, _yStartSub, _yEndSub,
            sppInit, shutterOpen, shutterClose);
    }

    // Compute the total number of pixels to be generated
    int nPix = xPixelCount * yPixelCount;
    int nSamplesInit = (scrambling.size() * sppInit) * nPix;
    int nSamplesAdapt = samplesPerPixel * nPix - nSamplesInit;
    pixelsToSampleTotal = Ceil2Int(float(nSamplesAdapt) / samplesPerPixel);

    if (parent != NULL) {
        _xPos = _xStartSub;
        _yPos = _yStartSub;
        _isMainSampler = false;
        finalize = parent->finalize;
        scrambling[0] = parent->scrambling[pass];
    }
    else {
        _xPos = xPixelStart;
        _yPos = yPixelStart;
        _isMainSampler = true;
        int nPixInit = (xPixelEnd-xPixelStart) * (yPixelEnd-yPixelStart);
        for (size_t i = 0; i < scrambling.size(); i++) {
            scrambling[i] = new MultiScramblingInfo[nPixInit];
        }
        samplingmap = new Buffer;
    }
}


Sampler *MultiSampler::GetSubSampler(int num, int count) {
    // The MultiSampler performs the sampling over two passes, one for each
    // destination buffer. The first half of the samplers will correspond to the
    // first pass, while the second half corresponds to the second pass. To
    // enable this, we simply modify the given 'num' and 'count' to cycle twice.
    count = count / scrambling.size();
    int pass = num / count;
    num = num % count;
    if (!adaptive) {
        int x0, x1, y0, y1;
        ComputeSubWindow(num, count, &x0, &x1, &y0, &y1);
        if (x0 == x1 || y0 == y1) return NULL;
        return new MultiSampler(this, x0, x1, y0, y1, pass);
    }
    else {
        // Compute this job's tile
        int x0, x1, y0, y1;
        ComputeSubWindow(num, count, &x0, &x1, &y0, &y1);
        
        // Ensure we don't go outside the sampling map bounds
        x0 = max(x0, 0); x1 = min(x1, xPixelCount);
        y0 = max(y0, 0); y1 = min(y1, yPixelCount);
        
        // Use the appropriate sampling map
        return new MultiSampler(this, x0, x1, y0, y1, *samplingmap, scrambling[pass]);
    }
}


int MultiSampler::GetMoreSamplesMap(float *samples, RNG &rng) {
    // Nothing to do for degenerate patch
    if (_xStartSub == _xEndSub || _yStartSub == _yEndSub)
        return 0;
    
    // During the initialization phase, we generate samples in each pixel using
    // a standard stratified sampler.
    if (!adaptive) {
        return samplerInit->GetMoreSamples(samples, rng);
    }
    
    // Go over the tile until we get some samples
    for ( ; _yPos < _yEndSub; _yPos++) {
        for ( ; _xPos < _xEndSub; _xPos++) {
            // Get requested sample count for current pixel
            int pix = _xPos + _yPos * xPixelCount;
            float req = (*samplingmap)[pix];

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
                samples[i*SAMPLE_SIZE + IMAGE_X] = _xPos + xTmp;
                samples[i*SAMPLE_SIZE + IMAGE_Y] = _yPos + yTmp;
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
                int xPosSmp = Floor2Int(samples[i*SAMPLE_SIZE + IMAGE_X]);
                int yPosSmp = Floor2Int(samples[i*SAMPLE_SIZE + IMAGE_Y]);
//                int xPosSmp = Floor2Int(samples[i].imageX);
//                int yPosSmp = Floor2Int(samples[i].imageY);
                if (xPosSmp >= 0 && yPosSmp >= 0 && xPosSmp < xPixelCount && yPosSmp < yPixelCount) {
                    int pix = (xPosSmp-xPixelStart) + (yPosSmp-yPixelStart) * (xPixelEnd-xPixelStart); // pixel offset
                    if (true) {//_scrambling[pix]._nGenerated < 32) {
                        // Draw the samples
                        MyLDShuffleScrambled2D(1, 1, scrambling[0][pix]._nGenerated, buffer, rng, scrambling[0][pix]._image);
                        samples[i*SAMPLE_SIZE + IMAGE_X] = xPosSmp + buffer[0];
                        samples[i*SAMPLE_SIZE + IMAGE_Y] = yPosSmp + buffer[1];
//                        samples[i].imageX = xPosSmp + buffer[0];
//                        samples[i].imageY = yPosSmp + buffer[1];
                        scrambling[0][pix]._nGenerated++;
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

int MultiSampler::GetMoreSamplesMapLD(float *samples, RNG &rng) {
    // Nothing to do for degenerate patch
    if (_xStartSub == _xEndSub || _yStartSub == _yEndSub)
        return 0;
    
    // During the initialization phase, we drawn samples one by one
    if (!adaptive) {
        // Move to the next line of the tile if the current is done
        if (_xPos == _xEndSub) {
            _xPos = _xStartSub;
            _yPos++;
        }
        
        // Stop if we processed all lines of the tile
        if (_yPos == _yEndSub)
            return 0;
        
        // Allocate the "samples buffer" needed for low-discrepancy sampling
        if (samplesBuf == NULL)
            samplesBuf = new float[LDPixelSampleFloatsNeeded(samples, sppInit)];
        
        // Draw the samples
        MyLDPixelSample(_xPos, _yPos, shutterOpen, shutterClose, sppInit, samples, rng, scrambling[0]);
        _xPos++;
        
        return sppInit;
    }
    
    // Allocate the "samples buffer" needed for low-discrepancy sampling
    if (samplesBuf == NULL)
        samplesBuf = new float[LDPixelSampleFloatsNeeded(samples, samplesPerPixel)];
    
    // Go over the tile until we get some samples
    for ( ; _yPos < _yEndSub; _yPos++) {
        for ( ; _xPos < _xEndSub; _xPos++) {
            // Get requested sample count for current pixel
            int pix = _xPos + _yPos * xPixelCount;
            float req = (*samplingmap)[pix];

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
            MyLDPixelSample(_xPos, _yPos, shutterOpen, shutterClose, nSamples, samples, rng, scrambling[0]);

            // Move to next pixel and return sample count
            _xPos++;
            return nSamples;
        }
        _xPos = _xStartSub;
    }
    
    return 0;
}

//Sampler *CreateMultiSampler(const ParamSet &params,
//                                const Film *film, const Camera *camera) {
//    int spp = params.FindOneInt("pixelsamples", 32);
//    float th = params.FindOneFloat("threshold", std::numeric_limits<float>::infinity());
//    int nIterations = params.FindOneInt("niterations", 2);
//    int sppInit = params.FindOneInt("pixelsamplesinit", std::max(spp/(1+nIterations), 1));

//    int xstart, xend, ystart, yend;
//    film->GetSampleExtent(&xstart, &xend, &ystart, &yend);

//    // Ensure we have a 'SmoothFilm'
//    const MultiFilm *multiFilm = dynamic_cast<const MultiFilm *> (film);
//    if (multiFilm == NULL) {
//        Error("CreateMultiSampler(): film is not of type 'MultiFilm'");
//        return NULL;
//    }
    
//    bool finalize = params.FindOneBool("finalize", true);
    
//    // Low-discrepancy
//    bool use_ld_samples = params.FindOneBool("use_ld_samples", true);
    
//    // Output the sampler parameters
//    Info("CreateMultiSampler:\n");
//    Info("   pixelsamples.....: %d\n", spp);
//    Info("   pixelsamplesinit.: %d\n", sppInit);
//    Info("   niterations......: %d\n", nIterations);
//    Info("   finalize.........: %s\n", finalize ? "true" : "false");
//    Info("   use_ld_samples...: %s\n", use_ld_samples ? "true" : "false");

//    return new MultiSampler(xstart, xend, ystart, yend, spp, camera->shutterOpen,
//        camera->shutterClose, th, nIterations, sppInit, multiFilm, finalize,
//            use_ld_samples);
//}


void MultiSampler::MyLDPixelSample(int xPos, int yPos, float shutterOpen,
    float shutterClose, int nPixelSamples, float *samples, RNG &rng,
    MultiScramblingInfo *scramblingArray) {
    // Prepare temporary array pointers for low-discrepancy camera samples
    float *buf = samplesBuf;
    float *imageSamples = buf; buf += 2 * nPixelSamples;
//    float *lensSamples = buf;  buf += 2 * nPixelSamples;
//    float *timeSamples = buf;  buf += nPixelSamples;

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
    MultiScramblingInfo &scrambling = scramblingArray[pix];
    
    // Define scrambling seeds on first call
    if (scrambling._nGenerated == 0) {
        int nSeeds = 2;
//        int nSeeds = 5 + count1D + 2 * count2D;
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
        samples[i*SAMPLE_SIZE + IMAGE_X] = xPos + imageSamples[2*i];
        samples[i*SAMPLE_SIZE + IMAGE_Y] = yPos + imageSamples[2*i+1];
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
//        samples[i].iX = xPos;
//        samples[i].iY = yPos;
    }
    
    scrambling._nGenerated += nPixelSamples;
}

