
/*
    Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.    

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */



// samplers/sbfsampler.cpp*
#include "stdafx.h"
#include "sbfsampler.h"
#include "montecarlo.h"
#include "SBFCommon.h"

#include <iostream>

// A modified LD samples generator that accepts an "offset"
inline void LDShuffleScrambled1D(int offset, int nSamples, int nPixel,
                                 float *samples, RNG &rng) {
    uint32_t scramble = rng.RandomUInt();
    for (int i = 0; i < nSamples * nPixel; ++i)
        samples[i] = VanDerCorput(offset+i, scramble);
    for (int i = 0; i < nPixel; ++i)
        Shuffle(samples + i * nSamples, nSamples, 1, rng);
    Shuffle(samples, nPixel, nSamples, rng);
}


inline void LDShuffleScrambled2D(int offset, int nSamples, int nPixel,
                                 float *samples, RNG &rng) {
    uint32_t scramble[2] = { (uint32_t)rng.RandomUInt(), (uint32_t)rng.RandomUInt() };
    for (int i = 0; i < nSamples * nPixel; ++i)
        Sample02(offset+i, scramble, &samples[2*i]);
    for (int i = 0; i < nPixel; ++i)
        Shuffle(samples + 2 * i * nSamples, nSamples, 2, rng);
    Shuffle(samples, nPixel, 2 * nSamples, rng);
}

void LDPixelSample(int offset, int xPos, int yPos, float shutterOpen,
        float shutterClose, int nPixelSamples, std::vector<float>* samples,
        float *buf, RNG &rng) {
    // Prepare temporary array pointers for low-discrepancy camera samples
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

    // Generate low-discrepancy pixel samples
    LDShuffleScrambled2D(offset, 1, nPixelSamples, imageSamples, rng);
//    LDShuffleScrambled2D(offset, 1, nPixelSamples, lensSamples, rng);
//    LDShuffleScrambled1D(offset, 1, nPixelSamples, timeSamples, rng);
//    for (uint32_t i = 0; i < count1D; ++i)
//        LDShuffleScrambled1D(offset, n1D[i], nPixelSamples, oneDSamples[i], rng);
//    for (uint32_t i = 0; i < count2D; ++i)
//        LDShuffleScrambled2D(offset, n2D[i], nPixelSamples, twoDSamples[i], rng);

    // Initialize _samples_ with computed sample values
    for (int i = 0; i < nPixelSamples; ++i) {
        samples->push_back(xPos + imageSamples[2 * i]);
        samples->push_back(yPos + imageSamples[2*i+1]);
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

SBFSampler::SBFSampler(int xstart, int xend,
        int ystart, int yend, float as, float sopen, float sclose, 
        int is, int ms, int it,
        vector<vector<int> > *pixoff, 
        vector<vector<int> > *pixsmp, 
        int bxs, int bys)
    : Sampler(xstart, xend, ystart, yend, Ceil2Int(is+it*as), sopen, sclose) {
    xPos = xPixelStart;
    yPos = yPixelStart;    
    initSamples = is;
    adaptiveSamples = as;
    maxSamples = ms;
    iteration = it;
    
    pixelOffset = pixoff;
    pixelSampleCount = pixsmp;
    sampleBuf = NULL;

    baseXStart = bxs != -1 ? bxs : xPixelStart;
    baseYStart = bys != -1 ? bys : yPixelStart;
}

Sampler *SBFSampler::GetSubSampler(int num, int count) {    
    int x0, x1, y0, y1;
    ComputeSubWindow(num, count, &x0, &x1, &y0, &y1);
    if (x0 == x1 || y0 == y1) return NULL;
    return new SBFSampler(x0, x1, 
        y0, y1, samplesPerPixel, shutterOpen, shutterClose,
        initSamples, maxSamples, iteration, 
        pixelOffset, pixelSampleCount, 
        baseXStart, baseYStart);
}

int SBFSampler::GetMoreSamples(std::vector<float> *samples, RNG &rng) {
    if (yPos == yPixelEnd) return 0;
    
    int spp = pixelSampleCount ?
        min(((*pixelSampleCount)[yPos-baseYStart][xPos-baseXStart]), maxSamples) : 
        initSamples;
    if(!sampleBuf) {
        sampleBuf = new float[LDPixelSampleFloatsNeeded(nullptr, maxSamples)];
    }

    LDPixelSample(pixelOffset ? (*pixelOffset)[yPos-baseYStart][xPos-baseXStart] : 0, xPos, yPos, 
            shutterOpen, shutterClose, spp, samples, sampleBuf, rng);
 
    if (++xPos == xPixelEnd) {
        xPos = xPixelStart;
        ++yPos;
    }

    return spp;
}



//Sampler *CreateSBFSampler(const ParamSet &params,
//                       const Film *film, const Camera *camera) {
//    int is = params.FindOneInt("initsamples", 8);
//    float ns = params.FindOneFloat("adaptivesamples", 24.f);
//    int ms = params.FindOneInt("maxsamples", 1024);
//    int it = params.FindOneInt("adaptiveiteration", 1);
//    int xstart, xend, ystart, yend;
//    film->GetPixelExtent(&xstart, &xend, &ystart, &yend);
//    return new SBFSampler(xstart, xend, ystart, yend, ns,
//                          camera->shutterOpen, camera->shutterClose,
//                          is, ms, it);
//}


