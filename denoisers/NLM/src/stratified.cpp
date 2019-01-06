
/*
    pbrt source code Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

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


// samplers/stratified.cpp*
#include "stdafx.h"
#include "stratified.h"
#include "montecarlo.h"

// StratifiedSampler Method Definitions
StratifiedSampler::StratifiedSampler(int xstart, int xend,
        int ystart, int yend, int xs, int ys, bool jitter,
        float sopen, float sclose)
    : Sampler(xstart, xend, ystart, yend, xs * ys, sopen, sclose) {
    jitterSamples = jitter;
    xPos = xPixelStart;
    yPos = yPixelStart;
    xPixelSamples = xs;
    yPixelSamples = ys;
    sampleBuf = new float[5 * xPixelSamples * yPixelSamples];
}


StratifiedSampler::~StratifiedSampler() {
    delete[] sampleBuf;
}


Sampler *StratifiedSampler::GetSubSampler(int num, int count) {
    int x0, x1, y0, y1;
    ComputeSubWindow(num, count, &x0, &x1, &y0, &y1);
    if (x0 == x1 || y0 == y1) return NULL;
    return new StratifiedSampler(x0, x1, y0, y1, xPixelSamples,
        yPixelSamples, jitterSamples, shutterOpen, shutterClose);
}


int StratifiedSampler::GetMoreSamples(float *samples, RNG &rng) {
    if (yPos == yPixelEnd) return 0;
    int nSamples = xPixelSamples * yPixelSamples;
    // Generate stratified camera samples for _(xPos, yPos)_

    // Generate initial stratified samples into _sampleBuf_ memory
    float *bufp = sampleBuf;
    float *imageSamples = bufp; bufp += 2 * nSamples;
    float *lensSamples = bufp;  bufp += 2 * nSamples;
    float *timeSamples = bufp;
    StratifiedSample2D(imageSamples, xPixelSamples, yPixelSamples, rng,
                       jitterSamples);
    StratifiedSample2D(lensSamples, xPixelSamples, yPixelSamples, rng,
                       jitterSamples);
    StratifiedSample1D(timeSamples, xPixelSamples * yPixelSamples, rng,
                       jitterSamples);

    // Shift stratified image samples to pixel coordinates
    for (int o = 0; o < 2 * xPixelSamples * yPixelSamples; o += 2) {
        imageSamples[o]   += xPos;
        imageSamples[o+1] += yPos;
    }

    // Decorrelate sample dimensions
    Shuffle(lensSamples, xPixelSamples*yPixelSamples, 2, rng);
    Shuffle(timeSamples, xPixelSamples*yPixelSamples, 1, rng);

    // Initialize stratified _samples_ with sample values
    for (int i = 0; i < nSamples; ++i) {
        samples[i*5] = imageSamples[2*i];
        samples[i*5 + 1] = imageSamples[2*i+1];
//        samples[i].imageX = imageSamples[2*i];
//        samples[i].imageY = imageSamples[2*i+1];
//        samples[i].lensU = lensSamples[2*i];
//        samples[i].lensV = lensSamples[2*i+1];
//        samples[i].time = Lerp(timeSamples[i], shutterOpen, shutterClose);
//        // Generate stratified samples for integrators
//        for (uint32_t j = 0; j < samples[i].n1D.size(); ++j)
//            LatinHypercube(samples[i].oneD[j], samples[i].n1D[j], 1, rng);
//        for (uint32_t j = 0; j < samples[i].n2D.size(); ++j)
//            LatinHypercube(samples[i].twoD[j], samples[i].n2D[j], 2, rng);
    }

    // Advance to next pixel for stratified sampling
    if (++xPos == xPixelEnd) {
        xPos = xPixelStart;
        ++yPos;
    }
    return nSamples;
}


//StratifiedSampler *CreateStratifiedSampler(const ParamSet &params, const Film *film,
//         const Camera *camera) {
//    bool jitter = params.FindOneBool("jitter", true);
//    // Initialize common sampler parameters
//    int xstart, xend, ystart, yend;
//    film->GetSampleExtent(&xstart, &xend, &ystart, &yend);
//    int xsamp = params.FindOneInt("xsamples", 2);
//    int ysamp = params.FindOneInt("ysamples", 2);
//    if (PbrtOptions.quickRender) xsamp = ysamp = 1;
//    return new StratifiedSampler(xstart, xend, ystart, yend, xsamp, ysamp,
//        jitter, camera->shutterOpen, camera->shutterClose);
//}


