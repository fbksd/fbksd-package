
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


#if defined(_MSC_VER)
#pragma once
#endif
#ifndef PBRT_SAMPLERS_SBF_SAMPLER_H
#define PBRT_SAMPLERS_SBF_SAMPLER_H

// samplers/sbfsampler.h*
#include "sampler.h"
#include "film.h"
#include "pbrt.h"
#include "parallel.h"

class SBFSampler : public Sampler {
public:
    SBFSampler(int xstart, int xend, int ystart,
        int yend, float as, float sopen, float sclose,
        int is, int ms, int it, 
        vector<vector<int> > *pixoff=NULL,
        vector<vector<int> > *pixsmp=NULL,
        int bxs=-1, int bys=-1);
    virtual ~SBFSampler() {
        if(sampleBuf) delete[] sampleBuf;
    }
    int MaximumSampleCount() { return maxSamples; }
    int GetMoreSamples(float* sample, RNG &rng);
    int RoundSize(int sz) const { 
        return sz; 
    }
    Sampler *GetSubSampler(int num, int count);
    float GetAdaptiveSPP() const {
        return adaptiveSamples;
    }
    int GetIteration() const {
        return iteration;
    }
    void SetPixelOffset(vector<vector<int> > *po) {
        pixelOffset = po;
    }
    void SetPixelSampleCount(vector<vector<int> > *ps) {
        pixelSampleCount = ps;
    }
private:

    // SBFSampler Private Data              
    vector<vector<int> > *pixelOffset;
    vector<vector<int> > *pixelSampleCount;     
    float *sampleBuf;
    int initSamples;
    float adaptiveSamples;
    int maxSamples;
    int iteration;
    int xPos, yPos;
    int baseXStart, baseYStart;
};


Sampler *CreateSBFSampler(const ParamSet &params, const Film *film,
    const Camera *camera);

#endif // PBRT_SAMPLERS_SBF_SAMPLER_H
