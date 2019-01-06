
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

#ifndef PBRT_SBF_FILM_IMAGE_H
#define PBRT_SBF_FILM_IMAGE_H

// film/image.h*
#include "pbrt.h"
#include "film.h"
#include "sampler.h"
#include "filter.h"
#include "parallel.h"

#include "sbf.h"

class SBFImageFilm : public Film {
public:
    // SBFImageFilm Public Methods
    SBFImageFilm(int xres, int yres, Filter *filt, bool dp, SBF::FilterType type,
              const vector<float> &interParams, const vector<float> &finalParams,
              float sigmaN, float sigmaR, float sigmaD,
              float interMseSigmaS, float finalMseSigmaS);
    ~SBFImageFilm() {
        delete sbf;
    }
    void AddSample(float* sample);
    void Splat(const CameraSample &sample, const Spectrum &L);
    void GetSampleExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void GetPixelExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void WriteImage(float* img);
    void UpdateDisplay(int x0, int y0, int x1, int y1, float splatScale) {}
    void GetAdaptPixels(float avgSpp, vector<vector<int> > &pixOff, vector<vector<int> > &pixSmp);
private:
    // SBFImageFilm Private Data
    Filter *filter;
    float cropWindow[4];
    string filename;
    int xPixelStart, yPixelStart, xPixelCount, yPixelCount;
    
    SBF *sbf;
    bool dump;
};

SBFImageFilm *CreateSBFImageFilm(const ParamSet &params, Filter *filter);

#endif // PBRT_SBF_FILM_IMAGE_H
