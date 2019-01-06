
/*
 *  Copyright(c) 2011 Fabrice Rousselle.
 * 
 *  You can redistribute and/or modify this file under the terms of the GNU
 *  General Public License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 */

#if defined(_MSC_VER)
#pragma once
#endif

#ifndef PBRT_FILM_SMOOTH_H
#define PBRT_FILM_SMOOTH_H

// film/image.h*
#include "pbrt.h"
#include "film.h"
#include "sampler.h"
#include "filter.h"
#include "denoiser.h"

// SmoothFilm Declarations
class SmoothFilm : public Film {
public:
    // ImageFilm Public Methods
    SmoothFilm(int xres, int yres, Filter *filt, float gamma);
    ~SmoothFilm() {
        delete pixels;
        delete variance;
        delete[] filterTable;
        delete _denoiser;
    }
    void AddSample(float* sample);
    void GetSampleExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void GetPixelExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void WriteImage(float* img);
    void UpdateDisplay(int x0, int y0, int x1, int y1, float splatScale);
    void GetWorstPixels(int nPixels, PixelAreaVec &pixelAreas, int spp) const {
        _denoiser->SetPixelMeanAndVariance(subPixels, variance);
        _denoiser->GetWorstPixels(nPixels, pixelAreas, spp);
    }
    int GetXPixelCount() const { return xPixelCount; }
    int GetYPixelCount() const { return yPixelCount; }
    void GetFilterbank(vector<const Kernel2D*> &filters) const {
        _denoiser->GetFilterbank(filters);
    }

private:
    bool _splatting;
    // ImageFilm Private Data
    Filter *filter;
    float cropWindow[4];
    string filename;
    int xPixelStart, yPixelStart, xPixelCount, yPixelCount;
    BlockedArray<Pixel> *pixels;
    BlockedArray<BoxVariance> *variance;
    float *filterTable;

    Denoiser *_denoiser;
    int subPixelRes;
    vector<SubPixel> subPixels;
};


SmoothFilm *CreateSmoothFilm(const ParamSet &params, Filter *filter);

#endif // PBRT_FILM_SMOOTH_H

