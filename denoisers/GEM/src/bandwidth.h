
/*
 *  Copyright(c) 2011 Fabrice Rousselle.
 * 
 *  You can redistribute and/or modify this file under the terms of the GNU
 *  General Public License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 */

#ifndef BANDWIDTHSAMPLER_H
#define	BANDWIDTHSAMPLER_H


#include "sampler.h"
#include "stratified.h"
#include "random.h"
#include "smooth.h"
#include "montecarlo.h"


struct ScramblingInfo {
    ScramblingInfo() { _nGenerated = 0; }
    uint32_t _nGenerated;
    uint32_t _image[2];
    uint32_t _lens[2];
    uint32_t _time;
    vector<uint32_t> _oneD;
    vector<uint32_t> _twoD;
};


class BandwidthSampler : public Sampler {
public:
    // BandwidthSampler public methods
    BandwidthSampler(int xstart, int xend, int ystart, int yend, int spp,
        float sopen, float sclose, float threshold, int nIterations,
        const SmoothFilm *film);
    BandwidthSampler(const BandwidthSampler *parent, int xstart, int xend,
        int ystart, int yend);
    BandwidthSampler(const BandwidthSampler *parent, const PixelAreaVec &pixels,
        int taskNum, int nPixels);
    virtual ~BandwidthSampler();

    Sampler *GetSubSampler(int num, int count);
    int GetMoreSamples(float *sample, RNG& rng);

    int MaximumSampleCount() { return max(1, samplesPerPixel); }

    int RoundSize(int size) const { return size; }

    void SetAdaptiveMode() { _adaptive = true; }

    int PixelsToSample() { return int(_pixels.size()); }
    int PixelsToSampleTotal() { return _pixelsToSampleTotal; }

    int GetIterationCount() { return _nIterations; }

    void GetWorstPixels(int nPixels) {
        _film->GetWorstPixels(min(nPixels, _pixelsToSampleTotal), _pixels, samplesPerPixel);
        _pixelsToSampleTotal -= _pixels.size();
    }
    
private:
    bool _isMainSampler;
    Mutex **_pixelMutexes;

    // Film attributes
    int _xPixelCount, _yPixelCount;
    
    // BandwidthSampler private attributes
    int _nIterations;
    const SmoothFilm *_film;
    bool _adaptive;
    int _pixelsToSampleTotal;

    // Attributes for initialization phase
    Sampler *_samplerInit;
    int _xPos, _yPos;
    int _sppInit;
    int _xStartSub, _xEndSub, _yStartSub, _yEndSub;

    // Attributes for adaptive phase
    PixelAreaVec _pixels;

    // Per-pixel scrambling info
    ScramblingInfo *_scrambling;

    // Filters used to produce the various scales
    vector<const Kernel2D*> _filters;

    // BandwidthSampler private methods
    void initBase(const BandwidthSampler * parent);
    void initAdapt(const BandwidthSampler * parent, const PixelAreaVec &pixels,
        int taskNum, int nPixels);

    // The following LD functions were extracted from montecarlo.h and extended
    // to provide scrambling info on the fly, as well as the number of samples
    // already generated. This allows our sampler to query iteratively from the
    // low-discrepancy sequence.
    void LDShuffleScrambled1D(int nSamples, int nPixel, int nGenerated,
        float *samples, RNG &rng, const uint32_t scramble) {
        for (int i = 0; i < nSamples * nPixel; ++i)
            samples[i] = VanDerCorput(i+nGenerated, scramble);
        for (int i = 0; i < nPixel; ++i)
            Shuffle(samples + i * nSamples, nSamples, 1, rng);
        Shuffle(samples, nPixel, nSamples, rng);
    }
    void LDShuffleScrambled2D(int nSamples, int nPixel, int nGenerated,
        float *samples, RNG &rng, const uint32_t scramble[2]) {
        for (int i = 0; i < nSamples * nPixel; ++i)
            Sample02(i+nGenerated, scramble, &samples[2*i]);
        for (int i = 0; i < nPixel; ++i)
            Shuffle(samples + 2 * i * nSamples, nSamples, 2, rng);
        Shuffle(samples, nPixel, 2 * nSamples, rng);
    }

};

Sampler *CreateBandwidthSampler(const ParamSet &params, const Film *film,
    const Camera *camera);

#endif	/* BANDWIDTHSAMPLER_H */

