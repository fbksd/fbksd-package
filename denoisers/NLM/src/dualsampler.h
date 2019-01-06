/* 
 * File:   DualSampler.h
 * Author: rousselle
 *
 * Created on March 16, 2012, 3:42 PM
 */

#ifndef DUALSAMPLER_H
#define	DUALSAMPLER_H

#include "sampler.h"
#include "stratified.h"
#include "random.h"
#include "dualfilm.h"
#include "montecarlo.h"
#include "nlmkernel.h"

struct NlmeansScramblingInfo {
    NlmeansScramblingInfo() { _nGenerated = 0; }
    uint32_t _nGenerated;
    vector<uint32_t> _seeds;
    uint32_t *_image;
    uint32_t *_lens;
    uint32_t *_time;
    uint32_t *_oneD;
    uint32_t *_twoD;
};


class DualSampler : public Sampler {
public:
    // DualSampler public methods
    DualSampler(int xstart, int xend, int ystart, int yend, int spp,
        float sopen, float sclose, float threshold, int nIterations,
        int sppInit, const DualFilm *film);
    // Constructor for sub-sampler during init phase
    DualSampler(const DualSampler *parent, int xstart, int xend,
        int ystart, int yend, bool firstPass);
    // Constructor for sub-sampler during adaptive phase, uses sampling map
    DualSampler(const DualSampler *parent, int xstart, int xend,
        int ystart, int yend, ImageBuffer &samplingMap,
        NlmeansScramblingInfo *scrambling);
    virtual ~DualSampler();

    Sampler *GetSubSampler(int num, int count);
    int GetMoreSamples(float *sample, RNG &rng) {
#ifdef LD_SAMPLING
        return GetMoreSamplesMapLD(sample, rng);
#else
        return GetMoreSamplesMap(sample, rng);
#endif
    }

    int MaximumSampleCount() {
        return max(1, samplesPerPixel);
    }

    int RoundSize(int size) const { return size; }

    void SetAdaptiveMode() { _adaptive = true; }

    int PixelsToSampleTotal() { return _pixelsToSampleTotal; }

    int GetIterationCount() { return _nIterations; }

    void GetSamplingMaps(int nPixels) {
        nPixels = min(nPixels, _pixelsToSampleTotal);
        _film->GetSamplingMaps(samplesPerPixel, nPixels*samplesPerPixel, *_samplingMapA, *_samplingMapB);
        _pixelsToSampleTotal -= nPixels;
    }
    
    void Finalize() const {
        _film->Finalize();
    }
    
private:
    bool _isMainSampler;

    // Film attributes
    int _xPixelCount, _yPixelCount;
    
    // DualSampler private attributes
    int _nIterations;
    const DualFilm *_film;
    bool _adaptive;
    int _pixelsToSampleTotal;

    // Attributes for initialization phase
    Sampler *_samplerInit;
    int _xPos, _yPos;
    int _sppInit, _sppInitReq;
    int _xStartSub, _xEndSub, _yStartSub, _yEndSub;

    // Attributes for adaptive phase
    ImageBuffer *_samplingMapA, *_samplingMapB;
    float _sppErr;

    // Per-pixel scrambling info
    NlmeansScramblingInfo *_scramblingA, *_scramblingB;

    // Filters used to produce the various scales
    vector<const Kernel2D*> _filters;

    // DualSampler private methods
    void initBase(const DualSampler * parent, bool firstPass = true);

    // The following LD functions were extracted from montecarlo.h and extended
    // to provide scrambling info on the fly, as well as the number of samples
    // already generated. This allows our sampler to query iteratively from the
    // low-discrepancy sequence.
    void MyLDShuffleScrambled1D(int nSamples, int nPixel, int nGenerated,
        float *samples, RNG &rng, const uint32_t scramble) {
        for (int i = 0; i < nSamples * nPixel; ++i)
            samples[i] = VanDerCorput(i+nGenerated, scramble);
        for (int i = 0; i < nPixel; ++i)
            Shuffle(samples + i * nSamples, nSamples, 1, rng);
        Shuffle(samples, nPixel, nSamples, rng);
    }
    void MyLDShuffleScrambled2D(int nSamples, int nPixel, int nGenerated,
        float *samples, RNG &rng, const uint32_t scramble[2]) {
        for (int i = 0; i < nSamples * nPixel; ++i)
            Sample02(i+nGenerated, scramble, &samples[2*i]);
        for (int i = 0; i < nPixel; ++i)
            Shuffle(samples + 2 * i * nSamples, nSamples, 2, rng);
        Shuffle(samples, nPixel, 2 * nSamples, rng);
    }

    int GetMoreSamplesMap(float *sample, RNG &rng);
    int GetMoreSamplesMapLD(float *sample, RNG &rng);
    
    float *_samplesBuf;
    void MyLDPixelSample(int xPos, int yPos, float shutterOpen,
        float shutterClose, int nPixelSamples, float *samples, RNG &rng,
        NlmeansScramblingInfo *scramblingArray);
    void MyLDPixelSampleInterleaved(int xPos, int yPos, float shutterOpen,
        float shutterClose, int nPixelSamples, float *samples, RNG &rng);
};

Sampler *CreateDualSampler(const ParamSet &params, const Film *film,
    const Camera *camera);

#endif	/* DUALSAMPLER_H */

