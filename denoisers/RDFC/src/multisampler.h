/* 
 * File:   multisampler.h
 * Author: Fabrice Rousselle
 *
 * Created on 31. mars 2013, 18:48
 */

#ifndef MULTISAMPLER_H
#define	MULTISAMPLER_H

#include "sampler.h"
#include "random.h"
#include "multifilm.h"
#include "montecarlo.h"

#include "featurefilter.h"

//#define LD_SAMPLING_MULTI


struct MultiScramblingInfo {
    MultiScramblingInfo() { _nGenerated = 0; }
    uint32_t _nGenerated;
    vector<uint32_t> _seeds;
    uint32_t *_image;
    uint32_t *_lens;
    uint32_t *_time;
    uint32_t *_oneD;
    uint32_t *_twoD;
};


class MultiSampler : public Sampler {
public:
    // MultiSampler public methods
    MultiSampler(int xstart, int xend, int ystart, int yend, int spp,
        float sopen, float sclose, float threshold, int nIterations,
        int sppInit, const MultiFilm *film, bool finalize, bool use_ld_samples);
    // Constructor for sub-sampler during init phase
    MultiSampler(const MultiSampler *parent, int xstart, int xend,
        int ystart, int yend, int pass);
    // Constructor for sub-sampler during adaptive phase, uses sampling map
    MultiSampler(const MultiSampler *parent, int xstart, int xend,
        int ystart, int yend, Buffer &samplingMap,
        MultiScramblingInfo *scrambling);
    virtual ~MultiSampler();

    Sampler *GetSubSampler(int num, int count);
    int GetMoreSamples(float *sample, RNG &rng) {
        return (use_ld_samples) ? GetMoreSamplesMapLD(sample, rng) : GetMoreSamplesMap(sample, rng);
    }

    int MaximumSampleCount() {
        return max(1, samplesPerPixel);
    }

    int RoundSize(int size) const { return size; }

    void SetAdaptiveMode() { adaptive = true; }

    int PixelsToSampleTotal() { return pixelsToSampleTotal; }
    
    int GetIterationCount() { return _nIterations; }

    void GetSamplingMaps(int nPixels) {
        nPixels = min(nPixels, pixelsToSampleTotal);
        film->GetSamplingMap(samplesPerPixel, nPixels*samplesPerPixel, *samplingmap);
        pixelsToSampleTotal -= nPixels;
    }
    
    void Finalize() const {
        if (finalize) film->Finalize();
    }
    
    bool IsSingleBuffered() const {
        return !finalize;
    }
    
    // This method ensures that the task granularity is reasonable, and that the
    // number of tasks is a multiple of the number of buffers
    int GetNumberOfTasks(int npixels) const {
        int ntasks = max(32 * NumSystemCores(), npixels / (16*16));
        return Ceil2Int(float(ntasks) / scrambling.size()) * scrambling.size();
    }
    
    int GetTarget(int n, int N) const {
        N = N / scrambling.size();
        return n / N;
    }
    
private:
    bool _isMainSampler;
    bool use_ld_samples;

    // Film attributes
    int xPixelCount, yPixelCount;
    
    // MultiSampler private attributes
    int _nIterations;
    MultiFilm *film;
    bool adaptive;
    int pixelsToSampleTotal;

    // Attributes for initialization phase
    Sampler *samplerInit;
    int _xPos, _yPos;
    int sppInit, sppInitReq;
    int _xStartSub, _xEndSub, _yStartSub, _yEndSub;

    // Attributes for adaptive phase
    Buffer *samplingmap;
    float _sppErr;

    // Per-pixel scrambling info
    vector<MultiScramblingInfo *> scrambling;

    // MultiSampler private methods
    void initBase(const MultiSampler * parent, int pass = 0);

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
    
    float *samplesBuf;
    void MyLDPixelSample(int xPos, int yPos, float shutterOpen,
        float shutterClose, int nPixelSamples, float *samples, RNG &rng,
        MultiScramblingInfo *scramblingArray);
    
    bool finalize;
};

Sampler *CreateMultiSampler(const ParamSet &params, const Film *film,
    const Camera *camera);


#endif	/* MULTISAMPLER_H */

