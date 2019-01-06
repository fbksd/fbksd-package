/* 
 * File:   dualfilm.h
 * Author: rousselle
 *
 * Created on March 13, 2012, 11:35 AM
 */

#ifndef DUALFILM_H
#define	DUALFILM_H

// film/image.h*
#include "pbrt.h"
#include "film.h"
#include "sampler.h"
#include "filter.h"
#include "nlmdenoiser.h"

enum TargetBuffer {
    BUFFER_A,
    BUFFER_B
};

// DualFilm Declarations
class DualFilm : public Film {
public:
    // ImageFilm Public Methods
    DualFilm(int xres, int yres, Filter *filt, int wnd_rad, float k, int ptc_rad);
    ~DualFilm() {
        delete[] filterTable;
        delete _denoiser;
    }
    
    // Only support the AddSample with explicit target assignment
    void Splat(const CameraSample &sample, const Spectrum &L) {
        Error("DualFilm::Splat: Not supported!");
    }
    void AddSample(const CameraSample &sample, const Spectrum &L) {
        Error("DualFilm::AddSample: No target buffer specified!");
    }
    void AddSample(float* sample, TargetBuffer target);
    
    void GetSamplingMaps(int spp, int nSamples, ImageBuffer &samplingMapA,
        ImageBuffer &samplingMapB) const {
        _denoiser->UpdatePixelData(pixelsA, pixelsB, subPixelsA, subPixelsB, NLM_DATA_INTER);
        _denoiser->GetSamplingMaps(spp, nSamples, samplingMapA, samplingMapB);
    }
    
    void GetSampleExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void GetPixelExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void WriteImage(float* img);
    void UpdateDisplay(int x0, int y0, int x1, int y1, float splatScale);

    int GetXPixelCount() const { return xPixelCount; }
    int GetYPixelCount() const { return yPixelCount; }
    
    void Finalize() const {
        _denoiser->UpdatePixelData(pixelsA, pixelsB, subPixelsA, subPixelsB, NLM_DATA_FINAL);
    }

private:
    // Output filename
//    string filename;
    // The pixel reconstruction filter
    Filter *filter;
    // The crop window
    float cropWindow[4];
    // Output pixel coordinates. Usually PixelStart values are set to 0 and the
    // PixelCount values are set the image dimensions. For tile-based rendering
    // the PixelStart and PixelCount values correspond to the tile being
    // rendered.
    int xPixelStart, yPixelStart, xPixelCount, yPixelCount;
    // Actual film buffer. This DualFilm class holds two buffers. Samples are
    // stored in either buffer according to the AddSample 'target' value.
    vector<NlmeansPixel> pixelsA,   pixelsB;
    float *filterTable;
    // The denoiser used to filter out the noise from the rendering.
    NlmeansDenoiser *_denoiser;
    // Subpixel buffer. Since our sampling is potentially non-uniform, we store
    // samples on a sub-pixel grid using a box filter, and then apply the pixel
    // filter on the sub-pixel grid instead of the actual samples.
    int subPixelRes;
    vector<NlmeansSubPixel> subPixelsA,   subPixelsB;

    inline
    float sqr(float v) { return v*v; }
};


DualFilm *CreateDualFilm(const ParamSet &params, Filter *filter);

#endif	/* DUALFILM_H */

