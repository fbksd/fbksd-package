/* 
 * File:   multifilm.h
 * Author: Fabrice Rousselle
 *
 * Created on 31. mars 2013, 16:12
 */

#ifndef MULTIFILM_H
#define	MULTIFILM_H

#include "film.h"
#include "featurefilter.h"

struct StatsInfo {
    void Init(int npixels) {
        nsamples.resize(npixels);
        pixel_mean.resize(2*npixels);
        rgb_mean.resize(3*npixels);
        normal_mean.resize(3*npixels);
        albedo_mean.resize(3*npixels);
        visibility_mean.resize(npixels);
        depth_mean.resize(npixels);

        pixel_m2.resize(2*npixels);
        rgb_m2.resize(3*npixels);
        normal_m2.resize(3*npixels);
        albedo_m2.resize(3*npixels);
        visibility_m2.resize(npixels);
        depth_m2.resize(npixels);

        pixel_mean_var.resize(2*npixels);
        rgb_mean_var.resize(3*npixels);
        rgb_mean_var_flt.resize(3*npixels);
        normal_mean_var.resize(3*npixels);
        albedo_mean_var.resize(3*npixels);
        visibility_mean_var.resize(npixels);
        depth_mean_var.resize(npixels);
    }
    vector<float> nsamples;
    // mean values
    vector<float> pixel_mean;
    vector<float> rgb_mean;
    vector<float> normal_mean;
    vector<float> albedo_mean;
    vector<float> visibility_mean;
    vector<float> depth_mean;
    // second moment
    vector<float> pixel_m2;
    vector<float> rgb_m2;
    vector<float> normal_m2;
    vector<float> albedo_m2;
    vector<float> visibility_m2;
    vector<float> depth_m2;
    // variance
    vector<float> pixel_mean_var;
    vector<float> rgb_mean_var;
    vector<float> rgb_mean_var_flt;
    vector<float> normal_mean_var;
    vector<float> albedo_mean_var;
    vector<float> visibility_mean_var;
    vector<float> depth_mean_var;
};

// DualFilm Declarations
class MultiFilm : public Film {
public:
    // ImageFilm Public Methods
    MultiFilm(int xres, int yres, int wnd_rad);
    ~MultiFilm() {
//        delete filter;
    }
    
    void AddSample(const float*){}
    void AddSample(const float* sample, int target);
    
    void GetSamplingMap(int spp, int nsamples, Buffer &map);
    
    void GetSampleExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void GetPixelExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void WriteImage(float* img);
    void UpdateDisplay(int x0, int y0, int x1, int y1, float splatScale);

    int GetXPixelCount() const { return xPixelCount; }
    int GetYPixelCount() const { return yPixelCount; }
    
    void Finalize() const {
//        _denoiser->UpdatePixelData(pixelsA, pixelsB, subPixelsA, subPixelsB, NLM_DATA_FINAL);
//        denoiser->UpdatePixelData(pixelsA, pixelsB, subPixelsA, subPixelsB,
//            featuresA, featuresB, subFeaturesA, subFeaturesB,
//            NLM_DATA_FINAL);
    }
    
    int GetBufferCount() const {
        return int(buffers_rgb.size());
    }
    
    void SetFlagLD(bool use_ld_samples) {
        this->use_ld_samples = use_ld_samples;
    }

    int numbuffers;
//private:
    // Output filename
    string filename;
    // The crop window
    float cropWindow[4];
    // Output pixel coordinates. Usually PixelStart values are set to 0 and the
    // PixelCount values are set the image dimensions. For tile-based rendering
    // the PixelStart and PixelCount values correspond to the tile being
    // rendered.
    int xPixelStart, yPixelStart, xPixelCount, yPixelCount;
    // The film buffers. We can have 1 to n buffers, stored in a vector.
    int npixels;
    Buffer reference, sel_map;
    BufferSet filtered_scales;
    BufferSet buffers_pixel;
    BufferSet buffers_rgb;
    BufferSet buffers_normal;
    BufferSet buffers_albedo;
    BufferSet buffers_position;
    BufferSet buffers_visibility;
    BufferSet buffers_depth;
    BufferSet buffers_nsamples;
    StatsInfo statistics;
    // User parameters
    int wnd_rad; // filter window radius
    int ptc_rad; // filter patch radius
    bool use_ld_samples;
    
    // The feature-based non-local means filter
    FeatureFilter *filter;
    
    void UpdateVariances();
    void GetBufferMeanVariance(Buffer &variance, const BufferSet &buffers, int nchannels);
    void GetFilteredDataFast(Buffer &filtered_data, Buffer &derivative, Buffer &filtered_data_spp);
    void GetFilteredData(Buffer &filtered_data1, Buffer &filtered_data2);
    
    void ComputeSURE(Buffer &mse, const Buffer &img, const Buffer &ref,
        const Buffer &ref_var, const Buffer &derivative, float var_scale = 1.f);
    
    int itr_count;
    enum DumpType {
        DUMP_FINAL,
        DUMP_ITERATION
    };
    void DumpMap(Buffer &data, const string &tag, DumpType dumpType, int buf = -1, int width = -1, int height = -1);
    void LoadImageRGB(Buffer &data, const string &tag, DumpType dumpType, int buf = -1);
    void DumpImageRGB(Buffer &data, const string &tag, DumpType dumpType, int buf = -1, int width = -1, int height = -1);
    
    void ComputeFilteredScales(BufferSet &scales_mse, BufferSet &weights,
        vector<BufferSet> &filtered_scales_buffers, BufferSet &confidence,
        float maxPos, BufferSet &derivatives);
    void ComputeSelectionMap(BufferSet &sel_weights,
        const BufferSet &scales_mse, float maxPos, const BufferSet &derivatives);
    void ApplySelectionMap(Buffer &filtered_data,
        BufferSet &filtered_buffers, const BufferSet &filtered_scales,
        const vector<BufferSet> &filtered_scales_buffers,
        const BufferSet &sel_weights);
    
    void PushAllFeatures(float *maxPos);
    
    inline float sqr(float v) const { return v*v; }
    inline float rgb2avg(const float *rgb) const { return (rgb[0]+rgb[1]+rgb[2])/3.f; }
    
    void ReadImageExr(Buffer &img, const char *name);
    void ReadFeatureRef(BufferSet &buffers, Buffer &var, const char *feat);
    void NormalizeFeature(BufferSet &buffers, Buffer &var);
};


#endif	/* MULTIFILM_H */

