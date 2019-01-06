/* 
 * File:   featurefilter.h
 * Author: Fabrice Rousselle
 *
 * Created on 31. mars 2013, 18:37
 */

#ifndef FEATUREFILTER_H
#define	FEATUREFILTER_H

#include <vector>
using std::vector;
#include <string>
using std::string;
#include <limits>
using std::numeric_limits;

#include <stdlib.h>

typedef vector<float> Buffer;
typedef vector<Buffer> BufferSet;

enum FeatureFilterMode {
    FILTER_WITH_FEATURES,
    FILTER_WITHOUT_FEATURES,
    FILTER_WITHOUT_COLOR
};

typedef vector<float> ImageBuffer;

class FeatureFilter {
public:
    // ImageFilm Public Methods
    FeatureFilter(int width, int height);
    virtual ~FeatureFilter();
    
    void Reset();
    
    void PushFeature(const BufferSet &buffers, int nchannels, float threshold, bool filter = true);
    
    void PushGuide(float* mean, float* mean_var, int nchannels);
    void PushGuide(const Buffer &mean, const Buffer &mean_var, int nchannels);
    void PushGuide(const BufferSet &buffers, int nchannels);
    void PushGuide(const BufferSet &buffers, const Buffer &var, int nchannels);
    void PushGuideNlm(const BufferSet &buffers, const Buffer &var, int nchannels);
    
    void PushData(float* buffer, int nchannels);
    void PushData(const Buffer &buffer, int nchannels);
    void PushData(const BufferSet &buffers, int nchannels);
    void PushDataDelta(const BufferSet &buffers, int nchannels);
    void PushDataDelta(const BufferSet &buffers, const Buffer &spp, int nchannels);
    void PushDataDelta(const Buffer &buffer, const Buffer &spp, int nchannels);
    void PushDataMeanVariance(const Buffer &variance, int nchannels);
    
    // retrieve data
    void GetFilteredData(Buffer &output);
    void GetFilteredDataBuffers(BufferSet &output);
    void GetFilteredDataAndDerivative(Buffer &output, Buffer &deriv);
    void GetFilteredDataAndDerivative(Buffer &output, Buffer &deriv, Buffer &output_spp);
    void GetDataMeanVar(Buffer &mean_var);
    void GetGuideVariance(Buffer &var);
    
    // set parameters
    void SetUseDiffVar(bool flag) { use_diff_var = flag; }
    void SetWndRad(int rad) { wnd_rad = rad; }
    void SetPtcRad(int rad) { ptc_rad = rad; }
    void SetVarNumScale(float scale) { var_num_scale = scale; }
    void SetVarDenScale(float scale) { var_den_scale = scale; }
    void SetVarNumScaleFeat(float scale) { var_num_scale_feat = scale; }
    void SetVarDenScaleFeat(float scale) { var_den_scale_feat = scale; }
    void SetFilterMode(FeatureFilterMode mode) { this->mode = mode; }
    void SetFeatureThreshold(size_t feat_idx, float threshold, bool use_grad = true, float scale = 1.f);
    
    // debugging tools
    void GetPixelWeights(int x, int y, Buffer &weights);
    
private:
    // Buffer dimensions
    int width;
    int height;
    int nbuffers;
    int nbuffers_alloc;
    int nchannels_data;
    int nchannels_guide;
    size_t size_img_bytes_1c;
    size_t size_img_bytes_3c;
    // parameters
    bool use_diff_var;
    int wnd_rad;
    int ptc_rad;
    float var_num_scale;
    float var_den_scale;
    float var_num_scale_feat;
    float var_den_scale_feat;
    FeatureFilterMode mode;
    bool feat_only;
    // Data
    float *data;
    float *data_mean;
    float *data_mean_var_num;
    float *data_mean_var_den;
    float *data_filtered;
    float *data_filtered_mean;
    float *data_s_mean;
    float *data_s_mean_filtered;
    float *spp_filtered;
    // Single-channel features
    size_t         feat_idx;
    vector<float*> features;
    vector<float*> features_var_num;
    vector<float*> features_var_den;
    vector<int>    features_nchannels;
    //
    float kdata;
    float kfeat;
    // temporary work buffers
    float *tmp, *tmp2, *tmp3, *tmp4;
    float *tmp_n1, *tmp_n2, *tmp_ns;
    float *d2_n1, *d2_n1_s, *d2_n1_b;
    float *d2_n2, *d2_n2_s, *d2_n2_b;
    float *d2_ns, *d2_ns_s, *d2_ns_b;
    float *wgt_n1, *wgt_n1_tmp, *wgt_n1_s, *wgt_n1_tmp_s;
    float *wgt_n2, *wgt_n2_tmp, *wgt_n2_s, *wgt_n2_tmp_s;
    float *wgt_ns, *wgt_ns_tmp, *wgt_ns_s, *wgt_ns_tmp_s;
    float *wgt_sum, *wgt_sum_s;
    float *wgt_max, *wgt_max_s;
    float *ones, *conf;
    Buffer img;
    
    void MallocData(size_t nbuffers);
    void WeightsData(float *wgt_buf, int x, int y, int wnd_rad, int ptc_rad,
        int ptc_rad_wgt, float var_num_scale, float var_den_scale,
        int nchannels_data, int nchannels_guide, FeatureFilterMode mode,
        float var_num_scale_feat, float var_den_scale_feat);
    void FilterData(int wnd_rad, int ptc_rad, int ptc_rad_wgt,
        float var_num_scale, float var_den_scale, int nchannels_data,
        int nchannels_guide, FeatureFilterMode mode, float var_num_scale_feat,
        float var_den_scale_feat, bool use_diff_var);
    void FilterDataDelta(int wnd_rad, int ptc_rad, float var_num_scale,
        float var_den_scale, int nchannels_data, int nchannels_guide, 
        FeatureFilterMode mode, float var_num_scale_feat,
        float var_den_scale_feat);
    void FilterDataDeltaSpp(int wnd_rad, int ptc_rad, float var_num_scale,
        float var_den_scale, int nchannels_data, int nchannels_guide, 
        FeatureFilterMode mode, float var_num_scale_feat,
        float var_den_scale_feat);
    
    void GetDataMean(float *dst, const float *src);
    void GetScaledSampleVariance(float *dst, const float *src, float *var_smp, int nchannels);
    void GetDataMeanVariance(float *dst, const float *src, float k = 0.1353f);
    void UpdateWeights(const float *data_mean, const float *data_mean_var_num,
        const float *data_mean_var_den, float data_var_num_scale,
        float data_var_den_scale, int ptc_rad, int ptc_rad_wgt, int dx, int dy,
        int nchannels, bool use_diff_var);
    void UpdateWeightsFeat(
        const float *data_mean,
        const float *data_mean_var_num,
        const float *data_mean_var_den,
        float var_num_scale,
        float var_den_scale,
        int ptc_rad,                // patch radius for distance filtering
        int ptc_rad_wgt,            // patch radius for weight filtering
        int dx, int dy,             // offset to first neighbor
        int nchannels,
        bool use_diff_var);
    void UpdateWeightsDelta(const float *data_mean, const float *data_mean_var_num,
        const float *data_mean_var_den, float data_var_num_scale,
        float data_var_den_scale, int ptc_rad, int dx, int dy, int nchannels);
    
    void DumpMap(Buffer &data, const string &tag, int buf = -1);
    void DumpImageRGB(Buffer &data, const string &tag, int buf = -1);
};

#endif	/* FEATUREFILTER_H */

