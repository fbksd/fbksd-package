/* 
 * File:   NlmeansKernel.h
 * Author: rousselle
 *
 * Created on March 19, 2012, 2:46 PM
 */

#ifndef NLMEANSKERNEL_H
#define	NLMEANSKERNEL_H


#include <vector>
using std::vector;

typedef vector<float> ImageBuffer;


#define LD_SAMPLING

enum NlmeansData {
    NLM_DATA_INTER,
    NLM_DATA_FINAL
};

class NlmeansKernel
{
public:
    NlmeansKernel() {};
    virtual ~NlmeansKernel() {};
    
    void Init(int wnd_rad, int ptc_rad, float k, int xPixelCount, int yPixelCount);
    
    // This function computes the 'out' image, by applying computing the
    // nlmeans filter weights at each pixel and applying them
    void Apply(NlmeansData dataType,
        const ImageBuffer &spp,
        const ImageBuffer &avg1, const ImageBuffer &var1, ImageBuffer &avgVar1,
        const ImageBuffer &avg2, const ImageBuffer &var2, ImageBuffer &avgVar2,
        ImageBuffer &avgOut1, ImageBuffer &sppOut1,
        ImageBuffer &avgOut2, ImageBuffer &sppOut2);
    
private:
    // Nlmeans parameters
    int _wnd_rad; // the filter window width
    int _ptc_rad; // the neighbourhood size
    float _k; // the damping factor: low values yield more conservative filter
    float _sigma; // the spatial sigma (set to very large for box filter)
    
    int _xPixelCount, _yPixelCount;
    
    template <typename T>
    T sqr(T v) const { return v*v; }
    
    float rgb2sum(const float *rgb) const { return rgb[0]+rgb[1]+rgb[2]; }
    float rgb2avg(const float *rgb) const { return rgb2sum(rgb)/3.f; }
    
    void ApplyIntVar(int nChannels, const float *img,
        const float *guide, const float *guideVar, int rad, float vScale);
    void ApplyInt(const float *inAvg, const float *inSpp,
        const float *guideAvg, const float *guideAvgVar,
        NlmeansData dataType, int wnd_rad, float gamma, float vScale);
};

#endif	/* NLMEANSKERNEL_H */

