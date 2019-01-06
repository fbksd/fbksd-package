
/*
    Copyright(c) 2012-2013 Tzu-Mao Li
    All rights reserved.

    The code is based on PBRT: http://www.pbrt.org

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

#ifndef PBRT_SBF_H
#define PBRT_SBF_H

#include "memory.h"
#include "rng.h"

#include "ReconstructionFilter.h"
#include "TwoDArray.h"
#include "VectorNf.h"
#include "SBFCommon.h"

class SBF {
public:       
    enum FilterType {
        CROSS_BILATERAL_FILTER,
        CROSS_NLM_FILTER
    };

    SBF(int xs, int ys, int w, int h, 
        const Filter *filt, FilterType type,
        const vector<float> &interParams,
        const vector<float> &finalParams,
        float sigmaN, float sigmaR, float sigmaD,
        float interMseSigma, float finalMseSigma);
    ~SBF() {
        delete pixelInfos;
    }
    void AddSample(float *sample);
    void GetAdaptPixels(float avgSpp, vector<vector<int> > &pixOff, vector<vector<int> > &pixSmp);
    void GetAdaptPDF(float* pdf);
    void WriteImage(const string &filename, int xres, int yres, bool dump, float* img);
    void GetResult(float* result);

    void Update(bool final);
private:
    void WriteImage(const string &filename, const TwoDArray<Color> &image, int xres, int yres) const;
    TwoDArray<Color> FloatImageToColor(const TwoDArray<float> &image) const;
    float CalculateAvgSpp() const;

    struct PixelInfo {
        PixelInfo() {
            for(int i = 0; i < 3; i++) {
                Lrgb[i] = sqLrgb[i] =
                    normal[i] = sqNormal[i] = 
                    rho[i] = sqRho[i] =  
                    depth = sqDepth = 0.f;
            }
            sampleCount = 0;
        }
        float Lrgb[3];
        float sqLrgb[3];
        float normal[3];
        float sqNormal[3];
        float rho[3];
        float sqRho[3];
        float depth;
        float sqDepth;
        float weightSum;
        int sampleCount;
    };

    FilterType fType;
    ReconstructionFilter rFilter;
    vector<float> interParams, finalParams;
    float sigmaN, sigmaR, sigmaD;
    float interMseSigma, finalMseSigma;

    BlockedArray<PixelInfo> *pixelInfos;    
    int xPixelStart, yPixelStart;
    int xPixelCount, yPixelCount;

    // Storing the image, features and their variance 
    // reconstructed by default filter
    TwoDArray<Color> colImg;
    TwoDArray<Color> varImg;
    TwoDArray<Feature> featureImg;
    TwoDArray<Feature> featureVarImg;
    // The adaptive sampling metric
    TwoDArray<float> adaptImg; 
    // Filtered image
    TwoDArray<Color> fltImg;

    // These images are stored for debug and visualization
    TwoDArray<Color> rhoImg;
    TwoDArray<Color> rhoVarImg;
    TwoDArray<Color> norImg;
    TwoDArray<Color> norVarImg;
    TwoDArray<float> depthImg;
    TwoDArray<float> depthVarImg;
    TwoDArray<float> minMseImg;
    TwoDArray<Color> sigmaImg;

    RNG rng;
};

#endif //PBRT_SBF_H

