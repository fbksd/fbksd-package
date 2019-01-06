
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

#ifndef SBF_CROSS_NLM_FILTER_H__
#define SBF_CROSS_NLM_FILTER_H__

#include "SBFCommon.h"
#include "TwoDArray.h"
#include "VectorNf.h"


class CrossNLMFilter {
public:
    CrossNLMFilter(
            int searchRadius,
            int patchRadius,            
            vector<float> &sigmaR,
            const Feature &sigmaF, 
            int width,
            int height
            );

    // Filter SURE images
    void Apply(float sigmaR,
               const vector<TwoDArray<float> > &mseArray,
               const vector<TwoDArray<float> > &priArray,
               const TwoDArray<Color> &rImg,
               const TwoDArray<Feature> &featureImg,
               const TwoDArray<Feature> &featureVarImg,
               vector<TwoDArray<float> > &outMSE,
               vector<TwoDArray<float> > &outPri) const;

    // Filter MC reconstructed image
    void Apply(const TwoDArray<Color> &img,
               const TwoDArray<Feature> &featureImg,
               const TwoDArray<Feature> &featureVarImg,
               const TwoDArray<Color> &rImg,
               const TwoDArray<Color> &VarImg,
               vector<TwoDArray<Color> > &fltArray,
               vector<TwoDArray<float> > &mseArray,
               vector<TwoDArray<float> > &priArray) const;

private:
    int searchRadius, patchRadius;
    int searchWidth, patchWidth;
    float invPatchWidth;
    float invPatchSize;
    vector<float> scaleR;
    Feature scaleF;
    int width, height;
    int nTasks;
};

#endif //#ifndef SBF_CROSS_NLM_FILTER_H__
