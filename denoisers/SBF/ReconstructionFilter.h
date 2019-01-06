
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

#ifndef SBF_RECONSTRUCTION_FILTER_H__
#define SBF_RECONSTRUCTION_FILTER_H__

#include "SBFCommon.h"
#include "TwoDArray.h"
#include "filter.h"
#include <omp.h>

enum FilterType {
    FILTER_MEAN,
    FILTER_VAR
};

class ReconstructionFilter {
public:
    inline ReconstructionFilter(const Filter *filter);
    template<typename T>
    void Apply(TwoDArray<T> &image) const;

private:
    inline void BuildKernel(vector<float> &kernel);

    vector<float> xKernel;
    vector<float> yKernel;
    int xWidth, yWidth;
};

ReconstructionFilter::ReconstructionFilter(const Filter *filter) {
    xWidth = filter->xWidth;
    yWidth = filter->yWidth;
    xKernel.resize(2*xWidth+1);
    yKernel.resize(2*yWidth+1);

    // Assume the filter is separable for efficiency 
    // (true for all default filters in pbrt)    
    for(size_t i = 0; i < xKernel.size(); i++) {
        float dist = (float)i - (float)filter->xWidth;
        float w = filter->Evaluate(dist, 0.f);
        xKernel[i] = w;        
    }

    for(size_t i = 0; i < yKernel.size(); i++) {
        float dist = (float)i - (float)filter->yWidth;
        float w = filter->Evaluate(0.f, dist);
        yKernel[i] = w;        
    }
}

template<typename T>
void ReconstructionFilter::Apply(TwoDArray<T> &image) const {
    TwoDArray<T> tmpBuf(image.GetColNum(), image.GetRowNum());

    // X direction filter    
#pragma omp parallel for num_threads(omp_get_num_procs())
    for(int y = 0; y < image.GetRowNum(); y++)
        for(int x = 0; x < image.GetColNum(); x++) {
            int minX = max(x - xWidth, 0);
            int maxX = min(x + xWidth, image.GetColNum()-1);
            T sum = 0.f;
            float wSum = 0.f;
            int kPos = minX - x + xWidth;            
            for(int xx = minX; xx <= maxX; xx++, kPos++) {
                float w = xKernel[kPos];
                sum += w*image(xx, y);
                wSum += w;
            }            
            tmpBuf(x, y) = sum/wSum;
        }

    // Y direction filter
#pragma omp parallel for num_threads(omp_get_num_procs())
    for(int y = 0; y < image.GetRowNum(); y++)
        for(int x = 0; x < image.GetColNum(); x++) {
            int minY = max(y - yWidth, 0);
            int maxY = min(y + yWidth, image.GetRowNum()-1);
            T sum = 0.f;
            float wSum = 0.f;
            int kPos = minY - y + yWidth;            
            for(int yy = minY; yy <= maxY; yy++, kPos++) {
                float w = yKernel[kPos];
                sum += w*tmpBuf(x, yy);
                wSum += w;
            } 
            image(x, y) = sum/wSum;
        }
}


#endif //#ifndef RECONSTRUCTION_FILTER_H__
