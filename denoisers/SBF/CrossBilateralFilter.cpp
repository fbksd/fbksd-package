
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


#include "CrossBilateralFilter.h"

#include "fmath.hpp"
#include <omp.h>

/**
 *  As we do not optimize feature parameters, It might be a good idea to do some local 
 *  normalization when using the feature buffers, as in Random Parameter 
 *  Filtering[Sen and Darabi 2012]. E.g. for the filter window of each pixel 
 *  we gather all the features in the window and calculate z-scores of them.
 *  Anyway, fixed feature parameters with Mahalanobis-like distance seems to 
 *  work well with our scenes, so we leave the normalization as a future work.
 */

const float c_VarMax = 1e-2f;

CrossBilateralFilter::CrossBilateralFilter(
            float sigmaS,
            float sigmaC,
            const Feature &sigmaF,
            int w, int h
            ) {
    radius = Round2Int(sigmaS*2.f);
    scaleS = sigmaS <= 0.f ? 
        0.f : -0.5f/(sigmaS*sigmaS);
    scaleC = sigmaC <= 0.f ? 
        0.f : -0.5f/(sigmaC*sigmaC);
    for(int i = 0; i < sigmaF.Size(); i++) {
        scaleF[i] = sigmaF[i] <= 0.f ?
            0.f : -0.5f/(sigmaF[i]*sigmaF[i]);
    }
    width = w; height = h;
    int nPixels = width*height;
    nTasks = max(32 * omp_get_num_procs(), nPixels / (16*16));
    nTasks = RoundUpPow2(nTasks);
}

void CrossBilateralFilter::Apply(
                const vector<TwoDArray<float> > &mseArray,
                const vector<TwoDArray<float> > &priArray,
                const TwoDArray<Feature> &featureImg,
                const TwoDArray<Feature> &featureVarImg,
                vector<TwoDArray<float> > &outMSE,
                vector<TwoDArray<float> > &outPri) const {
#pragma omp parallel for num_threads(omp_get_num_procs()) schedule(static)
    for(int taskId = 0; taskId < nTasks; taskId++) {
        int txs, txe, tys, tye;
        ComputeSubWindow(taskId, nTasks, width, height,
                         &txs, &txe, &tys, &tye);
        for(int y = tys; y < tye; y++) {
            for(int x = txs; x < txe; x++) {
                int ys = std::max(y-radius, 0);
                int ye = std::min(y+radius, featureImg.GetRowNum()-1);
                int xs = std::max(x-radius, 0);
                int xe = std::min(x+radius, featureImg.GetColNum()-1);
                Feature feature = featureImg(x, y);
                Feature featureVar = featureVarImg(x, y);            
                vector<float> mseSum(mseArray.size(), 0.f);
                vector<float> priSum(priArray.size(), 0.f);
                vector<float> wSum(mseArray.size(), 0.f);
                for(int yy = ys; yy <= ye; yy++) { 
                    int yDist = (yy-y)*(yy-y);
                    for(int xx = xs; xx <= xe; xx++) {
                        Feature fDiff = feature - featureImg(xx, yy);                    
                        Feature fVarSum = featureVar + featureVarImg(xx, yy);
                        Feature fDist = (fDiff*fDiff)/fVarSum.Max(c_VarMax);
                        float sDist = (float)(yDist + (xx - x)*(xx - x));
                        float w = fmath::exp(sDist*scaleS +
                                Sum(fDist*scaleF));

                        for(size_t i = 0; i < mseSum.size(); i++) {
                            mseSum[i] += w*mseArray[i](xx, yy);
                            priSum[i] += w*priArray[i](xx, yy);
                            wSum[i] += w;
                        }
                    }
                }

                for(size_t i = 0; i < mseSum.size(); i++) {
                    outMSE[i](x, y) = mseSum[i]/wSum[i];
                    outPri[i](x, y) = priSum[i]/wSum[i];
                }
            }
        }

    }
}

void CrossBilateralFilter::Apply(const TwoDArray<Color> &img,
                                 const TwoDArray<Feature> &featureImg,
                                 const TwoDArray<Feature> &featureVarImg,
                                 const TwoDArray<Color> &rImg,
                                 const TwoDArray<Color> &varImg,
                                 const TwoDArray<Color> &rVarImg,
                                 TwoDArray<Color> &outImg,                                  
                                 TwoDArray<float> &outMSE,
                                 TwoDArray<float> &outPri) const {
#pragma omp parallel for num_threads(omp_get_num_procs()) schedule(static)
    for(int taskId = 0; taskId < nTasks; taskId++) {
        int txs, txe, tys, tye;
        ComputeSubWindow(taskId, nTasks, width, height, 
                         &txs, &txe, &tys, &tye);
        for(int y = tys; y < tye; y++) {
            for(int x = txs; x < txe; x++) {
                int dxs = max(x-radius, 0);
                int dxe = min(x+radius, width-1);
                int dys = max(y-radius, 0);
                int dye = min(y+radius, height-1);
                Color rColor = rImg(x, y);
                Feature feature = featureImg(x, y);
                Feature featureVar = featureVarImg(x, y);            
                Color sum = 0.f, sqSum = 0.f, rSum = 0.f, rSqSum = 0.f;
                float wSum = 0.f;
                for(int dy = dys; dy <= dye; dy++) { 
                    int yDist = (dy-y)*(dy-y);
                    for(int dx = dxs; dx <= dxe; dx++) {
                        Color cDiff = rColor - rImg(dx, dy);
                        Color cDist = cDiff*cDiff;
                        Feature fDiff = feature - featureImg(dx, dy);                    
                        Feature fVarSum = featureVar + featureVarImg(dx, dy);
                        Feature fDist = (fDiff*fDiff)/fVarSum.Max(c_VarMax);
                        float sDist = (float)(yDist + (dx - x)*(dx - x));
                        float w = fmath::exp(sDist*scaleS +
                                Sum(cDist)*scaleC +
                                Sum(fDist*scaleF));

                        Color r = rImg(dx, dy);
                        sum += w*img(dx, dy);
                        sqSum += w*img(dx, dy)*img(dx, dy);
                        rSum += w*r;                        
                        rSqSum += w*r*r;
                        wSum += w;
                    }
                }

                float invWSum = 1.f/wSum;
                Color fY = sum*invWSum;
                Color rY = rColor;
                Color rfY = rSum*invWSum; 
                Color rdFdY = invWSum - scaleC*(rSqSum*invWSum-rfY*rfY);
                Color rError = (rfY-rY)*(rfY-rY) + 2.f*rVarImg(x, y)*rdFdY - rVarImg(x, y);
                Color pri = rError + rVarImg(x, y);

                outImg(x, y) = fY;
                outMSE(x, y) = Avg(rError);
                outPri(x, y) = Avg(pri) / (fY.Y()*fY.Y() + 1e-2f);
            }
        }

    }
}
