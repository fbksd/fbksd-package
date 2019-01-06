
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


#include "CrossNLMFilter.h"

#include "fmath.hpp"
#include <omp.h>

const float c_VarMax = 1e-2f;

/**
 *  We implemented a very brute force version of non-local means filter,
 *  where the computational complexity is O(NS^2P^2), there are many 
 *  acceleration algorithms out there, but most of them are approximation
 *  and have very different filter derivatives, therefore we will have to
 *  derive the derivatives if we want to use them.
 *
 *  There exists some exact non-local means filter acceleration algorithms
 *  (such as "An improved Non-Local Denoising Algorithm" by Goossens et al or
 *  "A Simple Trick to Speed Up and Improve the Non-Local Means" by Condat)
 *  that reduce the complexity to O(NS^2P) or even O(NS^2).
 *  However, as we have to access to both filtered pixel value and the weight 
 *  between pixels in SURE calculation, it is difficult to incorporate them.
 *  One way is to store the weights for each pixel and each filter parameter, 
 *  but it will introduce siginificant memory cost, for example, for a 1024x768
 *  image with 5x5 patch and 8 parameters, we will need about 600MB to store
 *  the weights, we estimate about 3-4 speedup factor in this case.
 */

CrossNLMFilter::CrossNLMFilter(
            int _searchRadius,
            int _patchRadius,            
            vector<float> &sigmaR,
            const Feature &sigmaF, 
            int w, int h
            ) {  
    searchRadius = _searchRadius;
    searchWidth = 2*searchRadius+1;
    patchRadius = _patchRadius;
    patchWidth = 2*patchRadius+1;
    invPatchWidth = 1.f/patchWidth;
    invPatchSize = 1.f/(float)(patchWidth*patchWidth);
    for(size_t i = 0; i < sigmaR.size(); i++) {
        scaleR.push_back(sigmaR[i] <= 0.f ? 
                0.f : -0.5f/(sigmaR[i]*sigmaR[i]));
    }
    for(int i = 0; i < sigmaF.Size(); i++) {
        scaleF[i] = sigmaF[i] <= 0.f ?
            0.f : -0.5f/(sigmaF[i]*sigmaF[i]);
    }
    width = w; height = h;
    int nPixels = width*height;
    nTasks = max(32 * omp_get_num_procs(), nPixels / (16*16));
    nTasks = RoundUpPow2(nTasks);
//    nTasks = 1;
}

void CrossNLMFilter::Apply(
                  float sigmaR,
                  const vector<TwoDArray<float> > &mseArray,
                  const vector<TwoDArray<float> > &priArray,
                  const TwoDArray<Color> &rImg,
                  const TwoDArray<Feature> &featureImg,
                  const TwoDArray<Feature> &featureVarImg,
                  vector<TwoDArray<float> > &outMSE,
                  vector<TwoDArray<float> > &outPri) const {
    float mseScaleR = -0.5f/(sigmaR*sigmaR);
#pragma omp parallel for num_threads(omp_get_num_procs()) schedule(static)
    for(int taskId = 0; taskId < nTasks; taskId++) {
        int txs, txe, tys, tye;
        ComputeSubWindow(taskId, nTasks, width, height,
                         &txs, &txe, &tys, &tye); 
        for(int y = tys; y < tye; y++) 
            for(int x = txs; x < txe; x++) { 
                vector<float> mseSum(mseArray.size(), 0.f);
                vector<float> priSum(priArray.size(), 0.f);
                vector<float> wSum(mseArray.size(), 0.f);
                Feature feature = featureImg(x, y);
                Feature featureVar = featureVarImg(x, y);            
 
                // Filter using pixels within search range
                for(int dy = -searchRadius; dy <= searchRadius; dy++)
                    for(int dx = -searchRadius; dx <= searchRadius; dx++) {
                        int xx = x + dx;
                        int yy = y + dy;
                        if(xx < 0 || yy < 0 || xx >= width || yy >= height) 
                            continue;
                        Color diffSqSum(0.f, 0.f, 0.f);
                        // Calculate block distance
                        for(int by = -patchRadius; by <= patchRadius; by++)
                            for(int bx = -patchRadius; bx <= patchRadius; bx++) {
                                int xbx = x + bx;
                                int yby = y + by;
                                int xxbx = xx + bx;
                                int yyby = yy + by;
                                if( xbx < 0 || xbx >= width ||
                                    yby < 0 || yby >= height ||
                                    xxbx < 0 || xxbx >= width ||
                                    yyby < 0 || yyby >= height)
                                    continue;

                                Color diff = rImg(xbx, yby) - rImg(xxbx, yyby);
                                diffSqSum += (diff*diff);
                            }
                        diffSqSum *= invPatchSize;
                        float dist = Avg(diffSqSum);
                        Feature fDiff = feature - featureImg(xx, yy);                    
                        Feature fVarSum = featureVar + featureVarImg(xx, yy);
                        Feature fDist = (fDiff*fDiff)/fVarSum.Max(c_VarMax);
                        float weight = fmath::exp(dist*mseScaleR +
                                                  Sum(fDist*scaleF)); 
                        for(size_t i = 0; i < mseArray.size(); i++) {
                            mseSum[i] += weight * mseArray[i](xx, yy);
                            priSum[i] += weight * priArray[i](xx, yy);
                            wSum[i] += weight;
                        }
                    }                

                for(size_t i = 0; i < mseArray.size(); i++) {
                    outMSE[i](x, y) = mseSum[i] / wSum[i];
                    outPri[i](x, y) = priSum[i] / wSum[i];
                }
            }            
    }   

}

void CrossNLMFilter::Apply(const TwoDArray<Color> &img,
               const TwoDArray<Feature> &featureImg,
               const TwoDArray<Feature> &featureVarImg,
               const TwoDArray<Color> &rImg,
               const TwoDArray<Color> &varImg,
               vector<TwoDArray<Color> > &fltArray,
               vector<TwoDArray<float> > &mseArray,
               vector<TwoDArray<float> > &priArray) const {    
#pragma omp parallel for num_threads(omp_get_num_procs()) schedule(static)
    for(int taskId = 0; taskId < nTasks; taskId++) {
        int txs, txe, tys, tye;
        ComputeSubWindow(taskId, nTasks, width, height,
                         &txs, &txe, &tys, &tye); 
        for(int y = tys; y < tye; y++) 
            for(int x = txs; x < txe; x++) { 
//                std::cout << "CrossNLMFilter::Apply: " << y << " of " << tye << std::endl;

                vector<Color> sum(scaleR.size(), Color(0.f));
                vector<Color> rSum(scaleR.size(), Color(0.f));
                vector<Color> rSumSq(scaleR.size(), Color(0.f));
                vector<float> wSum(scaleR.size(), 0.f);
                vector<vector<float> > wArray(scaleR.size());
                for(size_t p = 0; p < wArray.size(); p++) {
                    wArray[p].resize(patchWidth*patchWidth);
                }
                Feature feature = featureImg(x, y);
                Feature featureVar = featureVarImg(x, y);            
 
                // Filter using pixels within search range
                for(int dy = -searchRadius; dy <= searchRadius; dy++)
                    for(int dx = -searchRadius; dx <= searchRadius; dx++) {
                        int xx = x + dx;
                        int yy = y + dy;
                        if(xx < 0 || yy < 0 || xx >= width || yy >= height) 
                            continue;
                        Color diffSqSum(0.f, 0.f, 0.f);
                        // Calculate block distance
                        for(int by = -patchRadius; by <= patchRadius; by++)
                            for(int bx = -patchRadius; bx <= patchRadius; bx++) {
                                int xbx = x + bx;
                                int yby = y + by;
                                int xxbx = xx + bx;
                                int yyby = yy + by;
                                if( xbx < 0 || xbx >= width ||
                                    yby < 0 || yby >= height ||
                                    xxbx < 0 || xxbx >= width ||
                                    yyby < 0 || yyby >= height)
                                    continue;

                                Color diff = rImg(xbx, yby) - rImg(xxbx, yyby);
                                diffSqSum += (diff*diff);
                            }
                        diffSqSum *= invPatchSize;
                        float dist = Avg(diffSqSum);
                        Feature fDiff = feature - featureImg(xx, yy);                    
                        Feature fVarSum = featureVar + featureVarImg(xx, yy);
                        Feature fDist = (fDiff*fDiff)/fVarSum.Max(c_VarMax);
                        // For each parameter, calculate information necessary for 
                        // filtering and SURE estimation
                        for(size_t p = 0; p < scaleR.size(); p++) {
                            if(scaleR[p] == 0.f) {
                                continue;
                            }
                            float weight = fmath::exp(dist*scaleR[p] +
                                                      Sum(fDist*scaleF));
                            sum[p] += weight * img(xx, yy);
                            rSum[p] += weight * rImg(xx, yy);
                            rSumSq[p] += weight * rImg(xx, yy) * rImg(xx, yy);
                            wSum[p] += weight;
                            if(dy >= -patchRadius && dy <= patchRadius &&
                               dx >= -patchRadius && dx <= patchRadius) {
                                wArray[p][(dy+patchRadius)*patchWidth+(dx+patchRadius)] =
                                    weight;
                            }
                        }
                    }                

                for(size_t p = 0; p < scaleR.size(); p++) {
                    if(scaleR[p] == 0.f) {
                        fltArray[p](x, y) = img(x, y);
                        mseArray[p](x, y) = Avg(2.f*varImg(x, y));
                        continue;
                    }
                    float invWSum = 1.f/wSum[p];
                    Color xl    = sum[p] * invWSum;
                    Color rxl   = rSum[p] * invWSum;
                    Color rxlSq = rSumSq[p] * invWSum;
                    Color ryl   = rImg(x, y);
                    Color dxdy  = (-2.f*scaleR[p])*(rxlSq - rxl*rxl)*invPatchSize + Color(invWSum);

                    Color tmp;
                    for(int by = -patchRadius; by <= patchRadius; by++)
                        for(int bx = -patchRadius; bx <= patchRadius; bx++) {
                            int xbpx = x+bx;
                            int ybpy = y+by;
                            int xbmx = x-bx;
                            int ybmy = y-by;                        
                            if( xbpx < 0 || xbpx >= width  ||
                                ybpy < 0 || ybpy >= height ||
                                xbmx < 0 || xbmx >= width  ||
                                ybmy < 0 || ybmy >= height)
                                continue;
                            Color rylpb = rImg(xbpx, ybpy);
                            Color rylmb = rImg(xbmx, ybmy);
                            float w = wArray[p][(-by+patchRadius)*patchWidth+(-bx+patchRadius)];
                            tmp += w*(ryl - rylpb)*(rxl - rylmb);
                        }
                    tmp *= (-2.f*scaleR[p])*invPatchSize*invWSum;
                    dxdy += tmp;
                    Color mse = (rxl-ryl)*(rxl-ryl) + 2.f*varImg(x, y)*dxdy - varImg(x, y);
                    Color pri = (mse + varImg(x, y));
                    fltArray[p](x, y) = xl;
                    mseArray[p](x, y) = Avg(mse);
                    priArray[p](x, y) = Avg(pri) / (xl.Y()*xl.Y() + 1e-2f);
                } 
            }            
    }
}
