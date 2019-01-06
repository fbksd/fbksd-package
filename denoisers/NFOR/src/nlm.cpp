#include "nlm.h"
#include "featurefilter.h"

#include <cstring>
#include <cmath>


#define DIST_EPSILON 1e-10f


void nlm(float* input,
         int width,
         int height,
         int inc,
         int wRad,
         int pRad,
         float k,
         float* guide,
         float* guideVar,
         int gnc,
         float* output)
{
    float k2 = k*k;

    #pragma omp parallel for
    for(int y = 0; y < height; ++y)
    {
        int ymin = std::max(0, y - wRad);
        int ymax = std::min(height-1, y + wRad);
        int patch1Ymin = std::max(ymin, y - pRad);
        int patch1Ymax = std::min(ymax, y + pRad);

        for(int x = 0; x < width; ++x)
        {
            float wSum = 0.f;
            int xmin = std::max(0, x - wRad);
            int xmax = std::min(width-1, x + wRad);
            int patch1Xmin = std::max(xmin, x - pRad);
            int patch1Xmax = std::min(xmax, x + pRad);

            for(int j = ymin; j <= ymax; ++j)
            {
                int patch2Ymin = std::max(ymin, j - pRad);
                int patch2Ymax = std::min(ymax, j + pRad);

                for(int i = xmin; i <= xmax; ++i)
                {
                    //patch around (i,j)
                    int patch2Xmin = std::max(xmin, i - pRad);
                    int patch2Xmax = std::min(xmax, i + pRad);

                    // patch distance
                    float d2Sum = 0.f;
                    int n = 0;
                    for(int qj = patch2Ymin; qj <= patch2Ymax; ++qj)
                    {
                        int dy = qj - j;
                        int pj = y + dy;
                        if(pj < patch1Ymin || pj > patch1Ymax) continue;

                        for(int qi = patch2Xmin; qi <= patch2Xmax; ++qi)
                        {
                            int dx = qi - i;
                            int pi = x + dx;
                            if(pi < patch1Xmin || pi > patch1Xmax) continue;

                            for(int c = 0; c < gnc; ++c)
                            {
                                float varP = guideVar[pj*width*gnc + pi*gnc + c];
                                float varQ = guideVar[qj*width*gnc + qi*gnc + c];
                                float varPQ = std::min(varP, varQ);
                                float d2 = guide[pj*width*gnc + pi*gnc + c] - guide[qj*width*gnc + qi*gnc + c];
                                d2 *= d2;
                                d2 = (d2 - (varP + varPQ)) / (k2*(varP + varQ) + DIST_EPSILON);
                                d2Sum += d2;
                                ++n;
                            }
                        }
                    }
                    float d2 = d2Sum / (float)n;
                    float wr = std::exp(-std::max(0.f, d2));
                    //float ws = std::exp(-(i - x)*(i - x)*0.5f)*std::exp(-(j - y)*(j - y)*0.5f);
                    float ws = 1.0f;
                    float w = wr * ws;

                    wSum += w;
                    for(int c = 0; c < inc; ++c)
                        output[y*width*inc + x*inc + c] += w * input[j*width*inc + i*inc + c];
                }
            }

            for(int c = 0; c < inc; ++c)
                output[y*width*inc + x*inc + c] /= wSum;
        }
    }
}


VectorXf computeNlmWeights(float* img, float* imgVar, int width, int height, const MatrixBlock& block, int x, int y, int F, float k)
{
    float k2 = k*k;
    int pRad = F;
    int gnc = 3;

    int xmin = block.xBegin;
    int xmax = block.getXEnd();
    int ymin = block.yBegin;
    int ymax = block.getYEnd();

    int patch1Xmin = std::max(xmin, x - pRad);
    int patch1Xmax = std::min(xmax, x + pRad);
    int patch1Ymin = std::max(ymin, y - pRad);
    int patch1Ymax = std::min(ymax, y + pRad);

    VectorXf weights(block.getSize());
    int index = 0;

    for(int j = ymin; j <= ymax; ++j)
    {
        int patch2Ymin = std::max(ymin, j - pRad);
        int patch2Ymax = std::min(ymax, j + pRad);

        for(int i = xmin; i <= xmax; ++i)
        {
            //patch around (i,j)
            int patch2Xmin = std::max(xmin, i - pRad);
            int patch2Xmax = std::min(xmax, i + pRad);

            // patch distance
            float d2Sum = 0.f;
            int n = 0;
            for(int qj = patch2Ymin; qj <= patch2Ymax; ++qj)
            {
                int dy = qj - j;
                int pj = y + dy;
                if(pj < patch1Ymin || pj > patch1Ymax) continue;

                for(int qi = patch2Xmin; qi <= patch2Xmax; ++qi)
                {
                    int dx = qi - i;
                    int pi = x + dx;
                    if(pi < patch1Xmin || pi > patch1Xmax) continue;

                    for(int c = 0; c < gnc; ++c)
                    {
                        float varP = imgVar[pj*width*gnc + pi*gnc + c];
                        float varQ = imgVar[qj*width*gnc + qi*gnc + c];
                        float varPQ = std::min(varP, varQ);
                        float d2 = img[pj*width*gnc + pi*gnc + c] - img[qj*width*gnc + qi*gnc + c];
                        d2 *= d2;
                        d2 = (d2 - (varP + varPQ)) / (k2*(varP + varQ) + DIST_EPSILON);
                        d2Sum += d2;
                        ++n;
                    }
                }
            }
            float d2 = d2Sum / (float)n;
            float wr = std::exp(-std::max(0.f, d2));
//            float ws = std::exp(-(i - x)*(i - x)*0.5f)*std::exp(-(j - y)*(j - y)*0.5f);
            float ws = 1.0f;
            float w = wr * ws;

            weights(index) = w;
            ++index;
        }
    }

    return weights;
}


void gpuNlm(float* input,
         int width,
         int height,
         int inc,
         int wRad,
         int pRad,
         float k,
         float* guide,
         float* guideVar,
         int gnc,
         float* output)
{
    size_t numPixels = width * height;
    FeatureFilter* filter = new FeatureFilter(width, height);

    // set filter parameters
    filter->SetFilterMode(FILTER_WITHOUT_FEATURES);
    filter->SetWndRad(wRad);
    filter->SetPtcRad(pRad);
    filter->SetVarNumScale(1.f);
    filter->SetVarDenScale(k*k);

    filter->PushGuide(guide, guideVar, gnc);
    filter->PushData(input, inc);

    Buffer buffer(numPixels * inc, 0);
    filter->GetFilteredData(buffer);
    delete filter;
    memcpy(output, buffer.data(), buffer.size() * sizeof(float));
}

VectorXf gpuComputeNlmWeights(float* img, float* imgVar, int width, int height, int x, int y, int wRad, int pRad, float k)
{
    FeatureFilter* filter = new FeatureFilter(width, height);

    // set filter parameters
    filter->SetFilterMode(FILTER_WITHOUT_FEATURES);
    filter->SetWndRad(wRad);
    filter->SetPtcRad(pRad);
    filter->SetVarNumScale(1.f);
    filter->SetVarDenScale(k*k);

    filter->PushGuide(img, imgVar, 3);
    filter->PushData(img, 3);

    Buffer buffer;
    filter->GetPixelWeights(x, y, buffer);
    VectorXf weights(buffer.size());
    for(size_t i = 0; i < buffer.size(); ++i)
        weights(i) = buffer[i];

    delete filter;
    return weights;
}
