#ifndef NLM_H
#define NLM_H

#include <vector>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
using namespace Eigen;


struct MatrixBlock
{
public:
    MatrixBlock() :
        xBegin(0), yBegin(0), xSize(0), ySize(0), cIndex(0)
    {}

    MatrixBlock(long x, long y, long w, long h, int rad)
    {
        xBegin = std::max(x - rad, 0L);
        yBegin = std::max(y - rad, 0L);
        long xEnd = std::min(x + rad, w - 1);
        long yEnd = std::min(y + rad, h - 1);
        xSize = xEnd - xBegin + 1;
        ySize = yEnd - yBegin + 1;
        cIndex = (x - xBegin)*(y - yBegin);
    }

    long getXEnd() const
    { return xSize + xBegin - 1; }

    long getYEnd() const
    { return ySize + yBegin - 1; }

    long getSize() const
    { return xSize*ySize; }

    long xBegin;
    long yBegin;
    long xSize;
    long ySize;
    long cIndex;
};


void nlm(float* input,
         int width,
         int height,
         int inputNumChannels,
         int wRad,
         int pRad,
         float k,
         float* guide,
         float* guideVar,
         int guideNumChannels,
         float* output);


VectorXf computeNlmWeights(float* img,
                           float* imgVar,
                           int width,
                           int height,
                           const MatrixBlock& block,
                           int x,
                           int y,
                           int F=3,
                           float k=0.5f);

void gpuNlm(float* input,
             int width,
             int height,
             int inputNumChannels,
             int wRad,
             int pRad,
             float k,
             float* guide,
             float* guideVar,
             int guideNumChannels,
             float* output);

VectorXf gpuComputeNlmWeights(float* img,
                              float* imgVar,
                              int width,
                              int height,
                              int x,
                              int y,
                              int wRad,
                              int pRad,
                              float k);

#endif // NLM_H
