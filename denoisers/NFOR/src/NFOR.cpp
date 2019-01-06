#include "NFOR.h"
#include "nlm.h"
#include "imageio.h"

#include <iostream>
#include <thread>

using MatrixNf = Matrix<float, Dynamic, Dynamic, RowMajor>;
using VectorNf = Matrix<float, 1, Dynamic, RowMajor>;

//#define PRINT_BUFFERS

//#define USE_FEATURE_PIXEL
#define USE_FEATURE_NORMALS
#define USE_FEATURE_ALBEDO
#define USE_FEATURE_DEPTH
#define USE_FEATURE_VISIBILITY

enum EFeature
{
#ifdef USE_FEATURE_PIXEL
    IMAGE_X,
    IMAGE_Y,
#endif
#ifdef USE_FEATURE_NORMALS
    NORMAL_X,
    NORMAL_Y,
    NORMAL_Z,
#endif
#ifdef USE_FEATURE_ALBEDO
    ALBEDO_R,
    ALBEDO_G,
    ALBEDO_B,
#endif
#ifdef USE_FEATURE_DEPTH
    DEPTH,
#endif
#ifdef USE_FEATURE_VISIBILITY
    VISIBILITY,
#endif

    FEATURE_SIZE
};


static void copyFeatures(
        #ifdef USE_FEATURE_PIXEL
                  const BufferSet& buffers_pixel,
        #endif
        #ifdef USE_FEATURE_NORMALS
                  const BufferSet& buffers_normal,
        #endif
        #ifdef USE_FEATURE_ALBEDO
                  const BufferSet& buffers_albedo,
        #endif
        #ifdef USE_FEATURE_DEPTH
                  const BufferSet& buffers_depth,
        #endif
        #ifdef USE_FEATURE_VISIBILITY
                  const BufferSet& buffers_visibility,
        #endif
                  size_t numPixels,
                  BufferSet& features)
{
    features.resize(2);
    features[0].resize(numPixels * FEATURE_SIZE);
    features[1].resize(numPixels * FEATURE_SIZE);

    for(size_t j = 0; j < numPixels; ++j)
    {
#ifdef USE_FEATURE_PIXEL
        features[0][j*FEATURE_SIZE + IMAGE_X] = buffers_pixel[0][j*2];
        features[0][j*FEATURE_SIZE + IMAGE_Y] = buffers_pixel[0][j*2 + 1];
        features[1][j*FEATURE_SIZE + IMAGE_X] = buffers_pixel[1][j*2];
        features[1][j*FEATURE_SIZE + IMAGE_Y] = buffers_pixel[1][j*2 + 1];
#endif
#ifdef USE_FEATURE_NORMALS
        features[0][j*FEATURE_SIZE + NORMAL_X] = buffers_normal[0][j*3];
        features[0][j*FEATURE_SIZE + NORMAL_Y] = buffers_normal[0][j*3 + 1];
        features[0][j*FEATURE_SIZE + NORMAL_Z] = buffers_normal[0][j*3 + 2];
        features[1][j*FEATURE_SIZE + NORMAL_X] = buffers_normal[1][j*3];
        features[1][j*FEATURE_SIZE + NORMAL_Y] = buffers_normal[1][j*3 + 1];
        features[1][j*FEATURE_SIZE + NORMAL_Z] = buffers_normal[1][j*3 + 2];
#endif
#ifdef USE_FEATURE_ALBEDO
        features[0][j*FEATURE_SIZE + ALBEDO_R] = buffers_albedo[0][j*3];
        features[0][j*FEATURE_SIZE + ALBEDO_G] = buffers_albedo[0][j*3 + 1];
        features[0][j*FEATURE_SIZE + ALBEDO_B] = buffers_albedo[0][j*3 + 2];
        features[1][j*FEATURE_SIZE + ALBEDO_R] = buffers_albedo[1][j*3];
        features[1][j*FEATURE_SIZE + ALBEDO_G] = buffers_albedo[1][j*3 + 1];
        features[1][j*FEATURE_SIZE + ALBEDO_B] = buffers_albedo[1][j*3 + 2];
#endif
#ifdef USE_FEATURE_DEPTH
        features[0][j*FEATURE_SIZE + DEPTH] = buffers_depth[0][j];
        features[1][j*FEATURE_SIZE + DEPTH] = buffers_depth[1][j];
#endif
#ifdef USE_FEATURE_VISIBILITY
        features[0][j*FEATURE_SIZE + VISIBILITY] = buffers_visibility[0][j];
        features[1][j*FEATURE_SIZE + VISIBILITY] = buffers_visibility[1][j];
#endif
    }
}


static void copyVariances(
        #ifdef USE_FEATURE_PIXEL
                  const Buffer& pixel_mean_var,
        #endif
        #ifdef USE_FEATURE_NORMALS
                  const Buffer& normal_mean_var,
        #endif
        #ifdef USE_FEATURE_ALBEDO
                  const Buffer& albedo_mean_var,
        #endif
        #ifdef USE_FEATURE_DEPTH
                  const Buffer& depth_mean_var,
        #endif
        #ifdef USE_FEATURE_VISIBILITY
                  const Buffer& visibility_mean_var,
        #endif
                  size_t numPixels,
                  Buffer& featuresVar)
{
    featuresVar.resize(numPixels * FEATURE_SIZE);

    for(size_t j = 0; j < numPixels; ++j)
    {
#ifdef USE_FEATURE_PIXEL
        featuresVar[j*FEATURE_SIZE + IMAGE_X] = pixel_mean_var[j*2];
        featuresVar[j*FEATURE_SIZE + IMAGE_Y] = pixel_mean_var[j*2 + 1];
#endif
#ifdef USE_FEATURE_NORMALS
        featuresVar[j*FEATURE_SIZE + NORMAL_X] = normal_mean_var[j*3];
        featuresVar[j*FEATURE_SIZE + NORMAL_Y] = normal_mean_var[j*3 + 1];
        featuresVar[j*FEATURE_SIZE + NORMAL_Z] = normal_mean_var[j*3 + 2];
#endif
#ifdef USE_FEATURE_ALBEDO
        featuresVar[j*FEATURE_SIZE + ALBEDO_R] = albedo_mean_var[j*3];
        featuresVar[j*FEATURE_SIZE + ALBEDO_G] = albedo_mean_var[j*3 + 1];
        featuresVar[j*FEATURE_SIZE + ALBEDO_B] = albedo_mean_var[j*3 + 2];
#endif
#ifdef USE_FEATURE_DEPTH
        featuresVar[j*FEATURE_SIZE + DEPTH] = depth_mean_var[j];
#endif
#ifdef USE_FEATURE_VISIBILITY
        featuresVar[j*FEATURE_SIZE + VISIBILITY] = visibility_mean_var[j];
#endif
    }
}


class PatchUtil
{
public:
    PatchUtil(Buffer& img, size_t w, size_t h, int nChannels):
        img(img), w(w), h(h), nChannels(nChannels)
    {}

    MatrixXf getPatch(long x, long y, MatrixBlock* block = nullptr)
    {
        long xBegin = std::max(x - rad, 0L);
        long yBegin = std::max(y - rad, 0L);
        long xEnd = std::min<long>(x + rad, w - 1);
        long yEnd = std::min<long>(y + rad, h - 1);
        long xSize = xEnd - xBegin + 1;
        long ySize = yEnd - yBegin + 1;

        MatrixXf patch(ySize * xSize, nChannels);
        long k = 0;
        for(long j = yBegin; j <= yEnd; ++j)
        for(long i = xBegin; i <= xEnd; ++i)
        {
            float* pixel = getPixel(i, j);
            for(int c = 0; c < nChannels; ++c)
                patch(k, c) = pixel[c];
            ++k;
        }

        if(block)
        {
            block->xBegin = xBegin;
            block->yBegin = yBegin;
            block->xSize = xSize;
            block->ySize = ySize;
        }
        return patch;
    }

    MatrixXf getDesignPatch(long x, long y, const MatrixBlock& block)
    {
        float* cPixel = getPixel(x, y);
        MatrixXf patch = MatrixXf::Ones(block.ySize * block.xSize, nChannels + 1);
        long k = 0;
        float xEnd = block.getXEnd();
        float yEnd = block.getYEnd();
        for(long j = block.yBegin; j <= yEnd; ++j)
        for(long i = block.xBegin; i <= xEnd; ++i)
        {
            float* pixel = getPixel(i, j);
            for(int c = 0; c < nChannels; ++c)
                patch(k, c+1) = pixel[c] - cPixel[c];
            ++k;
        }

        return patch;
    }

    void set(long x, long y, const MatrixBlock& block, const MatrixXf& patch)
    {
        long xEnd = block.getXEnd();
        long yEnd = block.getYEnd();
        long k = 0;
        for(long j = block.yBegin; j <= yEnd; ++j)
        for(long i = block.xBegin; i <= xEnd; ++i)
        {
            float* pixel = getPixel(i, j);
            for(int c = 0; c < nChannels; ++c)
                pixel[c] = patch(k, c);
            ++k;
        }
    }

    void add(long x, long y, const MatrixBlock& block, const MatrixXf& patch)
    {
        long xEnd = block.getXEnd();
        long yEnd = block.getYEnd();
        long k = 0;
        for(long j = block.yBegin; j <= yEnd; ++j)
        for(long i = block.xBegin; i <= xEnd; ++i)
        {
            float* pixel = getPixel(i, j);
            for(int c = 0; c < nChannels; ++c)
                pixel[c] += patch(k, c);
            ++k;
        }
    }

    static void generateBlock(int x, int y, int w, int h, int rad, MatrixBlock* block)
    {
        block->xBegin = std::max<long>(x - rad, 0L);
        block->yBegin = std::max<long>(y - rad, 0L);
        int xEnd = std::min<int>(x + rad, w - 1);
        int yEnd = std::min<int>(y + rad, h - 1);
        int xSize = xEnd - block->xBegin + 1;
        int ySize = yEnd - block->yBegin + 1;
        block->xSize = xSize;
        block->ySize = ySize;
    }

private:
    float* getPixel(long x, long y)
    { return &img[y*w*nChannels + x*nChannels]; }

    static constexpr int rad = 9;
    Buffer& img;
    size_t w;
    size_t h;
    int nChannels;
};


static void prefilterFeatures(MultiFilm* film, BufferSet& filteredFeatures)
{
    size_t w = film->xPixelCount;
    size_t h = film->yPixelCount;
    size_t numPixels = w*h;
    Buffer tmpVar2(numPixels*3);

#ifdef USE_FEATURE_PIXEL
    BufferSet pixelsFlt(2);
    pixelsFlt[0].resize(numPixels*2);
    pixelsFlt[1].resize(numPixels*2);
    for(size_t i = 0; i < film->statistics.pixel_mean_var.size(); ++i)
        tmpVar2[i] = film->statistics.pixel_mean_var[i]*2.0;

    // GPU NLM does not support 2 channel images
    nlm(film->buffers_pixel[0].data(), w, h, 2, 5, 3, 0.5,
        film->buffers_pixel[1].data(),
        tmpVar2.data(), 2, pixelsFlt[0].data()
    );
    nlm(film->buffers_pixel[1].data(), w, h, 2, 5, 3, 0.5,
        film->buffers_pixel[0].data(),
        tmpVar2.data(), 2, pixelsFlt[1].data()
    );
#endif

#ifdef USE_FEATURE_NORMALS
    BufferSet normalsFlt(2);
    normalsFlt[0].resize(numPixels*3);
    normalsFlt[1].resize(numPixels*3);
    for(size_t i = 0; i < film->statistics.normal_mean_var.size(); ++i)
        tmpVar2[i] = film->statistics.normal_mean_var[i]*2.0;

    gpuNlm(film->buffers_normal[0].data(), w, h, 3, 5, 3, 0.5,
        film->buffers_normal[1].data(),
        tmpVar2.data(), 3, normalsFlt[0].data()
    );
    gpuNlm(film->buffers_normal[1].data(), w, h, 3, 5, 3, 0.5,
        film->buffers_normal[0].data(),
        tmpVar2.data(), 3, normalsFlt[1].data()
    );
#ifdef PRINT_BUFFERS
    WriteImage("normals_0.exr", film->buffers_normal[0].data(), nullptr, w, h, w, h, 0, 0);
    WriteImage("normalsFlt_0.exr", normalsFlt[0].data(), nullptr, w, h, w, h, 0, 0);
#endif
#endif

#ifdef USE_FEATURE_ALBEDO
    BufferSet albedoFlt(2);
    albedoFlt[0].resize(numPixels*3);
    albedoFlt[1].resize(numPixels*3);
    for(size_t i = 0; i < film->statistics.albedo_mean_var.size(); ++i)
        tmpVar2[i] = film->statistics.albedo_mean_var[i]*2.0;

    gpuNlm(film->buffers_albedo[0].data(), w, h, 3, 5, 3, 0.5,
        film->buffers_albedo[1].data(),
        tmpVar2.data(), 3, albedoFlt[0].data()
    );
    gpuNlm(film->buffers_albedo[1].data(), w, h, 3, 5, 3, 0.5,
        film->buffers_albedo[0].data(),
        tmpVar2.data(), 3, albedoFlt[1].data()
    );
#ifdef PRINT_BUFFERS
    WriteImage("albedos_0.exr", film->buffers_albedo[0].data(), nullptr, w, h, w, h, 0, 0);
    WriteImage("albedosFlt_0.exr", albedoFlt[0].data(), nullptr, w, h, w, h, 0, 0);
#endif
#endif

#ifdef USE_FEATURE_DEPTH
    BufferSet depthFlt(2);
    depthFlt[0].resize(numPixels);
    depthFlt[1].resize(numPixels);
    for(size_t i = 0; i < film->statistics.depth_mean_var.size(); ++i)
        tmpVar2[i] = film->statistics.depth_mean_var[i]*2.0;

    gpuNlm(film->buffers_depth[0].data(), w, h, 1, 5, 3, 0.5,
        film->buffers_depth[1].data(),
        tmpVar2.data(), 1, depthFlt[0].data()
    );
    gpuNlm(film->buffers_depth[1].data(), w, h, 1, 5, 3, 0.5,
        film->buffers_depth[0].data(),
        tmpVar2.data(), 1, depthFlt[1].data()
    );
#endif

#ifdef USE_FEATURE_VISIBILITY
    BufferSet visFlt(2);
    visFlt[0].resize(numPixels);
    visFlt[1].resize(numPixels);
    for(size_t i = 0; i < film->statistics.visibility_mean_var.size(); ++i)
        tmpVar2[i] = film->statistics.visibility_mean_var[i]*2.0;

    gpuNlm(film->buffers_visibility[0].data(), w, h, 1, 5, 3, 0.5,
        film->buffers_visibility[1].data(),
        tmpVar2.data(), 1, visFlt[0].data()
    );
    gpuNlm(film->buffers_visibility[1].data(), w, h, 1, 5, 3, 0.5,
        film->buffers_visibility[0].data(),
        tmpVar2.data(), 1, visFlt[1].data()
    );
#endif

    copyFeatures(
    #ifdef USE_FEATURE_PIXEL
        pixelsFlt,
    #endif
    #ifdef USE_FEATURE_NORMALS
        normalsFlt,
    #endif
    #ifdef USE_FEATURE_ALBEDO
        albedoFlt,
    #endif
    #ifdef USE_FEATURE_DEPTH
        depthFlt,
    #endif
    #ifdef USE_FEATURE_VISIBILITY
        visFlt,
    #endif
        numPixels, filteredFeatures
    );
}


static void colaborativeRegression( Buffer& image,
                                    Buffer& features,
                                    Buffer& imageVar,
                                    float k,
                                    size_t width,
                                    size_t height,
                                    int nFeatureChannels,
                                    Buffer& result)
{
    size_t numPixels = width*height;

    // pre-compute weights for each pixel
    std::vector<VectorXf> nlmWeights(numPixels);
    #pragma omp parallel for
    for(size_t y = 0; y < height; ++y)
    {
        MatrixBlock block;
        for(size_t x = 0; x < width; ++x)
        {
            PatchUtil::generateBlock(x, y, width, height, 9, &block);
            nlmWeights[y*width + x] = computeNlmWeights(image.data(), imageVar.data(), width, height, block, x, y, 3, k);
        }
    }

    Buffer weights(numPixels, 0.f);
    PatchUtil weightsPatchUtil(weights, width, height, 1);
    PatchUtil resultPatchUtil(result, width, height, 3);
    PatchUtil imgPatchUtil(image, width, height, 3);
    PatchUtil featuresPatchUtil(features, width, height, nFeatureChannels);

    // lame ass parallelization with a critical section
    #pragma omp parallel for
    for(size_t y = 0; y < height; ++y)
    {
        MatrixBlock block;
        for(size_t x = 0; x < width; ++x)
        {
            MatrixXf image_P = imgPatchUtil.getPatch(x, y, &block);
            MatrixXf features_P = featuresPatchUtil.getDesignPatch(x, y, block);

            // Normalize features to [-1, 1]
            VectorXf minFeatures = features_P.colwise().minCoeff();
            VectorXf maxFeatures = features_P.colwise().maxCoeff();
            VectorXf normFac = (maxFeatures - minFeatures).array()*0.5f + 0.0001f;
            features_P = (features_P.rowwise() - minFeatures.transpose()).array().rowwise() / normFac.transpose().array();
            features_P = features_P.array() - 1.f;

            // Solve using SVD decomposition
            VectorXf& w = nlmWeights[y*width + x];
            MatrixXf& Y = image_P;
            MatrixXf& X = features_P;
            MatrixXf reconstruction = X * (w.asDiagonal()*X).jacobiSvd(ComputeThinU | ComputeThinV).solve(w.asDiagonal()*Y);

            // check for NaNs
            /*
            Matrix<bool, Dynamic, Dynamic> test = reconstruction.array().isFinite().matrix();
            size_t numFinites = test.cast<size_t>().sum();
            if(numFinites != reconstruction.rows()*reconstruction.cols())
            {
                std::cout << "Has non-finite elements" << std::endl;
                continue;
            }
            */

            #pragma omp critical
            {
                resultPatchUtil.add(x, y, block, reconstruction.array().colwise() * w.array());
                weightsPatchUtil.add(x, y, block, w);
            }
        }
    }

    #pragma omp parallel for
    for(size_t i = 0; i < numPixels; ++i)
    {
        if(weights[i] != 0)
        {
            result[i*3] /= weights[i];
            result[i*3] = std::max(result[i*3], 0.f);
            result[i*3 + 1] /= weights[i];
            result[i*3 + 1] = std::max(result[i*3 + 1], 0.f);
            result[i*3 + 2] /= weights[i];
            result[i*3 + 2] = std::max(result[i*3 + 2], 0.f);
        }
    }
}


void NFOR(MultiFilm* film, float* result)
{
//    const unsigned numThreads = std::thread::hardware_concurrency();
//    if(numThreads != 0)
//        omp_set_num_threads(numThreads);
//    else
//        omp_set_num_threads(4);

    film->UpdateVariances();
    size_t w = film->xPixelCount;
    size_t h = film->yPixelCount;
    size_t numPixels = w*h;
    Buffer& color = film->statistics.rgb_mean;
    Buffer& colorVar = film->statistics.rgb_mean_var;
    Buffer colorVar2(colorVar.size());
    for(size_t i = 0; i < colorVar.size(); ++i)
        colorVar2[i] = colorVar[i]*2.0;

    // prefilter features
    BufferSet filteredFeatures;
    prefilterFeatures(film, filteredFeatures);

    Buffer featuresVar;
    copyVariances(
            #ifdef USE_FEATURE_PIXEL
                film->statistics.pixel_mean_var,
            #endif
            #ifdef USE_FEATURE_NORMALS
                film->statistics.normal_mean_var,
            #endif
            #ifdef USE_FEATURE_ALBEDO
                film->statistics.albedo_mean_var,
            #endif
            #ifdef USE_FEATURE_DEPTH
                film->statistics.depth_mean_var,
            #endif
            #ifdef USE_FEATURE_VISIBILITY
                film->statistics.visibility_mean_var,
            #endif
                numPixels, featuresVar
                );

    // Main regression
    float ks[2] = {0.5f, 1.0f};
    BufferSet filteredColorsA(2);
    BufferSet filteredColorsB(2);
    BufferSet mses(2);
    mses[0].resize(numPixels*3);
    mses[1].resize(numPixels*3);
    for(int i = 0; i < 2; ++i)
    {
        // Regression pass
        Buffer filteredColorA(color.size());
        Buffer filteredColorB(color.size());
        colaborativeRegression(film->buffers_rgb[0], filteredFeatures[1], colorVar2, ks[i], w, h, FEATURE_SIZE, filteredColorA);
        colaborativeRegression(film->buffers_rgb[1], filteredFeatures[0], colorVar2, ks[i], w, h, FEATURE_SIZE, filteredColorB);
        filteredColorsA[i] = filteredColorA;
        filteredColorsB[i] = filteredColorB;
#ifdef PRINT_BUFFERS
        WriteImage("filteredColorA_" + std::to_string(i) + ".exr", filteredColorA.data(), nullptr, w, h, w, h, 0, 0);
        WriteImage("filteredColorB_" + std::to_string(i) + ".exr", filteredColorB.data(), nullptr, w, h, w, h, 0, 0);
#endif

        // MSE estimation
        Buffer noisyMse(numPixels*3);
        #pragma omp parallel for
        for(size_t j = 0; j < numPixels; ++j)
        {
            for(int c = 0; c < 3; ++c)
            {
                float tmpA = film->buffers_rgb[1][j*3 + c] - filteredColorA[j*3 + c];
                float mseA = tmpA*tmpA - colorVar2[j*3 + c];
                float tmpB = film->buffers_rgb[0][j*3 + c] - filteredColorB[j*3 + c];
                float mseB = tmpB*tmpB - colorVar2[j*3 + c];
                float tmp = filteredColorB[j*3 + c] - filteredColorA[j*3 + c];
                float residualColorVar = tmp*tmp * 0.25f;
                noisyMse[j*3 + c] = (mseA + mseB)*0.5f - residualColorVar;
            }
        }
        //ORIGINAL: mses[i] = nlMeans(noisyMse, color, colorVar, 1, 9, 1.0f);
        gpuNlm(noisyMse.data(), w, h, 3, 9, 1, 1.0f, color.data(), colorVar.data(), 3, mses[i].data());
#ifdef PRINT_BUFFERS
        WriteImage("noisyMSE_" + std::to_string(i) + ".exr", noisyMse.data(), nullptr, w, h, w, h, 0, 0);
        WriteImage("mse_" + std::to_string(i) + ".exr", mses[i].data(), nullptr, w, h, w, h, 0, 0);
#endif
    }

    // Bandwidth selection
    Buffer resultA(numPixels*3), resultB(numPixels*3);
    for(int i=0; i < 2; ++i)
    {
        Buffer noisySelection(numPixels*3, 0.f);
        #pragma omp parallel for
        for(size_t j = 0; j < numPixels; ++j)
            for(int c = 0; c < 3; ++c)
                if(mses[i][j*3 + c] < mses[1 - i][j*3 + c])
                    noisySelection[j*3 + c] = 1.f;
                else if(mses[i][j*3 + c] == mses[1 - i][j*3 + c])
                    noisySelection[j*3 + c] = i == 0 ? 1.f : 0.f;

        Buffer selection(numPixels*3);
        //ORIGINAL: selection = nlMeans(noisySelection, color, colorVar, F=1, R=9, k=1.0)
        gpuNlm(noisySelection.data(), w, h, 3, 9, 1, 1.0f, color.data(), colorVar.data(), 3, selection.data());
#ifdef PRINT_BUFFERS
        WriteImage("noisySelection_" + std::to_string(i) + ".exr", noisySelection.data(), nullptr, w, h, w, h, 0, 0);
        WriteImage("selection_" + std::to_string(i) + ".exr", selection.data(), nullptr, w, h, w, h, 0, 0);
#endif

        #pragma omp parallel for
        for(size_t j = 0; j < numPixels; ++j)
        for(int c = 0; c < 3; ++c)
        {
            resultA[j*3 + c] += filteredColorsA[i][j*3 + c] * selection[j*3 + c];
            resultB[j*3 + c] += filteredColorsB[i][j*3 + c] * selection[j*3 + c];
        }
    }

    // Second filter pass (section 5.4)
    Buffer finalFeatures(numPixels*FEATURE_SIZE);
    {
//        Buffer combinedFeature(numPixels*FEATURE_SIZE);
//        Buffer combinedFeatureVar(numPixels*FEATURE_SIZE);
//        #pragma omp parallel for
//        for(size_t j = 0; j < numPixels; ++j)
//        for(int c = 0; c < FEATURE_SIZE; ++c)
//        {
//            combinedFeature[j*FEATURE_SIZE + c] = (filteredFeatures[0][j*FEATURE_SIZE + c] + filteredFeatures[1][j*FEATURE_SIZE + c])*0.5f;
//            float tmp = filteredFeatures[1][j*FEATURE_SIZE + c] - filteredFeatures[0][j*FEATURE_SIZE + c];
//            combinedFeatureVar[j*FEATURE_SIZE + c] = tmp*tmp * 0.25f;
//        }
//        nlm(combinedFeature.data(), w, h, FEATURE_SIZE, 3, 2, 0.5f, combinedFeature.data(), combinedFeatureVar.data(), FEATURE_SIZE, finalFeatures.data());

        Buffer tmpFinalFeat(numPixels*3);
        Buffer combinedFeature(numPixels*3);
        Buffer combinedFeatureVar(numPixels*3);
        //Pixel
#ifdef USE_FEATURE_PIXEL
        for(size_t j = 0; j < numPixels; ++j)
        for(int c = 0; c < 2; ++c)
        {
            combinedFeature[j*2 + c] = (filteredFeatures[0][j*FEATURE_SIZE + IMAGE_X + c] + filteredFeatures[1][j*FEATURE_SIZE + IMAGE_X + c])*0.5f;
            float tmp = filteredFeatures[1][j*FEATURE_SIZE + IMAGE_X + c] - filteredFeatures[0][j*FEATURE_SIZE + IMAGE_X + c];
            combinedFeatureVar[j*2 + c] = tmp*tmp * 0.25f;
        }
        nlm(combinedFeature.data(), w, h, 2, 3, 2, 0.5f, combinedFeature.data(), combinedFeatureVar.data(), 2, tmpFinalFeat.data());
        for(size_t j = 0; j < numPixels; ++j)
        for(int c = 0; c < 2; ++c)
            finalFeatures[j*FEATURE_SIZE + IMAGE_X + c] = tmpFinalFeat[j*2 + c];
#endif
#ifdef USE_FEATURE_ALBEDO
        //Albedo
        for(size_t j = 0; j < numPixels; ++j)
        for(int c = 0; c < 3; ++c)
        {
            combinedFeature[j*3 + c] = (filteredFeatures[0][j*FEATURE_SIZE + ALBEDO_R + c] + filteredFeatures[1][j*FEATURE_SIZE + ALBEDO_R + c])*0.5f;
            float tmp = filteredFeatures[1][j*FEATURE_SIZE + ALBEDO_R + c] - filteredFeatures[0][j*FEATURE_SIZE + ALBEDO_R + c];
            combinedFeatureVar[j*3 + c] = tmp*tmp * 0.25f;
        }
        gpuNlm(combinedFeature.data(), w, h, 3, 3, 2, 0.5f, combinedFeature.data(), combinedFeatureVar.data(), 3, tmpFinalFeat.data());
        for(size_t j = 0; j < numPixels; ++j)
        for(int c = 0; c < 3; ++c)
            finalFeatures[j*FEATURE_SIZE + ALBEDO_R + c] = tmpFinalFeat[j*3 + c];
#endif
#ifdef USE_FEATURE_NORMALS
        //Normal
        for(size_t j = 0; j < numPixels; ++j)
        for(int c = 0; c < 3; ++c)
        {
            combinedFeature[j*3 + c] = (filteredFeatures[0][j*FEATURE_SIZE + NORMAL_X + c] + filteredFeatures[1][j*FEATURE_SIZE + NORMAL_X + c])*0.5f;
            float tmp = filteredFeatures[1][j*FEATURE_SIZE + NORMAL_X + c] - filteredFeatures[0][j*FEATURE_SIZE + NORMAL_X + c];
            combinedFeatureVar[j*3 + c] = tmp*tmp * 0.25f;
        }
        gpuNlm(combinedFeature.data(), w, h, 3, 3, 2, 0.5f, combinedFeature.data(), combinedFeatureVar.data(), 3, tmpFinalFeat.data());
        for(size_t j = 0; j < numPixels; ++j)
        for(int c = 0; c < 3; ++c)
            finalFeatures[j*FEATURE_SIZE + NORMAL_X + c] = tmpFinalFeat[j*3 + c];
#endif
#ifdef USE_FEATURE_DEPTH
        //Depth
        for(size_t j = 0; j < numPixels; ++j)
        {
            combinedFeature[j] = (filteredFeatures[0][j*FEATURE_SIZE + DEPTH] + filteredFeatures[1][j*FEATURE_SIZE + DEPTH])*0.5f;
            float tmp = filteredFeatures[1][j*FEATURE_SIZE + DEPTH] - filteredFeatures[0][j*FEATURE_SIZE + DEPTH];
            combinedFeatureVar[j] = tmp*tmp * 0.25f;
        }
        gpuNlm(combinedFeature.data(), w, h, 1, 3, 2, 0.5f, combinedFeature.data(), combinedFeatureVar.data(), 1, tmpFinalFeat.data());
        for(size_t j = 0; j < numPixels; ++j)
            finalFeatures[j*FEATURE_SIZE + DEPTH] = tmpFinalFeat[j];
#endif
#ifdef USE_FEATURE_VISIBILITY
        //Visibility
        for(size_t j = 0; j < numPixels; ++j)
        {
            combinedFeature[j] = (filteredFeatures[0][j*FEATURE_SIZE + VISIBILITY] + filteredFeatures[1][j*FEATURE_SIZE + VISIBILITY])*0.5f;
            float tmp = filteredFeatures[1][j*FEATURE_SIZE + VISIBILITY] - filteredFeatures[0][j*FEATURE_SIZE + VISIBILITY];
            combinedFeatureVar[j] = tmp*tmp * 0.25f;
        }
        gpuNlm(combinedFeature.data(), w, h, 1, 3, 2, 0.5f, combinedFeature.data(), combinedFeatureVar.data(), 1, tmpFinalFeat.data());
        for(size_t j = 0; j < numPixels; ++j)
            finalFeatures[j*FEATURE_SIZE + VISIBILITY] = tmpFinalFeat[j];
#endif
    }

    Buffer combinedResult(numPixels*3);
    Buffer combinedResultVar(numPixels*3);
    #pragma omp parallel for
    for(size_t j = 0; j < numPixels; ++j)
    for(int c = 0; c < 3; ++c)
    {
        combinedResult[j*3 + c] = (resultA[j*3 + c] + resultB[j*3 + c])*0.5f;
        float tmp = resultB[j*3 + c] - resultA[j*3 + c];
        combinedResultVar[j*3 + c] = tmp*tmp * 0.25f;
    }

    Buffer resultBuffer(numPixels*3);
    colaborativeRegression(combinedResult, finalFeatures, combinedResultVar, 1.0f, w, h, FEATURE_SIZE, resultBuffer);
    memcpy(result, resultBuffer.data(), numPixels*3*sizeof(float));
}


using MapImg = Map<MatrixNf, Unaligned, Stride<Dynamic, Dynamic>>;

inline MapImg mapImgChannel(float* img, size_t w, size_t h, int nChannels, int channel)
{ return MapImg(img + channel, h, w, Stride<Dynamic, Dynamic>(w*nChannels, nChannels)); }

inline MapImg mapImg(float* img, size_t w, size_t h, int nChannels)
{ return MapImg(img, w*h, nChannels, Stride<Dynamic, Dynamic>(nChannels, 1)); }

#define MAP_IMG_BLOCK(img, w, h, nChannels, xb, yb, xs, ys) \
    MapImg(img, xs*ys, nChannels, Stride<Dynamic, Dynamic>(w, 1))

void testBlock()
{
    int w = 500;
    int h = 300;
    int nc = 3;
    std::unique_ptr<float[]> img(new float[w*h*nc]);
    memset(img.get(), 0, w*h*3*sizeof(float));

    MatrixBlock P(250, 150, w, h, 50);
//    MatrixXf p = m.block(P.yBegin, P.xBegin, P.ySize, P.xSize);

    mapImgChannel(img.get(), w, h, 3, 0) = MatrixXf::Constant(h, w, 0.1f);
    mapImgChannel(img.get(), w, h, 3, 1) = MatrixXf::Constant(h, w, 0.2f);
    mapImgChannel(img.get(), w, h, 3, 2).block(P.yBegin, P.xBegin, P.ySize, P.xSize) += MatrixXf::Constant(h, w, 0.3f);

//    mapImg(img.get(), w, h, 3).row(150*w + 250) = VectorXf::Constant(3, 1.f);
    MAP_IMG_BLOCK(img.get(), w, h, 3, P.xBegin, P.yBegin, P.xSize, P.ySize) = MatrixXf::Constant(P.xSize*P.ySize, 3, 1.f);

    WriteImage("block_test.exr", img.get(), nullptr, w, h, w, h, 0, 0);
}


void testFilter(MultiFilm* film, float* result)
{
    film->UpdateVariances();

    size_t w = film->xPixelCount;
    size_t h = film->yPixelCount;
    size_t numPixels = w * h;
    std::vector<float>& input = film->statistics.rgb_mean;
    std::vector<float>& inputVar = film->statistics.rgb_mean_var;

    std::vector<float> rgbCPU(numPixels*3, 0.f);
    std::vector<float> rgbGPU(numPixels*3, 0.f);

    nlm(film->statistics.rgb_mean.data(), w, h, 3, 7, 3, 1.0f,
        film->statistics.rgb_mean.data(),
        film->statistics.rgb_mean_var.data(), 3, rgbCPU.data());
    gpuNlm(film->statistics.rgb_mean.data(), w, h, 3, 7, 3, 1.0f,
           film->statistics.rgb_mean.data(),
           film->statistics.rgb_mean_var.data(), 3, rgbGPU.data());
    WriteImage("rgb_cpu_1.exr", rgbCPU.data(), nullptr, w, h, w, h, 0, 0);
    WriteImage("rgb_gpu_1.exr", rgbGPU.data(), nullptr, w, h, w, h, 0, 0);

    nlm(film->statistics.rgb_mean.data(), w, h, 3, 7, 3, 0.5f,
        film->statistics.rgb_mean.data(),
        film->statistics.rgb_mean_var.data(), 3, rgbCPU.data());
    gpuNlm(film->statistics.rgb_mean.data(), w, h, 3, 7, 3, 0.5f,
           film->statistics.rgb_mean.data(),
           film->statistics.rgb_mean_var.data(), 3, rgbGPU.data());
    WriteImage("rgb_cpu_05.exr", rgbCPU.data(), nullptr, w, h, w, h, 0, 0);
    WriteImage("rgb_gpu_05.exr", rgbGPU.data(), nullptr, w, h, w, h, 0, 0);

    Buffer ones(numPixels*3, 1.f);
    Buffer point(numPixels*3, 0.f);
    memset(rgbCPU.data(), 0, rgbCPU.size()*sizeof(float));
    memset(rgbGPU.data(), 0, rgbGPU.size()*sizeof(float));
    point[100*w*3 + 100*3] = 1.f;
    point[100*w*3 + 100*3 + 1] = 1.f;
    point[100*w*3 + 100*3 + 2] = 1.f;
    nlm(point.data(), w, h, 3, 7, 3, 0.5f,
        point.data(),
        ones.data(), 3, rgbCPU.data());
    gpuNlm(point.data(), w, h, 3, 7, 3, 0.5f,
           point.data(),
           ones.data(), 3, rgbGPU.data());
    WriteImage("response_cpu_05.exr", rgbCPU.data(), nullptr, w, h, w, h, 0, 0);
    WriteImage("response_gpu_05.exr", rgbGPU.data(), nullptr, w, h, w, h, 0, 0);

    MatrixBlock block;
    PatchUtil::generateBlock(100, 100, w, h, 7, &block);
    VectorXf cpuWeights = computeNlmWeights(film->statistics.rgb_mean.data(), film->statistics.rgb_mean_var.data(),
                      w, h, block, 100, 100, 3, 0.5f);
    VectorXf gpuWeights = gpuComputeNlmWeights(film->statistics.rgb_mean.data(), film->statistics.rgb_mean_var.data(),
                      w, h, 100, 100, 7, 3, 0.5f);
    std::cout << "cpu weights = \n" << cpuWeights << std::endl;
    std::cout << "gpu weights = \n" << gpuWeights << std::endl;

}
