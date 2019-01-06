#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;

#include "gaussian.h"
#include "lwr_sampler.h"
#include "lwr_film.h"
#include "gaussian.h"
#include <memory>
#include <iostream>


#include <ImfOutputFile.h>
#include <ImfChannelList.h>
using namespace Imath;

static void saveExrGray(const std::string& filename, int w, int h, float* layer)
{
    Imf::Header header(w, h);
    header.channels().insert ("Y", Imf::Channel(Imf::FLOAT));

    Imf::OutputFile file(filename.c_str(), header);
    Imf::FrameBuffer frameBuffer;

    frameBuffer.insert("Y", Imf::Slice( Imf::PixelType(Imf::FLOAT), (char*)layer, sizeof(float), sizeof(float)*w));

    file.setFrameBuffer(frameBuffer);
    file.writePixels(h);
}

static void saveExrRgb(const std::string& filename, int w, int h, float* img)
{
    size_t numPixels = w * h;
    Imf::Header header(w, h);
    header.channels().insert ("R", Imf::Channel(Imf::FLOAT));
    header.channels().insert ("G", Imf::Channel(Imf::FLOAT));
    header.channels().insert ("B", Imf::Channel(Imf::FLOAT));

    Imf::OutputFile file(filename.c_str(), header);
    Imf::FrameBuffer frameBuffer;
    frameBuffer.insert ("R", Imf::Slice( Imf::PixelType(Imf::FLOAT), (char*)img, sizeof(float)*3, sizeof(float)*w*3));
    frameBuffer.insert ("G", Imf::Slice( Imf::PixelType(Imf::FLOAT), (char*)(img + 1), sizeof(float)*3, sizeof(float)*w*3));
    frameBuffer.insert ("B", Imf::Slice( Imf::PixelType(Imf::FLOAT), (char*)(img + 2), sizeof(float)*3, sizeof(float)*w*3));

    file.setFrameBuffer(frameBuffer);
    file.writePixels(h);
}


int main(int argc, char **argv)
{
    BenchmarkClient client;
    SceneInfo sceneInfo = client.getSceneInfo();
    auto width = sceneInfo.get<int64_t>("width");
    auto height = sceneInfo.get<int64_t>("height");
    auto spp = sceneInfo.get<int64_t>("max_spp");
    float shutterOpen = sceneInfo.get<float>("shutter_open");
    float shutterClose = sceneInfo.get<float>("shutter_close");
    std::cout << "Img size: [" << width << ", " << height << "]" << std::endl;

    SampleLayout layout;
    layout("IMAGE_X")("IMAGE_Y")
          ("COLOR_R")("COLOR_G")("COLOR_B")
          ("NORMAL_X_NS")("NORMAL_Y_NS")("NORMAL_Z_NS")
          ("TEXTURE_COLOR_R_NS")("TEXTURE_COLOR_G_NS")("TEXTURE_COLOR_B_NS")
          ("DEPTH");

    int sampleSize = layout.getSampleSize();
    client.setSampleLayout(layout);

    // Parameters
    int nIterations = 8;
    float rayScale = 0.f;
    auto initSpp = nIterations > 0 ? std::min(INT64_C(4), spp) : spp;

    GaussianFilter filter(2.f, 2.f, 2.f);
    LWR_Film film(width, height, &filter, rayScale);
    LWR_Sampler sampler(0, width, 0, height, initSpp, spp, shutterOpen, shutterClose, nIterations, &film);
    film.initializeGlobalVariables(sampler.GetInitSPP());
    film.generateScramblingInfo(0, 0);

    float* samples = client.getSamplesBuffer();
    size_t numPixels = width*height;
    size_t maxSampleBudget = spp * numPixels;
    size_t currentUsedBudget = initSpp * numPixels;
    client.evaluateSamples(SPP(initSpp));

    float maxDepth = 0;
    for(size_t i = 0; i < initSpp * numPixels; ++i)
    {
        float v = samples[i*sampleSize + DEPTH];
        if(std::isinf(v))
            samples[i*sampleSize + DEPTH] = 1000.f;
        else if(v > maxDepth)
            maxDepth = v;
    }
    std::cout << "Max depth = " << maxDepth << std::endl;

    film.m_maxDepth = maxDepth;
    film.m_samplesPerPixel = sampler.samplesPerPixel;

    for(size_t p = 0; p < numPixels; ++p)
    {
        for(size_t s = 0; s < initSpp; ++s)
            film.AddSampleExtended(&samples[p*initSpp*sampleSize + s*sampleSize], p);
    }

    std::vector<float> adaptiveImg(numPixels * 3, 0.f);
    std::vector<float> sppCountImg(numPixels, 0.f);

    if(nIterations > 0)
    {
        int numSamplePerIterations = (sampler.samplesPerPixel - sampler.GetInitSPP()) * numPixels / nIterations;
        if(numSamplePerIterations > 0)
        {
            layout.setElementIO(0, SampleLayout::INPUT);
            layout.setElementIO(1, SampleLayout::INPUT);
            client.setSampleLayout(layout);

            RNG rng;
            for(int i = 0; i < nIterations; ++i)
            {
                film.test_lwrr(numSamplePerIterations);
                sampler.SetMaximumSampleCount(film.m_maxSPP);
                int nSamples = 0;
                std::vector<int> nSamplesVec;
                int pixIdx = 0;
                std::vector<int> pixIdxVec;
                size_t totalSamples = 0;
                float* curSamples = samples;
                while(nSamples = sampler.GetMoreSamplesWithIdx(curSamples, rng, pixIdx))
                {
                    currentUsedBudget += nSamples;
                    if(currentUsedBudget > maxSampleBudget)
                    {
                        std::cout << "Sample budget exceeded!" << std::endl;
                        i = nIterations;
                        break;
                    }
                    nSamplesVec.push_back(nSamples);
                    pixIdxVec.push_back(pixIdx);
                    totalSamples += nSamples;
                    curSamples += nSamples*sampleSize;
                }

                std::cout << "Adaptive iterationSample budget exceeded!" << std::endl;
                printf("Adaptive iteration %d of %d. # samples = %d\n", i+1, nIterations, totalSamples);
                client.evaluateSamples(totalSamples);
                curSamples = samples;
                for(size_t j = 0; j < nSamplesVec.size(); ++j)
                {
                    int nSamples = nSamplesVec[j];
                    int pixIdx = pixIdxVec[j];
                    for(size_t k = 0; k < nSamples; ++k)
                    {
                        float cx = curSamples[k*sampleSize + IMAGE_X];
                        float cy = curSamples[k*sampleSize + IMAGE_Y];
                        int ix = cx;
                        int iy = cy;
                        adaptiveImg[iy*width*3 + ix*3] += curSamples[k*sampleSize + COLOR_R];
                        adaptiveImg[iy*width*3 + ix*3 + 1] += curSamples[k*sampleSize + COLOR_G];
                        adaptiveImg[iy*width*3 + ix*3 + 2] += curSamples[k*sampleSize + COLOR_B];
                        sppCountImg[iy*width + ix] += 1.f;

                        float depth = curSamples[k*sampleSize + DEPTH];
                        if(std::isinf(depth))
                            curSamples[k*sampleSize + DEPTH] = 1000.f;

                        film.AddSampleExtended(&curSamples[k*sampleSize], pixIdx);
                    }

                    curSamples += nSamples*sampleSize;
                }
            }
        }
    }

    for(size_t i = 0; i < numPixels; ++i)
    {
        if(sppCountImg[i] > 0.5f)
        {
            float inv_spp = 1.f / sppCountImg[i];
            adaptiveImg[i*3] *= inv_spp;
            adaptiveImg[i*3 + 1] *= inv_spp;
            adaptiveImg[i*3 + 2] *= inv_spp;
        }
    }
    saveExrRgb("adaptive_img.exr", width, height, adaptiveImg.data());
    saveExrGray("spp.exr", width, height, sppCountImg.data());

    float* result = client.getResultBuffer();
    film.WriteImage(result);
    client.sendResult();
    return 0;
}
