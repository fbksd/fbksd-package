
#include "Benchmark/BenchmarkClient/BenchmarkClient.h"
#include "gaussian.h"
#include "lwr_sampler.h"
#include "lwr_film.h"
#include "gaussian.h"
#include <memory>
#include <iostream>



int main(int argc, char **argv)
{
    BenchmarkClient client;
    SceneInfo sceneInfo = client.getSceneInfo();
    int width = sceneInfo.get<int>("width");
    int height = sceneInfo.get<int>("height");
    int spp = sceneInfo.get<int>("max_spp");
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
    int initSpp = nIterations > 0 ? std::min(4, spp) : spp;

    GaussianFilter filter(2.f, 2.f, 2.f);
    LWR_Film film(width, height, &filter, rayScale);
    LWR_Sampler sampler(0, width, 0, height, initSpp, spp, shutterOpen, shutterClose, nIterations, &film);
    film.initializeGlobalVariables(sampler.GetInitSPP());
    film.generateScramblingInfo(0, 0);

    float* samples = client.getSamplesBuffer();
    size_t numPixels = width*height;

    client.evaluateSamples(BenchmarkClient::SAMPLES_PER_PIXEL, initSpp);

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

    /*
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
                int nSamples = 0;
                int pixIdx = 0;
                while(nSamples = sampler.GetMoreSamplesWithIdx(samples, rng, pixIdx))
                {
                    client.evaluateSamples(BenchmarkClient::SAMPLES, nSamples);

                    for(int j = 0; j < nSamples; ++j)
                    {
                        float x = samples[j*sampleSize + IMAGE_X];
                        float y = samples[j*sampleSize + IMAGE_Y];
                        if(x < 0 || x > width || y < 0 || y > height)
                        {
                            std::cout << "ERROR: sample outside of image range generated! [" << x << ", " << y << "]" << std::endl;
                            continue;
                        }

                        float depth = samples[j*sampleSize + DEPTH];
                        if(std::isinf(depth))
                            samples[j*sampleSize + DEPTH] = 1000.f;

                        film.AddSampleExtended(&samples[j*sampleSize], pixIdx);
                    }
                }
            }
        }
    }
    */

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
                    nSamplesVec.push_back(nSamples);
                    pixIdxVec.push_back(pixIdx);
                    totalSamples += nSamples;
                    curSamples += nSamples*sampleSize;
                }

                client.evaluateSamples(BenchmarkClient::SAMPLES, totalSamples);
                curSamples = samples;
                for(size_t j = 0; j < nSamplesVec.size(); ++j)
                {
                    int nSamples = nSamplesVec[j];
                    int pixIdx = pixIdxVec[j];
                    for(size_t k = 0; k < nSamples; ++k)
                    {
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

    float* result = client.getResultBuffer();
    film.WriteImage(result);
    client.sendResult();
    return 0;
}
