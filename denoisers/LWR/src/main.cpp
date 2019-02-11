#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;

#include "gaussian.h"
#include "lwr_sampler.h"
#include "lwr_film.h"
#include "gaussian.h"
#include <memory>
#include <iostream>


int main(int argc, char* argv[])
{
    BenchmarkClient client(argc, argv);
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
          ("NORMAL_X")("NORMAL_Y")("NORMAL_Z")
          ("TEXTURE_COLOR_R")("TEXTURE_COLOR_G")("TEXTURE_COLOR_B")
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

    client.evaluateSamples(SPP(initSpp), [&](const BufferTile& tile)
    {
        for(auto y = tile.beginY(); y < tile.endY(); ++y)
        for(auto x = tile.beginX(); x < tile.endX(); ++x)
        for(size_t s = 0; s < initSpp; ++s)
        {
            float* sample = tile(x, y, s);
            float depth = sample[DEPTH];
            if(std::isinf(depth))
                sample[DEPTH] = 1000.f;
            film.AddSampleExtended(sample, y*width + x);
        }
    });

    float* result = client.getResultBuffer();
    int64_t numPixels = width * height;
    std::vector<int> samplesCount(numPixels, 0);

    if(nIterations > 0)
    {
        int numSamplesPerIterations = (sampler.samplesPerPixel - sampler.GetInitSPP()) * numPixels / nIterations;
        if(numSamplesPerIterations > 0)
        {
            layout.setElementIO(0, SampleLayout::INPUT);
            layout.setElementIO(1, SampleLayout::INPUT);
            client.setSampleLayout(layout);
            std::vector<float> samplesBuffer;
            samplesBuffer.reserve(sampleSize * numSamplesPerIterations);

            RNG rng;
            for(int i = 0; i < nIterations; ++i)
            {
                film.test_lwrr(numSamplesPerIterations);
                sampler.SetMaximumSampleCount(film.m_maxSPP);
                int nSamples = 0;
                std::vector<int> nSamplesVec;
                int pixIdx = 0;
                std::vector<int> pixIdxVec;
                size_t totalSamples = 0;
                while(nSamples = sampler.GetMoreSamplesWithIdx(&samplesBuffer, rng, pixIdx))
                {
                    nSamplesVec.push_back(nSamples);
                    pixIdxVec.push_back(pixIdx);
                    totalSamples += nSamples;
                }

                float* currentInSample = samplesBuffer.data();
                client.evaluateInputSamples(totalSamples,
                    [&](int64_t count, float* samples)
                    {
                        for(int i = 0; i < count; ++i)
                        {
                            samples[i*sampleSize + IMAGE_X] = currentInSample[0];
                            samples[i*sampleSize + IMAGE_Y] = currentInSample[1];
                            currentInSample += 2;
                        }
                    },
                    [&](int64_t count, float* samples)
                    {
                        for(int i = 0; i < count; ++i)
                        {
                            float* sample = &samples[i*sampleSize];
                            float depth = sample[DEPTH];
                            if(std::isinf(depth))
                                sample[DEPTH] = 1000.f;
                            int pixIdx = ((int)sample[IMAGE_Y])*width + (int)sample[IMAGE_X];
                            film.AddSampleExtended(sample, pixIdx);
                        }
                    }
                );

                samplesBuffer.clear();
            }
        }
    }

    film.WriteImage(result);
    client.sendResult();
    return 0;
}
