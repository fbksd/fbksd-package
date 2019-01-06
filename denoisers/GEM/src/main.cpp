#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;

#include "gaussian.h"
#include "bandwidth.h"
#include "smooth.h"
#include <memory>
#include <iostream>


int main(int argc, char **argv)
{
    BenchmarkClient client;
    SceneInfo sceneInfo = client.getSceneInfo();
    const auto width = sceneInfo.get<int64_t>("width");
    const auto height = sceneInfo.get<int64_t>("height");
    const auto spp = sceneInfo.get<int64_t>("max_spp");

    std::cout << "Image resolution: " << width << " x " << height << std::endl;

    SampleLayout layout;
    layout("IMAGE_X")("IMAGE_Y")("COLOR_R")("COLOR_G")("COLOR_B");
    int sampleSize = layout.getSampleSize();
    client.setSampleLayout(layout);

    // Parameters
    float gamma = 0.2f;
    float threshold = std::numeric_limits<float>::infinity();
    int nIterations = 8;

    auto initSpp = std::min(INT64_C(4), spp);
    std::cout << "Initial sampling:  " << initSpp << " spp ..." << std::endl;
    client.evaluateSamples(SPP(initSpp));
    std::cout << "finished initial sampling" << std::endl;

    GaussianFilter filter(2.f, 2.f, 2.f);
    SmoothFilm film(width, height, &filter, gamma);
    float* samples = client.getSamplesBuffer();
    size_t numPixels = width*height;
    std::cout << "num pixels: " << numPixels << std::endl;
    size_t numInitSamples = initSpp * numPixels;
    for (size_t i = 0; i < numInitSamples; ++i)
        film.AddSample(&samples[i*sampleSize]);

    BandwidthSampler sampler(0, width, 0, height, spp, 0.f, 1.f, threshold, nIterations, &film);
    if (sampler.PixelsToSampleTotal() > 0)
    {
        layout.setElementIO(0, SampleLayout::INPUT);
        layout.setElementIO(1, SampleLayout::INPUT);
        client.setSampleLayout(layout);

        sampler.SetAdaptiveMode();
        int nIterations = sampler.GetIterationCount();
        int nPixelsPerIteration = Ceil2Int(float(sampler.PixelsToSampleTotal()) / nIterations);
        RNG rng;
        int it = 1;
        size_t totalSampleCount = 0;
        while (sampler.PixelsToSampleTotal() > 0) {
            std::cout << "Adaptive iteration " << it++ << " of " << nIterations << std::endl;
            sampler.GetWorstPixels(nPixelsPerIteration);

            int nSamples = 0;
            size_t nSum = 0;
            while(nSamples = sampler.GetMoreSamples(&samples[nSum*sampleSize], rng))
                nSum += nSamples;

            totalSampleCount += nSum;
            std::cout << "num samples: " << nSum << std::endl;
            client.evaluateSamples(nSum);
            for(size_t i = 0; i < nSum; ++i)
            {
                //if(it == 3 && i == 26882)
                film.AddSample(&samples[i*sampleSize]);
            }
        }
        std::cout << "Finished adaptive sampling: " << totalSampleCount << " samples (" << totalSampleCount/(float)numPixels << " spp)" << std::endl;
    }

    float* result = client.getResultBuffer();
    film.WriteImage(result);
    client.sendResult();
    return 0;
}
