#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;
#include <memory>
#include "gaussian.h"
#include "dualfilm.h"
#include "dualsampler.h"


int main(int argc, char **argv)
{
    BenchmarkClient client;
    SceneInfo sceneInfo = client.getSceneInfo();
    auto width = sceneInfo.get<int64_t>("width");
    auto height = sceneInfo.get<int64_t>("height");
    auto spp = sceneInfo.get<int64_t>("max_spp");
    float shutterOpen = sceneInfo.get<float>("shutter_open");
    float shutterClose = sceneInfo.get<float>("shutter_close");

    SampleLayout layout;
    layout("IMAGE_X")("IMAGE_Y")("COLOR_R")("COLOR_G")("COLOR_B");
    int sampleSize = layout.getSampleSize();
    client.setSampleLayout(layout);

    // Parameters
    int wnd_rad = 10; // Filter width
    float k = 0.45f;   // Damping parameter (results for our paper used 0.1)
    int ptc_rad = 3;  // Patchsize
    float threshold = std::numeric_limits<float>::infinity();
    int nIterations = 8;
    int64_t initSpp = 0;
    if(nIterations > 0)
        initSpp = std::min(INT64_C(4), spp);
    else
        initSpp = spp;

    GaussianFilter filter(2.f, 2.f, 2.f);
    DualFilm film(width, height, &filter, wnd_rad, k, ptc_rad);
    float* samples = client.getSamplesBuffer();
    size_t numPixels = width*height;

    client.evaluateSamples(SPP(initSpp));
    // give half the samples for each buffer
    for(size_t p = 0; p < numPixels; ++p)
    {
        for(size_t s = 0; s < initSpp/2; ++s)
            film.AddSample(&samples[p*initSpp*sampleSize + s*sampleSize], BUFFER_A);
        for(size_t s = initSpp/2; s < initSpp; ++s)
            film.AddSample(&samples[p*initSpp*sampleSize + s*sampleSize], BUFFER_B);
    }

    DualSampler sampler(0, width, 0, height, spp, shutterOpen, shutterClose, threshold, nIterations, initSpp, &film);
    if(sampler.PixelsToSampleTotal() > 0)
    {
        layout.setElementIO(0, SampleLayout::INPUT);
        layout.setElementIO(1, SampleLayout::INPUT);
        client.setSampleLayout(layout);

        sampler.SetAdaptiveMode();
        int nIterations = sampler.GetIterationCount();
        int nPixelsPerIteration = Ceil2Int(float(sampler.PixelsToSampleTotal()) / nIterations);
        RNG rng;
        for(int i = 0; i < nIterations; ++i)
        {
            sampler.GetSamplingMaps(nPixelsPerIteration);

            // use two subsamplers, one for BUFFER_A and another for BUFFER_B
//            for(int j = 0; j < 2; ++j)
//            {
//                std::unique_ptr<Sampler> subSampler(sampler.GetSubSampler(j, 2));

//                int nSamples = 0;
//                while(nSamples = subSampler->GetMoreSamples(samples, rng))
//                {
//                    client.evaluateSamples(BenchmarkClient::SAMPLES, nSamples);
//                    for(int i = 0; i < nSamples; ++i)
//                        film.AddSample(&samples[i*sampleSize], TargetBuffer(j));
//                };
//            }

            size_t nSamples[2] = {0, 0};
            size_t nSum = 0;
            for(int j = 0; j < 2; ++j)
            {
                std::unique_ptr<Sampler> subSampler(sampler.GetSubSampler(j, 2));
                int n = 0;
                while(n = subSampler->GetMoreSamples(&samples[nSum*sampleSize], rng))
                {
                    nSamples[j] += n;
                    nSum += n;
                }
            }
            client.evaluateSamples(nSum);
            for(size_t i = 0; i < nSamples[0]; ++i)
                film.AddSample(&samples[i*sampleSize], TargetBuffer(0));
            for(size_t i = 0; i < nSamples[1]; ++i)
                film.AddSample(&samples[(nSamples[0] + i)*sampleSize], TargetBuffer(1));
        }
        sampler.Finalize();
    }

    float* result = client.getResultBuffer();
    film.WriteImage(result);
    client.sendResult();
    return 0;
}
