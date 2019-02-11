#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;
#include <memory>
#include <iostream>
#include "box.h"
#include "multifilm.h"
#include "multisampler.h"


int main(int argc, char* argv[])
{
    BenchmarkClient client(argc, argv);
    SceneInfo sceneInfo = client.getSceneInfo();
    auto width = sceneInfo.get<int64_t>("width");
    auto height = sceneInfo.get<int64_t>("height");
    auto spp = sceneInfo.get<int64_t>("max_spp");
    float shutterOpen = sceneInfo.get<float>("shutter_open");
    float shutterClose = sceneInfo.get<float>("shutter_close");

    SampleLayout layout;
    layout("IMAGE_X")("IMAGE_Y")
          ("COLOR_R")("COLOR_G")("COLOR_B")
          ("DIRECT_LIGHT_R")("DIRECT_LIGHT_G")("DIRECT_LIGHT_B")
          ("WORLD_X_NS")("WORLD_Y_NS")("WORLD_Z_NS")
          ("NORMAL_X_NS")("NORMAL_Y_NS")("NORMAL_Z_NS")
          ("TEXTURE_COLOR_R_NS")("TEXTURE_COLOR_G_NS")("TEXTURE_COLOR_B_NS");
    int sampleSize = layout.getSampleSize();
    client.setSampleLayout(layout);

    // Parameters
    int wnd_rad = 20; // Filter width
    int nIterations = 0;
    bool finalize = true;
    bool useLdSamples = true;
    auto initSpp = std::max(spp/(1 + nIterations), INT64_C(1));
    float threshold = std::numeric_limits<float>::infinity();
    std::cout << "wnd_rad: " << wnd_rad << std::endl;
    std::cout << "nIterations: " << nIterations << std::endl;
    std::cout << "initSpp: " << initSpp << std::endl;

    BoxFilter filter(0.5f, 0.5f);
    MultiFilm film(width, height, &filter, wnd_rad);

    client.evaluateSamples(SPP(initSpp), [&](const BufferTile& tile)
    {
        for(size_t y = tile.beginY(); y < tile.endY(); ++y)
        for(size_t x = tile.beginX(); x < tile.endX(); ++x)
        {
            // give half the samples for each buffer
            for(size_t s = 0; s < initSpp/2; ++s)
                film.AddSample(tile(x, y, s), 0);
            for(size_t s = initSpp/2; s < initSpp; ++s)
                film.AddSample(tile(x, y, s), 1);
        }
    });

    MultiSampler sampler(0, width, 0, height, spp, shutterOpen, shutterClose, threshold, nIterations, initSpp, &film, finalize, useLdSamples);
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

            // use two subsamplers, one for each buffer
            std::vector<float> inSamples;
            for(int j = 0; j < 2; ++j)
            {
                std::unique_ptr<Sampler> subSampler(sampler.GetSubSampler(j, 2));
                int nSamples = 0;
                int n = 0;
                while(n = subSampler->GetMoreSamples(&inSamples, rng))
                    nSamples += n;

                float* currentInSample = inSamples.data();
                client.evaluateInputSamples(nSamples,
                    [&](size_t count, float* samples){
                        for(size_t i = 0; i < count; ++i)
                        {
                            samples[i*sampleSize + IMAGE_X] = currentInSample[0];
                            samples[i*sampleSize + IMAGE_Y] = currentInSample[1];
                            currentInSample += 2;
                        }
                    },
                    [&](size_t count, float* samples){
                        for(size_t i = 0; i < count; ++i)
                            film.AddSample(&samples[i*sampleSize], j);
                    }
                );

                inSamples.clear();
            }
        }
        sampler.Finalize();
    }

    float* result = client.getResultBuffer();
    film.WriteImage(result);
    client.sendResult();
    return 0;
}
