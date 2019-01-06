#include "fbksd/client/BenchmarkClient.h"
using namespace fbksd;

int main(int argc, char* argv[])
{
    BenchmarkClient client(argc, argv);
    SceneInfo scene = client.getSceneInfo();
    const auto w = scene.get<int64_t>("width");
    const auto h = scene.get<int64_t>("height");
    const auto spp = scene.get<int64_t>("max_spp");

    SampleLayout layout;
    layout("COLOR_R")("COLOR_G")("COLOR_B");
    client.setSampleLayout(layout);

    float* samples = client.getSamplesBuffer();
    client.evaluateSamples(SPP(spp));
    
    float* result = client.getResultBuffer();
    const float sppInv = 1.f / (float)spp;
    for(int64_t y = 0; y < h; ++y)
    for(int64_t x = 0; x < w; ++x)
    {
        float* pixel = &result[y*w*3 + x*3];
        float* pixelSamples = &samples[y*w*spp*3 + x*3*spp];
        for(int s = 0; s < spp; ++s)
        {
            float* sample = &pixelSamples[s*3];
            pixel[0] += sample[0];
            pixel[1] += sample[1];
            pixel[2] += sample[2];
        }

        pixel[0] *= sppInv;
        pixel[1] *= sppInv;
        pixel[2] *= sppInv;
    }

    client.sendResult();
    return 0;
}

