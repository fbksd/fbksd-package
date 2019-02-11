
#include <fbksd/client/BenchmarkClient.h>
#include <iostream>
#include <random>
#include <algorithm>
using namespace fbksd;

#define IMAGE_X  0
#define IMAGE_Y  1
#define LENS_U   2
#define LENS_V   3
#define TIME     4
#define LIGHT_X  5
#define LIGHT_Y  6
#define COLOR    7
#define COLOR_R  7
#define COLOR_G  8
#define COLOR_B  9


class Sampler
{
public:
    // min and max are inclusive
    Sampler()
    {
        // Seed with a real random value, if available
        std::random_device rd;
        // Choose a random mean between 1 and 6
        m_re = std::mt19937(rd());
        m_dist = std::uniform_real_distribution<float>(0.f, 1.f);
    }

    float nextFloat()
    { return m_dist(m_re); }

private:
    std::mt19937 m_re;
    std::uniform_real_distribution<float> m_dist;
};

int main(int argc, char* argv[])
{
    BenchmarkClient client(argc, argv);
    SceneInfo scene = client.getSceneInfo();
    const int w = scene.get<int64_t>("width");
    const int h = scene.get<int64_t>("height");
    const int spp = scene.get<int64_t>("max_spp");

    SampleLayout layout;
    layout("IMAGE_X", SampleLayout::INPUT)
          ("IMAGE_Y", SampleLayout::INPUT)
          ("LENS_U", SampleLayout::INPUT)
          ("LENS_V", SampleLayout::INPUT)
          ("TIME", SampleLayout::INPUT)
          ("LIGHT_X", SampleLayout::INPUT)
          ("LIGHT_Y", SampleLayout::INPUT)
          ("COLOR_R")("COLOR_G")("COLOR_B");
    client.setSampleLayout(layout);
    int sampleSize = layout.getSampleSize();

    Sampler sampler;
    float* result = client.getResultBuffer();
    const float sppInv = 1.f / (float)spp;

    client.evaluateInputSamples(SPP(spp),
        [&](const BufferTile& tile)
        {
            for(int64_t y = tile.beginY(); y < tile.endY(); ++y)
            for(int64_t x = tile.beginX(); x < tile.endX(); ++x)
            for(int s = 0; s < spp; ++s)
            {
                float* sample = tile(x, y, s);
                sample[IMAGE_X] = sampler.nextFloat() + x;
                sample[IMAGE_Y] = sampler.nextFloat() + y;
                sample[LENS_U] = sampler.nextFloat();
                sample[LENS_V] = sampler.nextFloat();
                sample[TIME] = sampler.nextFloat();
                sample[LIGHT_X] = sampler.nextFloat();
                sample[LIGHT_Y] = sampler.nextFloat();
            }
        },
        [&](const BufferTile& tile)
        {
            for(int64_t y = tile.beginY(); y < tile.endY(); ++y)
            for(int64_t x = tile.beginX(); x < tile.endX(); ++x)
            {
                float* pixel = &result[y*w*3 + x*3];
                for(int s = 0; s < spp; ++s)
                {
                    float* sample = tile(x, y, s);
                    pixel[0] += sample[COLOR_R];
                    pixel[1] += sample[COLOR_G];
                    pixel[2] += sample[COLOR_B];
                }

                pixel[0] *= sppInv;
                pixel[1] *= sppInv;
                pixel[2] *= sppInv;
            }
        }
    );

    client.sendResult();
    return 0;
}

