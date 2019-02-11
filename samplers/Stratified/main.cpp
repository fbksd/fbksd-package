
#include "stratified.h"
#include <fbksd/client/BenchmarkClient.h>
#include <iostream>
#include <random>
using namespace fbksd;

//#define IMAGE_X  0
//#define IMAGE_Y  1
//#define LENS_U   2
//#define LENS_V   3
//#define TIME     4
//#define COLOR    5
//#define COLOR_R  5
//#define COLOR_G  6
//#define COLOR_B  7

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


class Independent
{
public:
    // min and max are inclusive
    Independent()
    {
        // Seed with a real random value, if available
        std::random_device rd;
        // Choose a random mean between 1 and 6
        m_re = std::mt19937(rd());
        m_dist = std::uniform_real_distribution<float>(0.f, 1.f);
    }

    float next1D()
    { return m_dist(m_re); }

    Point2 next2D()
    { return Point2(m_dist(m_re), m_dist(m_re)); }

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
    int spp = scene.get<int64_t>("max_spp");

    int stratifiedSpp = spp;
    int independentSpp = 0;
    bool perfectSquare = true;
    int i = 1;
    while (i * i < spp)
        ++i;
    if(spp != i*i)
    {
        stratifiedSpp = (i-1)*(i-1);
        independentSpp = spp - stratifiedSpp;
        perfectSquare = false;

        std::cout << "spp is not perfect square, falling back to independent" << std::endl;
        std::cout << "spp stratified = " << stratifiedSpp << std::endl;
        std::cout << "spp independent = " << independentSpp << std::endl;
    }

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

    StratifiedSampler sampler(stratifiedSpp, 7);
    sampler.setFilmResolution(Vector2i(w, h), false);
    Independent independentSampler;
    float* result = client.getResultBuffer();
    const float sppInv = 1.f / (float)spp;

    client.evaluateInputSamples(SPP(spp),
        [&](const BufferTile& tile)
        {
            for(int64_t y = tile.beginY(); y < tile.endY(); ++y)
            for(int64_t x = tile.beginX(); x < tile.endX(); ++x)
            {
                sampler.generate(Point2i(x, y));
                for(int s = 0; s < stratifiedSpp; ++s)
                {
                    float* sample = tile(x, y, s);
                    Point2 p = sampler.next2D();
                    Point2 l = sampler.next2D();
                    float t = sampler.next1D();
                    Point2 light = sampler.next2D();
                    sample[IMAGE_X] = p.x + x;
                    sample[IMAGE_Y] = p.y + y;
                    sample[LENS_U] = l.x;
                    sample[LENS_V] = l.y;
                    sample[TIME] = t;
                    sample[LIGHT_X] = light.x;
                    sample[LIGHT_Y] = light.y;
                    sampler.advance();
                }
                for(int s = stratifiedSpp; s < spp; ++s)
                {
                    float* sample = tile(x, y, s);
                    Point2 p = independentSampler.next2D();
                    Point2 l = independentSampler.next2D();
                    float t = independentSampler.next1D();
                    Point2 light = independentSampler.next2D();
                    sample[IMAGE_X] = p.x + x;
                    sample[IMAGE_Y] = p.y + y;
                    sample[LENS_U] = l.x;
                    sample[LENS_V] = l.y;
                    sample[TIME] = t;
                    sample[LIGHT_X] = light.x;
                    sample[LIGHT_Y] = light.y;
                }
            }
        },
        [&](const BufferTile& tile)
        {
            for(int64_t y = tile.beginY(); y < tile.endY(); ++y)
            for(int64_t x = tile.beginX(); x < tile.endX(); ++x)
            {
                float* pixel = &result[y * w * 3 + x * 3];
                for(int s = 0; s < spp; ++s)
                {
                    float *sample = tile(x, y, s);
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

