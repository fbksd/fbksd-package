#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;
#include <memory>
#include <cmath>


//TODO:
//  - improve performance (tabulate filter values and parallelize);
//  - adjust filter parameters based on scene info.
namespace
{
    // Filter parameters
    float xWidth = 2.f;
    float yWidth = 2.f;
    float B = 2.f;
    float C = 2.f;
    float invXWidth = 1.f/xWidth;
    float invYWidth = 1.f/yWidth;

    float mitchell(float x)
    {
        x = std::abs(2.f * x);
        if (x > 1.f)
            return ((-B - 6*C) * x*x*x + (6*B + 30*C) * x*x +
                    (-12*B - 48*C) * x + (8*B + 24*C)) * (1.f/6.f);
        else
            return ((12 - 9*B - 6*C) * x*x*x +
                    (-18 + 12*B + 6*C) * x*x +
                    (6 - 2*B)) * (1.f/6.f);
    }

    inline float mitchell(float x, float y)
    {
        return mitchell(x * invXWidth) * mitchell(y * invYWidth);
    }
}

int main(int argc, char* argv[])
{
    BenchmarkClient client(argc, argv);
    SceneInfo scene = client.getSceneInfo();
    auto width = scene.get<int64_t>("width");
    auto height = scene.get<int64_t>("height");
    auto numSamples = scene.get<int64_t>("max_samples");
    auto spp = scene.get<int64_t>("max_spp");

    SampleLayout layout;
    layout("IMAGE_X")("IMAGE_Y")("COLOR_R")("COLOR_G")("COLOR_B");
    client.setSampleLayout(layout);


    size_t numPixels = width * height;
    std::vector<float> weightSum(numPixels, 0.f);
    float* result = client.getResultBuffer();

    client.evaluateSamples(SPP(spp), [&](const BufferTile& tile)
    {
        for(const auto sample: tile)
        {
            float fminX = sample[0] - xWidth;
            float fminY = sample[1] - yWidth;
            float fmaxX = sample[0] + xWidth;
            float fmaxY = sample[1] + yWidth;

            int64_t minX = floor(fminX + 0.5f);
            int64_t minY = floor(fminY + 0.5f);
            int64_t maxX = floor(fmaxX - 0.5f);
            int64_t maxY = floor(fmaxY - 0.5f);

            minX = std::max(INT64_C(0), minX);
            minY = std::max(INT64_C(0), minY);
            maxX = std::min(width-1, maxX);
            maxY = std::min(height-1, maxY);

            for(int64_t y = minY; y <= maxY; ++y)
            for(int64_t x = minX; x <= maxX; ++x)
            {
                float *pixel = &result[y*width*3 + x*3];
                float w = mitchell((x + 0.5f) - sample[0], (y + 0.5f) - sample[1]);
                if(w > 0.f)
                {
                    pixel[0] += sample[2] * w;
                    pixel[1] += sample[3] * w;
                    pixel[2] += sample[4] * w;
                    weightSum[y*width + x]  += w;
                }
            }
        }
    });

    for(int64_t y = 0; y < height; ++y)
    for(int64_t x = 0; x < width; ++x)
    {
        float *pixel = &result[y*width*3 + x*3];
        float ws = weightSum[y*width + x];
        if(ws > 0.f)
        {
            pixel[0] /= ws;
            pixel[1] /= ws;
            pixel[2] /= ws;
        }
    }

    client.sendResult();
    return 0;
}
