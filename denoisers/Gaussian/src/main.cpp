#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;
#include <memory>
#include <cmath>


//TODO: improve performance (tabulate gaussian values and parallelize).


// Filter parameters
static float xWidth = 2.f;
static float yWidth = 2.f;
static float alpha = 2.f;
static float expX = std::exp(-alpha * xWidth * xWidth);
static float expY = std::exp(-alpha * yWidth * yWidth);

inline float gaussian(float x, float y)
{
    return std::max(0.f, std::exp(-alpha * x * x) - expX) *
           std::max(0.f, std::exp(-alpha * y * y) - expY);
}


int main()
{

    BenchmarkClient client;
    SceneInfo scene = client.getSceneInfo();
    auto width = scene.get<int64_t>("width");
    auto height = scene.get<int64_t>("height");
    auto numSamples = scene.get<int64_t>("max_samples");
    auto spp = scene.get<int64_t>("max_spp");

    SampleLayout layout;
    layout("IMAGE_X")("IMAGE_Y")("COLOR_R")("COLOR_G")("COLOR_B");
    client.setSampleLayout(layout);

    float* samples = client.getSamplesBuffer();
    client.evaluateSamples(SPP(spp));

    size_t numPixels = width * height;
    std::vector<float> weightSum(numPixels, 0.f);
    float* result = client.getResultBuffer();

    float* sample = samples;
    for(size_t i = 0; i < numSamples; ++i)
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
            float w = gaussian((x + 0.5f) - sample[0], (y + 0.5f) - sample[1]);
            if(w > 0.f)
            {
                pixel[0] += sample[2] * w;
                pixel[1] += sample[3] * w;
                pixel[2] += sample[4] * w;
                weightSum[y*width + x]  += w;
            }
        }

        sample += layout.getSampleSize();
    }

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
