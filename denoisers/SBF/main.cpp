
#include "sbf.h"
#include "sbfsampler.h"
#include "sbfimage.h"

#include "filter.h"
#include "filters/gaussian.h"
#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;


float computeMaxDepth(size_t numSamples, float* samples)
{
    float maxDepth = 0.f;
    float* sample = samples;
    for(size_t i = 0; i < numSamples; ++i)
    {
        if((!std::isinf(sample[DEPTH])) && sample[DEPTH] > maxDepth)
            maxDepth = sample[DEPTH];
        sample += SAMPLE_SIZE;
    }
    std::cout << "Max depth = " << maxDepth << std::endl;
    return maxDepth;
}


// Fix samples with inf depth values;
// Also normalize depth values.
void fixSamples(float maxDepth, size_t numSamples, float* samples)
{
    float* sample = samples;
    for(size_t i = 0; i < numSamples; ++i)
    {
        if(std::isinf(sample[DEPTH]) || std::abs(maxDepth) < 1.e-10f)
            sample[DEPTH] = 1.f;
        else
            sample[DEPTH] /= maxDepth;

        sample += SAMPLE_SIZE;
    }
}


int main(int argc, char *argv[])
{
    //===========================================================
    // Initialization
    //===========================================================
    BenchmarkClient client(argc, argv);

    // Scene info
    SceneInfo scene = client.getSceneInfo();
    auto width = scene.get<int64_t>("width");
    auto height = scene.get<int64_t>("height");
    auto spp = scene.get<int64_t>("max_spp");
    auto shutterOpen = scene.get<float>("shutter_open");
    auto shutterClose = scene.get<float>("shutter_close");
    auto numPixels = width * height;

    // Sample layout
    SampleLayout layout;
    layout  ("IMAGE_X")
            ("IMAGE_Y")
            ("COLOR_R")
            ("COLOR_G")
            ("COLOR_B")
            ("NORMAL_X")
            ("NORMAL_Y")
            ("NORMAL_Z")
            ("TEXTURE_COLOR_R")
            ("TEXTURE_COLOR_G")
            ("TEXTURE_COLOR_B")
            ("DEPTH");
    client.setSampleLayout(layout);
    std::cout << "Correct sample size? " << (layout.getSampleSize() == SAMPLE_SIZE) << std::endl;

    // Sample budget management
    int adaptiveIterations = 1; // 0 turns off adaptive sampling (supports only 0 or 1 for now)
    int initSPP = adaptiveIterations > 0 ? std::min(INT64_C(8), spp) : spp;
    int adaptiveSPP = spp - initSPP;
    const int maxNSamples = 1024;
    std::cout << "adaptive iterations: " << adaptiveIterations << std::endl;
    std::cout << "initial spp: " << initSPP << std::endl;
    std::cout << "adaptive spp: " << adaptiveSPP << std::endl;

    // Filtering parameters
    //TODO: tune filter parameters based on scene info
    SBF::FilterType filterType = SBF::CROSS_NLM_FILTER;
    // Filtering parameters for intermediate adaptive sampling stage
    vector<float> interParams = {0.0f, 0.005f, 0.01f, 0.02f, 0.04f, 0.08f, 0.16f, 0.32f, 0.64f};
    // Filtering parameters for final output stage
    vector<float> finalParams = {0.0f, 0.005f, 0.01f, 0.02f, 0.04f, 0.08f, 0.16f, 0.32f, 0.64f};
    // Filtering parameters for feature buffers
    float sigman = 0.8f;
    float sigmar = 0.25f;
    float sigmad = 0.6f;
    // Filtering parameters for MSE filtering
    float intermsesigma = 4.0f;
    float finalmsesigma = 4.0f;
    // Filter parameters
    float xwidth = 2.f;
    float ywidth = 2.f;
    float alpha = 0.5f;

    GaussianFilter filter(xwidth, ywidth, alpha);
    SBFImageFilm film(width, height, &filter, false, filterType, interParams, finalParams, sigman, sigmar, sigmad, intermsesigma, finalmsesigma);
    float maxDepth = 0.f; // doesn't include inf

    size_t totalSamplesUsed = 0;
    //===========================================================
    // Initial sampling
    //===========================================================
    {
        if(initSPP > 0)
        {
            // use 1 spp to estimate maxDepth
            std::vector<float> samples(SAMPLE_SIZE * numPixels);
            client.evaluateSamples(SPP(1), [&](const BufferTile& tile) {
                for(size_t y = tile.beginY(); y < tile.endY(); ++y)
                for(size_t x = tile.beginX(); x < tile.endX(); ++x)
                {
                    float* sample = tile(x, y, 0);
                    if((!std::isinf(sample[DEPTH])) && sample[DEPTH] > maxDepth)
                        maxDepth = sample[DEPTH];
                    memcpy(&samples[(y*width + x)*SAMPLE_SIZE], sample, SAMPLE_SIZE*sizeof(float));
                }
            });

            fixSamples(maxDepth, numPixels, samples.data());
            for(size_t i = 0; i < numPixels; ++i)
                film.AddSample(&samples[i*SAMPLE_SIZE]);
        }

        if(initSPP > 1)
        {
            int n = initSPP - 1; // compute initSPP - 1 here because we already added the 1 spp used to estimate maxDepth.
            client.evaluateSamples(SPP(n), [&](const BufferTile& tile) {
                for(size_t y = tile.beginY(); y < tile.endY(); ++y)
                for(size_t x = tile.beginX(); x < tile.endX(); ++x)
                for(size_t s = 0; s < n; ++s)
                {
                    float* sample = tile(x, y, s);
                    fixSamples(maxDepth, 1, sample);
                    film.AddSample(sample);
                }
            });
        }

        totalSamplesUsed = numPixels * initSPP;
    }

    //===========================================================
    // Adaptive sampling
    //===========================================================
    if(adaptiveIterations > 0 && adaptiveSPP > 0)
    {
        layout.setElementIO("IMAGE_X", SampleLayout::INPUT);
        layout.setElementIO("IMAGE_Y", SampleLayout::INPUT);
        client.setSampleLayout(layout);

        RNG rng;
        SBFSampler sampler(0, width, 0, height, adaptiveSPP, shutterOpen, shutterClose, initSPP, maxNSamples, adaptiveIterations);
        size_t numAdaptiveSamples = numPixels * adaptiveSPP;
        std::vector<float> inSamples;
        for(int iter = 0; iter < adaptiveIterations; ++iter)
        {
            vector<vector<int> > pixOff;
            vector<vector<int> > pixSmp;
            film.GetAdaptPixels(sampler.GetAdaptiveSPP(), pixOff, pixSmp);
            sampler.SetPixelOffset(&pixOff);
            sampler.SetPixelSampleCount(&pixSmp);

            int sampleCount = 0;
            size_t countSum = 0;
            while((sampleCount = sampler.GetMoreSamples(&inSamples, rng)) > 0)
                countSum += sampleCount;

            if(countSum > numAdaptiveSamples)
            {
                std::cout << "Adaptive sampling exceeded budged: " << countSum << " > " << numAdaptiveSamples << std::endl;
                countSum = numAdaptiveSamples;
            }
            std::cout << "iteration " << iter+1 << " of " << adaptiveIterations << " (" << countSum << " samples)" << std::endl;
            totalSamplesUsed += countSum;

            float* currentInSample = inSamples.data();
            client.evaluateInputSamples(countSum,
                [&](size_t count, float* samples){
                    for(size_t i = 0; i < count; ++i)
                    {
                        float* sample = &samples[i*SAMPLE_SIZE];
                        sample[IMAGE_X] = currentInSample[0];
                        sample[IMAGE_Y] = currentInSample[1];
                        currentInSample += 2;
                    }
                },
                [&](size_t count, float* samples){
                    for(size_t i = 0; i < count; ++i)
                    {
                        float* sample = &samples[i*SAMPLE_SIZE];
                        fixSamples(maxDepth, 1, sample);
                        film.AddSample(sample);
                    }
                }
            );

            inSamples.clear();
        }
    }

    //===========================================================
    // Reconstruction
    //===========================================================
    std::cout << "Total samples used = " << totalSamplesUsed << std::endl;
    float *result = client.getResultBuffer();
    film.WriteImage(result);
    client.sendResult();

    return 0;
}
