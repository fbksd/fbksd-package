
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
        if(std::isinf(sample[DEPTH]))
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
    BenchmarkClient client;

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
    float *samples = client.getSamplesBuffer();
    float maxDepth = 0.f;

    size_t totalSamplesUsed = 0;
    //===========================================================
    // Initial sampling
    //===========================================================
    {
        totalSamplesUsed = client.evaluateSamples(SPP(initSPP));
        size_t numSamples = numPixels * initSPP;
        maxDepth = computeMaxDepth(numSamples, samples);
        fixSamples(maxDepth, numSamples, samples);
        float* sample = samples;
        for(size_t i = 0; i < numSamples; ++i)
        {
            film.AddSample(sample);
            sample += SAMPLE_SIZE;
        }
    }

    //===========================================================
    // Adaptive sampling
    //===========================================================
    if(adaptiveIterations > 0 && adaptiveSPP > 0)
    {
        RNG rng;
        SBFSampler sampler(0, width, 0, height, adaptiveSPP, shutterOpen, shutterClose, initSPP, maxNSamples, adaptiveIterations);
        size_t numAdaptiveSamples = numPixels * adaptiveSPP;
        float adaptiveSamples[maxNSamples * SAMPLE_SIZE];
        for(int iter = 0; iter < adaptiveIterations; ++iter)
        {
            vector<vector<int> > pixOff;
            vector<vector<int> > pixSmp;
            film.GetAdaptPixels(sampler.GetAdaptiveSPP(), pixOff, pixSmp);
            sampler.SetPixelOffset(&pixOff);
            sampler.SetPixelSampleCount(&pixSmp);

            int sampleCount = 0;
            size_t countSum = 0;
            while((sampleCount = sampler.GetMoreSamples(&samples[countSum*SAMPLE_SIZE], rng)) > 0)
                countSum += sampleCount;

            if(countSum > numAdaptiveSamples)
            {
                std::cout << "Adaptive sampling exceeded budged: " << countSum << " > " << numAdaptiveSamples << std::endl;
                countSum = numAdaptiveSamples;
            }
            std::cout << "iteration " << iter+1 << " of " << adaptiveIterations << " (" << countSum << " samples)" << std::endl;
            layout.setElementIO("IMAGE_X", SampleLayout::INPUT);
            layout.setElementIO("IMAGE_Y", SampleLayout::INPUT);
            client.setSampleLayout(layout);
            totalSamplesUsed += client.evaluateSamples(countSum);
            fixSamples(maxDepth, countSum, samples);
            for(size_t s = 0; s < countSum; ++s)
                film.AddSample(&samples[s*SAMPLE_SIZE]);
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
