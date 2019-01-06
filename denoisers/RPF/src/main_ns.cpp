#include "Benchmark/BenchmarkClient/BenchmarkClient.h"
#include "SampleWriter/SampleWriter.h"
#include "Globals.h"
#include <memory>
#include <iomanip>
#include <limits>


int main()
{
    BenchmarkClient client;
    SceneInfo scene = client.getSceneInfo();
    int w = scene.get<int>("width");
    int h = scene.get<int>("height");
    int spp = scene.get<int>("max_spp");

    // RPF only works with spp >= 4 for now.
    // NOTE: Returning non-zero makes the benchmark system to not account this execution.
    if(spp < 4)
        return 1;

    SampleLayout layout;
    layout  ("IMAGE_X")
            ("IMAGE_Y")
            ("COLOR_R")
            ("COLOR_G")
            ("COLOR_B")
            ("WORLD_X_NS")
            ("WORLD_Y_NS")
            ("WORLD_Z_NS")
            ("NORMAL_X_NS")
            ("NORMAL_Y_NS")
            ("NORMAL_Z_NS")
            ("TEXTURE_COLOR_R_NS")
            ("TEXTURE_COLOR_G_NS")
            ("TEXTURE_COLOR_B_NS")
            ("WORLD_X")[1]
            ("WORLD_Y")[1]
            ("WORLD_Z")[1]
            ("NORMAL_X")[1]
            ("NORMAL_Y")[1]
            ("NORMAL_Z")[1]
            ("TEXTURE_COLOR_R")[1]
            ("TEXTURE_COLOR_G")[1]
            ("TEXTURE_COLOR_B")[1]
            ("LENS_U")
            ("LENS_V")
            ("TIME")
            ("LIGHT_X")
            ("LIGHT_Y");
    client.setSampleLayout(layout);

    float* samples = client.getSamplesBuffer();
    client.evaluateSamples(BenchmarkClient::SAMPLES_PER_PIXEL, spp);

    float* result = client.getResultBuffer();
    RPF(result, samples, w, h, spp, SAMPLE_LENGTH, NUM_OF_POSITIONS, NUM_OF_COLORS, NUM_OF_FEATURES, NUM_OF_RANDOM_PARAMS, NULL);
    client.sendResult();

    return 0;
}
