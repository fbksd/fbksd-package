#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;
#include "SampleWriter/SampleWriter.h"
#include "Globals.h"
#include <memory>
#include <iomanip>
#include <limits>


int main()
{
    BenchmarkClient client;
    SceneInfo scene = client.getSceneInfo();
    auto w = scene.get<int64_t>("width");
    auto h = scene.get<int64_t>("height");
    auto spp = scene.get<int64_t>("max_spp");

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
            ("WORLD_X")
            ("WORLD_Y")
            ("WORLD_Z")
            ("NORMAL_X")
            ("NORMAL_Y")
            ("NORMAL_Z")
            ("TEXTURE_COLOR_R")
            ("TEXTURE_COLOR_G")
            ("TEXTURE_COLOR_B")
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
    client.evaluateSamples(SPP(spp));

    float* result = client.getResultBuffer();
    RPF(result, samples, w, h, spp, SAMPLE_LENGTH, NUM_OF_POSITIONS, NUM_OF_COLORS, NUM_OF_FEATURES, NUM_OF_RANDOM_PARAMS, NULL);
    client.sendResult();

    return 0;
}
