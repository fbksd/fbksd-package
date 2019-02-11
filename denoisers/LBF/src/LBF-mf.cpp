#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;
#include <memory>
#include "SampleWriter/SampleWriter.h"


int main(int argc, char* argv[])
{
    BenchmarkClient client(argc, argv);
    SceneInfo sceneInfo = client.getSceneInfo();
    auto width = sceneInfo.get<int64_t>("width");
    auto height = sceneInfo.get<int64_t>("height");
    auto spp = sceneInfo.get<int64_t>("max_spp");

    SampleLayout layout;
    layout("IMAGE_X")("IMAGE_Y")
          ("COLOR_R")("COLOR_G")("COLOR_B")
          ("WORLD_X_NS")("WORLD_Y_NS")("WORLD_Z_NS")
          ("NORMAL_X_NS")("NORMAL_Y_NS")("NORMAL_Z_NS")
          ("TEXTURE_COLOR_R_NS")("TEXTURE_COLOR_G_NS")("TEXTURE_COLOR_B_NS")
          ("TEXTURE_COLOR_R")[1]("TEXTURE_COLOR_G")[1]("TEXTURE_COLOR_B")[1]
          ("DIRECT_LIGHT_R")("DIRECT_LIGHT_G")("DIRECT_LIGHT_B");
    client.setSampleLayout(layout);
    SampleWriter::Initialize(width, height, spp);

    client.evaluateSamples(SPP(spp), [&](const BufferTile& tile)
    {
        for(auto y = tile.beginY(); y < tile.endY(); ++y)
        for(auto x = tile.beginX(); x < tile.endX(); ++x)
        for(size_t s = 0; s < spp; ++s)
        {
            float* sample = tile(x, y, s);
            float x = sample[X_COORD];
            float y = sample[Y_COORD];

            SampleWriter::SetPosition(x, y, s, sample[X_COORD], X_COORD_OFFSET);
            SampleWriter::SetPosition(x, y, s, sample[Y_COORD], Y_COORD_OFFSET);
            SampleWriter::SetColor(x, y, s, sample[COLOR_1], COLOR_1_OFFSET);
            SampleWriter::SetColor(x, y, s, sample[COLOR_2], COLOR_2_OFFSET);
            SampleWriter::SetColor(x, y, s, sample[COLOR_3], COLOR_3_OFFSET);

            SampleWriter::SetFeature(x, y, s, sample[WORLD_1_X], WORLD_1_X_OFFSET);
            SampleWriter::SetFeature(x, y, s, sample[WORLD_1_Y], WORLD_1_Y_OFFSET);
            SampleWriter::SetFeature(x, y, s, sample[WORLD_1_Z], WORLD_1_Z_OFFSET);

            SampleWriter::SetFeature(x, y, s, sample[NORM_1_X], NORM_1_X_OFFSET);
            SampleWriter::SetFeature(x, y, s, sample[NORM_1_Y], NORM_1_Y_OFFSET);
            SampleWriter::SetFeature(x, y, s, sample[NORM_1_Z], NORM_1_Z_OFFSET);

            SampleWriter::SetFeature(x, y, s, sample[TEXTURE_1_X], TEXTURE_1_X_OFFSET);
            SampleWriter::SetFeature(x, y, s, sample[TEXTURE_1_Y], TEXTURE_1_Y_OFFSET);
            SampleWriter::SetFeature(x, y, s, sample[TEXTURE_1_Z], TEXTURE_1_Z_OFFSET);

            SampleWriter::SetFeature(x, y, s, sample[TEXTURE_2_X], TEXTURE_2_X_OFFSET);
            SampleWriter::SetFeature(x, y, s, sample[TEXTURE_2_Y], TEXTURE_2_Y_OFFSET);
            SampleWriter::SetFeature(x, y, s, sample[TEXTURE_2_Z], TEXTURE_2_Z_OFFSET);

            float visibility = sample[VISIBILITY_1] + sample[VISIBILITY_1 + 1] + sample[VISIBILITY_1 + 2] > 0;
            SampleWriter::SetFeature(x, y, s, visibility, VISIBILITY_1_OFFSET);
        }
    });

    float* result = client.getResultBuffer();
    SampleWriter::ProcessData(result);
    client.sendResult();
    return 0;
}
