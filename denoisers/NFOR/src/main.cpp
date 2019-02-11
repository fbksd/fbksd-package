#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;
#include <random>
#include <algorithm>
#include "NFOR.h"


/*
 * NOTES
 *
 * Features used: pixel coordinates (2D), normal (3D), albedo (3D), depth (1D), visibility (1D).
 *  - it also can use a secondary albedo buffer (second ray intersection)
 *
 * The visibility feature is defined as: the fraction of shadow rays that hit any light source over
 * all direct shadow rays evaluated in a pixel.
 *
 * Main Steps (as described in the paper):
 *  1. auxiliary buffers prefiltering;
 *      - prefilter all auxiliary buffers using NL-Means (as in RDFC)
 *      - use the squared difference of the two filtered half buffers as an estimate of their
 *        residual variance
 *      - use the estimated residual variance in a second filtering pass
 *
 *  2. collaborative first-order regression;
 *      - Feature cross-filtering
 *          * filter the two half buffers separately
 *          * use the feature vector of the first buffer when fitting the second, and vice-versa
 *      - First-order regression
 *          * compute the regression weiths (Eq. 12) only on the pixel color
 *      - Collaborative filtering
 *          * in first-order regression, the entire filter window is denoised at once (not only the center pixel)
 *          * compute the filtered output as a weighted average of denoised filter windows, each
 *            weighted by its regression kernel
 *
 *  3. bandwidth selection;
 *      - perform selection-based bandwidth estimation k = {0.5, 1.0}
 *      - estimate each candidate bandwidth MSE
 *          * compute the squared difference of the filtered first half buffer to the second input half buffer,
 *            and vice-versa, and subtract the variance of the input. This gives an estimator of the MSE of each
 *            half buffer
 *          * compute the MSE estimate of the average of the two filtered half buffers (Sec. 5.3)
 *          * filter the estimated MSE and use it to generate selection maps for each filter bandwidth (similar to RDFC)
 *
 *  4. second filtering pass;
 *      - perform a second filtering pass on the color buffer (as in RDFC)
 *      - use the variance of the first pass output to compute the regression weights, which is estimated as the
 *        variance across the filtered half buffers
 *      - do NOT perform cross-filtering on this second pass, just refilter the average of the filtered half buffers
 *
 *  Miscellaneous details:
 *      - locally normalize the auxiliary features to occupy the same range (similar to WLR)
 *          * offset and scale each feature within the filtering window to the [-1, 1] interval
 *      - space-time filtering
 *          * expand the regression window to the spatio-temporal domain by including noisy pixels and features
 *            from the two previous and two next frames (similar to WLR)
 *          * include the frame index as an additional auxiliary vector, and compute the regression weights
 *            using a spatio-temporal NL-Means kernel (BCM07, RKZ12).
 *      - the albedo feature is from the first non-specular surface along a path
 *      - all scenes except San Miguel use independent samples. Sam Miguel uses low discrepancy
 *          * for low discrepancy, the variance across the pair of half buffers is used as an estimate
 *            of the sample mean variance.
 *
*/

static void shuffleVector(std::vector<size_t>& vec)
{
    static std::random_device rd;
    static std::mt19937 re(rd());
    std::shuffle(vec.begin(), vec.end(), re);
}


int main(int argc, char* argv[])
{
    BenchmarkClient client(argc, argv);
    SceneInfo scene = client.getSceneInfo();
    const auto w = scene.get<int64_t>("width");
    const auto h = scene.get<int64_t>("height");
    const auto spp = scene.get<int64_t>("max_spp");

    SampleLayout layout;
    layout  ("IMAGE_X")
            ("IMAGE_Y")
            ("COLOR_R")
            ("COLOR_G")
            ("COLOR_B")
            ("NORMAL_X")
            ("NORMAL_Y")
            ("NORMAL_Z")
            ("TEXTURE_COLOR_R_NS")
            ("TEXTURE_COLOR_G_NS")
            ("TEXTURE_COLOR_B_NS")
            ("DEPTH")
            ("DIRECT_LIGHT_R");
    client.setSampleLayout(layout);

    const size_t numPixels = w*h;
    size_t sampleSize = layout.getSampleSize();
    MultiFilm film(w, h, 20);

    //// Split the samples in half
    // sequence used to scramble the samples in each pixel to avoid bias if the samples have
    // a predefined order.
    std::vector<size_t> indexSequence(spp);
    std::iota(indexSequence.begin(), indexSequence.end(), 0);
    client.evaluateSamples(SPP(spp), [&](const BufferTile& tile)
    {
        for(size_t y = tile.beginY(); y < tile.endY(); ++y)
        for(size_t x = tile.beginX(); x < tile.endX(); ++x)
        {
            shuffleVector(indexSequence);
            for(size_t s = 0; s < spp/2; ++s)
                film.AddSample(tile(x, y, indexSequence[s]), 0);
            for(size_t s = spp/2; s < spp; ++s)
                film.AddSample(tile(x, y, indexSequence[s]), 1);
        }
    });

    float* result = client.getResultBuffer();
    NFOR(&film, result);
//    testFilter(&film, result);
    client.sendResult();
    return 0;
}

