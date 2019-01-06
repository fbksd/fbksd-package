/*----------------------------------------------------------------------------
 RHF - Ray Histogram Fusion
 ----------------------------------------------------------------------------*/

#include <fbksd/client/BenchmarkClient.h>
using namespace fbksd;
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <memory>
#include "libdenoising.h"
#include "imageHisto.h"
#include "io_exr.h"

#define BENCHMARK_MODE


struct program_argums
{
    program_argums():
        max_distance(0.8f),
        knn(2),
        win(1),
        bloc(6),
        nscales(2)
    {}

    float max_distance;    // Max-distance between patchs
    int knn;               // Minimum number of similar patchs (default: 2)
    int win;               // Half the windows size (default: 1)
    int bloc;              // Half the block size [1, 6] (default: 6)
    int nscales;           // Number of Scales - Multi-Scale (default: 2)
};

void reconstructInput(float* samples, size_t numSamples, int sampleSize, ImageFilmHisto* film)
{
    for(size_t i = 0; i < numSamples; ++i)
        film->AddSample(&samples[i*sampleSize]);
}

int main(int argc, char **argv)
{
#ifdef BENCHMARK_MODE
    BenchmarkClient client;
    SceneInfo sceneInfo = client.getSceneInfo();
    auto nx = sceneInfo.get<int64_t>("width");
    auto ny = sceneInfo.get<int64_t>("height");
    int64_t nc = 3;
    auto spp = sceneInfo.get<int64_t>("max_spp");

    SampleLayout layout;
    layout("IMAGE_X")("IMAGE_Y")("COLOR_R")("COLOR_G")("COLOR_B");
    client.setSampleLayout(layout);
#else
    std::string inputArg(argv[1]);

    int nx = 0, ny = 0, nc = 3;
    float *alpha = NULL;
    Imath::Box2i dataWindow, displayWindow;
    std::unique_ptr<float[]> d_v(ReadImageEXR(inputArg.c_str(), dataWindow, displayWindow, alpha));
    nx = dataWindow.max.x - dataWindow.min.x + 1;
    ny = dataWindow.max.y - dataWindow.min.y + 1;
#endif

    // Default parameters
    program_argums param;
    int    iBins   = 20;
    float  fgamma  = 2.2;
    float  fMval = 2.5;
    float  fsval = 2.0;
#ifdef BENCHMARK_MODE
    MitchellFilter filter(1.f/3.f, 1.f/3.f, 2.f, 2.f);
    ImageFilmHisto imgHist(nx, ny, &filter, iBins, fgamma, fMval, fsval);
#endif

    // variables
    int d_w = (int) nx;
    int d_h = (int) ny;
    int d_c = (int) nc;
    int d_wh = d_w * d_h;

    // denoise
    std::unique_ptr<float*[]> fpO(new float*[d_c]);
    std::unique_ptr<float*[]> fpI(new float*[d_c]);
    std::unique_ptr<float[]> denoised(new float[nx * ny * nc]);

#ifdef BENCHMARK_MODE
    client.evaluateSamples(SPP(spp));
    float* samples = client.getSamplesBuffer();
    reconstructInput(samples, nx*ny*spp, layout.getSampleSize(), &imgHist);
    auto input = imgHist.WriteImage();

    auto& img = input.first;
    std::unique_ptr<float[]> d_v(new float[nx * ny * nc]);
    for(int c = 0; c < nc; ++c)
    for(int y = 0; y < ny; ++y)
    for(int x = 0; x < nx; ++x)
        d_v[x + y*nx + c*nx*ny] = img[x*nc + y*nx*nc + c];
#endif

    for (int ii=0; ii < d_c; ii++) {
        fpI[ii] = &d_v[ii * d_wh];
        fpO[ii] = &denoised[ii * d_wh];
    }

    //-Read Histogram image----------------------------------------------------
#ifdef BENCHMARK_MODE
    int nx_h = nx, ny_h = ny, nc_h = 3*iBins + 1;
    std::unique_ptr<float[]> fpH(new float[nx_h * ny_h * nc_h]);
    auto& hist = input.second;
    memcpy(fpH.get(), hist.get(), nx_h * ny_h * nc_h * sizeof(float));
#else
    std::string histArg(argv[2]);
    int nx_h = 0, ny_h = 0, nc_h = 0;
    std::unique_ptr<float[]> fpH(readMultiImageEXR(histArg.c_str(), &nx_h, &ny_h, &nc_h));
#endif

    std::unique_ptr<float*[]> fpHisto(new float*[nc_h]);
    for (int ii=0; ii < nc_h; ii++)
        fpHisto[ii] = &fpH[ii * nx_h*ny_h];

    rhf_multiscale(param.win,
                   param.bloc,
                   param.max_distance,
                   param.knn,
                   param.nscales,
                   fpHisto.get(),
                   fpI.get(), fpO.get(), nullptr, d_c, d_w, d_h, nc_h);

#ifdef BENCHMARK_MODE
    float* result = client.getResultBuffer();
    for(int c = 0; c < nc; ++c)
    for(int y = 0; y < ny; ++y)
    for(int x = 0; x < nx; ++x)
        result[x*nc + y*nx*nc + c] = denoised[x + y*nx + c*nx*ny];
    client.sendResult();
#else
    const int channelStride = d_c == 1 ? 0 : d_w*d_h;
    WriteImageEXR("img_filtered.exr", denoised.get(), alpha, dataWindow, displayWindow, channelStride);
#endif

    return 0;
}
