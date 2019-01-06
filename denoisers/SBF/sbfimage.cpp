
/*
    Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


// film/sbfimage.cpp*
#include "stdafx.h"
#include "sbfimage.h"
#include "parallel.h"

// SBFImageFilm Method Definitions
SBFImageFilm::SBFImageFilm(int xres, int yres, Filter *filt, bool dp, SBF::FilterType type,
                     const vector<float> &interParams, const vector<float> &finalParams,
                     float sigmaN, float sigmaR, float sigmaD,
                     float interMseSigma, float finalMseSigma)
    : Film(xres, yres) {
    filter = filt;
//    memcpy(cropWindow, crop, 4 * sizeof(float));
    cropWindow[0] = 0;
    cropWindow[1] = 1;
    cropWindow[2] = 0;
    cropWindow[3] = 1;
    // Compute film image extent
    xPixelStart = Ceil2Int(xResolution * cropWindow[0]);
    xPixelCount = max(1, Ceil2Int(xResolution * cropWindow[1]) - xPixelStart);
    yPixelStart = Ceil2Int(yResolution * cropWindow[2]);
    yPixelCount = max(1, Ceil2Int(yResolution * cropWindow[3]) - yPixelStart);   

    dump = dp;
    sbf = new SBF(xPixelStart, yPixelStart, xPixelCount, yPixelCount, 
                  filter, type, interParams, finalParams, 
                  sigmaN, sigmaR, sigmaD, interMseSigma, finalMseSigma);
}


void SBFImageFilm::AddSample(float* sample) {
    sbf->AddSample(sample);
}


void SBFImageFilm::Splat(const CameraSample &sample, const Spectrum &L) {
    // TODO: Implement splatting
    Warning("[SBFImageFilm] Splatting is currently not supported");
}


void SBFImageFilm::GetSampleExtent(int *xstart, int *xend,
                                int *ystart, int *yend) const {
    *xstart = Floor2Int(xPixelStart + 0.5f - filter->xWidth);
    *xend   = Ceil2Int(xPixelStart + 0.5f + xPixelCount  +
                        filter->xWidth);

    *ystart = Floor2Int(yPixelStart + 0.5f - filter->yWidth);
    *yend   = Ceil2Int(yPixelStart + 0.5f + yPixelCount +
                        filter->yWidth);
}


void SBFImageFilm::GetPixelExtent(int *xstart, int *xend,
                               int *ystart, int *yend) const {
    *xstart = xPixelStart;
    *xend   = xPixelStart + xPixelCount;
    *ystart = yPixelStart;
    *yend   = yPixelStart + yPixelCount;
}


void SBFImageFilm::WriteImage(float* img) {
    sbf->WriteImage(filename, xResolution, yResolution, dump, img);
}

void SBFImageFilm::GetAdaptPixels(float avgSpp, vector<vector<int> > &pixOff, vector<vector<int> > &pixSam) {
    sbf->GetAdaptPixels(avgSpp, pixOff, pixSam);
}

//SBFImageFilm *CreateSBFImageFilm(const ParamSet &params, Filter *filter) {
//    string filename = params.FindOneString("filename", PbrtOptions.imageFile);
//    if (filename == "")
//#ifdef PBRT_HAS_OPENEXR
//        filename = "pbrt.exr";
//#else
//        filename = "pbrt.tga";
//#endif

//    int xres = params.FindOneInt("xresolution", 640);
//    int yres = params.FindOneInt("yresolution", 480);
//    if (PbrtOptions.quickRender) xres = max(1, xres / 4);
//    if (PbrtOptions.quickRender) yres = max(1, yres / 4);
//    float crop[4] = { 0, 1, 0, 1 };
//    int cwi;
//    const float *cr = params.FindFloat("cropwindow", &cwi);
//    if (cr && cwi == 4) {
//        crop[0] = Clamp(min(cr[0], cr[1]), 0., 1.);
//        crop[1] = Clamp(max(cr[0], cr[1]), 0., 1.);
//        crop[2] = Clamp(min(cr[2], cr[3]), 0., 1.);
//        crop[3] = Clamp(max(cr[2], cr[3]), 0., 1.);
//    }
//    bool debug = params.FindOneBool("dumpfeaturebuffer", false);
//    string filterType = params.FindOneString("filter", "cbf");
//    SBF::FilterType type;
//    if(filterType == "cbf") {
//        type = SBF::CROSS_BILATERAL_FILTER;
//    } else if(filterType == "cnlmf") {
//        type = SBF::CROSS_NLM_FILTER;
//    } else {
//        Warning("[SBFFilm] Unsuporrted filter type, set to default.");
//        type = SBF::CROSS_BILATERAL_FILTER;
//    }

//    int nInterParams = 0;
//    const float *interParams = params.FindFloat("interparams", &nInterParams);
//    vector<float> interParamsV;
//    if(nInterParams == 0) {
//        interParamsV.push_back(0.f);
//    } else {
//        for(int i = 0; i < nInterParams; i++)
//            interParamsV.push_back(interParams[i]);
//    }
//    int nFinalParams = 0;
//    const float *finalParams = params.FindFloat("finalparams", &nFinalParams);
//    vector<float> finalParamsV;
//    if(nFinalParams == 0) {
//        finalParamsV.push_back(0.f);
//    } else {
//        for(int i = 0; i < nFinalParams; i++)
//            finalParamsV.push_back(finalParams[i]);
//    }

//    float sigmaN = params.FindOneFloat("sigman", 0.8f);
//    float sigmaR = params.FindOneFloat("sigmar", 0.25f);
//    float sigmaD = params.FindOneFloat("sigmad", 0.6f);
//    float interMseSigma = params.FindOneFloat("intermsesigma", 4.f);
//    float finalMseSigma = params.FindOneFloat("finalmsesigma", 8.f);

//    return new SBFImageFilm(xres, yres, filter, crop, filename,
//                            debug, type, interParamsV, finalParamsV,
//                            sigmaN, sigmaR, sigmaD,
//                            interMseSigma, finalMseSigma);
//}

