/* 
 * File:   dualfilm.cpp
 * Author: rousselle
 * 
 * Created on March 13, 2012, 11:35 AM
 */

#include "dualfilm.h"

#include <iostream>
#include <fstream>
#include "stdafx.h"
#include "parallel.h"
#include "image.h"
#include <limits>
#include <vector>
#include <sys/param.h>
#include <string.h>
#include "nlmdenoiser.h"


// DualFilm Method Definitions
DualFilm::DualFilm(int xres, int yres, Filter *filt, int wnd_rad, float k, int ptc_rad)
    : Film(xres, yres) {
    filter = filt;
    cropWindow[0] = 0.f;
    cropWindow[1] = 1.f;
    cropWindow[2] = 0.f;
    cropWindow[3] = 1.f;
    // Compute film image extent
    xPixelStart = Ceil2Int(xResolution * cropWindow[0]);
    xPixelCount = max(1, Ceil2Int(xResolution * cropWindow[1]) - xPixelStart);
    yPixelStart = Ceil2Int(yResolution * cropWindow[2]);
    yPixelCount = max(1, Ceil2Int(yResolution * cropWindow[3]) - yPixelStart);

    // Allocate film image storage
    int nPix = xPixelCount * yPixelCount;
    pixelsA.resize(nPix);   pixelsB.resize(nPix);

    // Precompute filter weight table
#define FILTER_TABLE_SIZE 16
    filterTable = new float[FILTER_TABLE_SIZE * FILTER_TABLE_SIZE];
    float *ftp = filterTable;
    for (int y = 0; y < FILTER_TABLE_SIZE; ++y) {
        float fy = ((float)y + .5f) *
                   filter->yWidth / FILTER_TABLE_SIZE;
        for (int x = 0; x < FILTER_TABLE_SIZE; ++x) {
            float fx = ((float)x + .5f) *
                       filter->xWidth / FILTER_TABLE_SIZE;
            *ftp++ = filter->Evaluate(fx, fy);
        }
    }

    // Possibly open window for image display
//    if (openWindow || PbrtOptions.openWindow) {
//        Warning("Support for opening image display window not available in this build.");
//    }

    // Allocate subpixel film image storage
    subPixelRes = 4;
    int nSubPix = subPixelRes * xPixelCount * subPixelRes * yPixelCount;
    subPixelsA.resize(nSubPix);   subPixelsB.resize(nSubPix);

    // Create the appropriate denoiser
    _denoiser = new NlmeansDenoiser(xPixelCount, yPixelCount, filt, wnd_rad, k, ptc_rad, subPixelRes);
}


void DualFilm::AddSample(float* sample, TargetBuffer target) {
    // Compute sample's raster extent
    float dimageX = sample[0] - 0.5f;
    float dimageY = sample[1] - 0.5f;
    int x0 = Ceil2Int (dimageX - filter->xWidth);
    int x1 = Floor2Int(dimageX + filter->xWidth);
    int y0 = Ceil2Int (dimageY - filter->yWidth);
    int y1 = Floor2Int(dimageY + filter->yWidth);
    x0 = max(x0, xPixelStart);
    x1 = min(x1, xPixelStart + xPixelCount - 1);
    y0 = max(y0, yPixelStart);
    y1 = min(y1, yPixelStart + yPixelCount - 1);
//    if ((x1-x0) < 0 || (y1-y0) < 0)
//    {
//        PBRT_SAMPLE_OUTSIDE_IMAGE_EXTENT(const_cast<CameraSample *>(&sample));
//        return;
//    }

    // Loop over filter support and add sample to pixel arrays
    float rgb[3];
    memcpy(rgb, sample+2, sizeof(float)*3);

    // Precompute $x$ and $y$ filter table offsets
    int *ifx = ALLOCA(int, x1 - x0 + 1);
    for (int x = x0; x <= x1; ++x) {
        float fx = fabsf((x - dimageX) *
                         filter->invXWidth * FILTER_TABLE_SIZE);
        ifx[x-x0] = min(Floor2Int(fx), FILTER_TABLE_SIZE-1);
    }
    int *ify = ALLOCA(int, y1 - y0 + 1);
    for (int y = y0; y <= y1; ++y) {
        float fy = fabsf((y - dimageY) *
                         filter->invYWidth * FILTER_TABLE_SIZE);
        ify[y-y0] = min(Floor2Int(fy), FILTER_TABLE_SIZE-1);
    }

    // Select the right target buffer
    vector<NlmeansPixel> &pixels = (target == BUFFER_A) ? pixelsA : pixelsB;
    
    // Always use AtomicAdd since adaptive sampling might be using large kernels
    bool syncNeeded = true; // (filter->xWidth > 0.5f || filter->yWidth > 0.5f);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            // Evaluate filter value at $(x,y)$ pixel
            int offset = ify[y-y0]*FILTER_TABLE_SIZE + ifx[x-x0];
            float filterWt = filterTable[offset];
            
            // Update pixel values with filtered sample contribution
            int pix = xPixelCount * (y - yPixelStart) + x - xPixelStart;
            NlmeansPixel &pixel = pixels[pix];
            if (!syncNeeded) {
                pixel._Lrgb[0] += filterWt * rgb[0];
                pixel._Lrgb[1] += filterWt * rgb[1];
                pixel._Lrgb[2] += filterWt * rgb[2];
                pixel._weightSum += filterWt;
            }
            else {
                // Safely update _Lrgb_ and _weightSum_ even with concurrency
                AtomicAdd(&pixel._Lrgb[0], filterWt * rgb[0]);
                AtomicAdd(&pixel._Lrgb[1], filterWt * rgb[1]);
                AtomicAdd(&pixel._Lrgb[2], filterWt * rgb[2]);
                AtomicAdd(&pixel._weightSum, filterWt);
            }
        }
    }
    
    // We're done is this sample is outside the film buffer
    int x = Floor2Int(sample[0]);
    int y = Floor2Int(sample[1]);
    if (x < 0 || y < 0 || x >= xPixelCount || y >= yPixelCount)
        return;
    
    // Store variance information
    int pix = xPixelCount * (y - yPixelStart) + x - xPixelStart;
    NlmeansPixel &pixel = pixels[pix];
    AtomicAdd(&pixel._LrgbSumBox[0], rgb[0]);
    AtomicAdd(&pixel._LrgbSumBox[1], rgb[1]);
    AtomicAdd(&pixel._LrgbSumBox[2], rgb[2]);
    AtomicAdd(&pixel._LrgbSumSqrBox[0], rgb[0]*rgb[0]);
    AtomicAdd(&pixel._LrgbSumSqrBox[1], rgb[1]*rgb[1]);
    AtomicAdd(&pixel._LrgbSumSqrBox[2], rgb[2]*rgb[2]);
    AtomicAdd((AtomicInt32*)&pixel._nSamplesBox, (int32_t)1);
    
    // Store on sub-pixel grid
    x = Floor2Int(subPixelRes * sample[0]);
    y = Floor2Int(subPixelRes * sample[1]);
    pix = x + y * subPixelRes * xPixelCount;
    NlmeansSubPixel &subpix = (target == BUFFER_A) ? subPixelsA[pix] : subPixelsB[pix];
    AtomicAdd(&subpix._LrgbSumBox[0], rgb[0]);
    AtomicAdd(&subpix._LrgbSumBox[1], rgb[1]);
    AtomicAdd(&subpix._LrgbSumBox[2], rgb[2]);
    AtomicAdd(&subpix._nSamplesBox, 1);
}


void DualFilm::GetSampleExtent(int *xstart, int *xend,
                                int *ystart, int *yend) const {
    *xstart = Floor2Int(xPixelStart + 0.5f - filter->xWidth);
    *xend   = Floor2Int(xPixelStart + 0.5f + xPixelCount  +
                        filter->xWidth);

    *ystart = Floor2Int(yPixelStart + 0.5f - filter->yWidth);
    *yend   = Floor2Int(yPixelStart + 0.5f + yPixelCount +
                        filter->yWidth);
}


void DualFilm::GetPixelExtent(int *xstart, int *xend,
                               int *ystart, int *yend) const {
    *xstart = xPixelStart;
    *xend   = xPixelStart + xPixelCount;
    *ystart = yPixelStart;
    *yend   = yPixelStart + yPixelCount;
}


void DualFilm::WriteImage(float* img) {
    // Dump the noisy data by combining the two buffers
    ImageBuffer rgb(3 * xPixelCount * yPixelCount);
    ImageBuffer rgbA(3 * xPixelCount * yPixelCount);
    ImageBuffer rgbB(3 * xPixelCount * yPixelCount);
    for (int y = 0, pix = 0; y < yPixelCount; ++y) {
        for (int x = 0; x < xPixelCount; ++x, ++pix) {
            int r = 3*pix+0, g = 3*pix+1, b = 3*pix+2;
            
            // Normalize pixel with weight sum
            NlmeansPixel &pixelA = pixelsA[pix];
            float wgtSumA = pixelA._weightSum;
            if (wgtSumA != 0.f) {
                    float invWt = 1.f / wgtSumA;
                    rgbA[r] = max(0.f, pixelA._Lrgb[0] * invWt);
                    rgbA[g] = max(0.f, pixelA._Lrgb[1] * invWt);
                    rgbA[b] = max(0.f, pixelA._Lrgb[2] * invWt);
            }
            
            // Normalize pixel with weight sum
            NlmeansPixel &pixelB = pixelsB[pix];
            float wgtSumB = pixelB._weightSum;
            if (wgtSumB != 0.f) {
                    float invWt = 1.f / wgtSumB;
                    rgbB[r] = max(0.f, pixelB._Lrgb[0] * invWt);
                    rgbB[g] = max(0.f, pixelB._Lrgb[1] * invWt);
                    rgbB[b] = max(0.f, pixelB._Lrgb[2] * invWt);
            }
            
            // Sum both buffers
            float wgtSum = wgtSumA + wgtSumB;
            if (wgtSumA != 0.f) {
                    rgb[r] = (wgtSumA * rgbA[r] + wgtSumB * rgbB[r]) / wgtSum;
                    rgb[g] = (wgtSumA * rgbA[g] + wgtSumB * rgbB[g]) / wgtSum;
                    rgb[b] = (wgtSumA * rgbA[b] + wgtSumB * rgbB[b]) / wgtSum;
            }
        }
    }
//    ::WriteImage(filename, &rgb[0], NULL, xPixelCount, yPixelCount, xPixelCount, yPixelCount, 0, 0);
    
    // Filter out noise from data and store the result
    if (!_denoiser->IsReady())
        _denoiser->UpdatePixelData(pixelsA, pixelsB, subPixelsA, subPixelsB, NLM_DATA_FINAL);
    _denoiser->WriteImage(img);
}


void DualFilm::UpdateDisplay(int x0, int y0, int x1, int y1,
    float splatScale) {
}


//DualFilm *CreateDualFilm(const ParamSet &params, Filter *filter) {
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
//    bool openwin = params.FindOneBool("display", false);
//    float crop[4] = { 0, 1, 0, 1 };
//    int cwi;
//    const float *cr = params.FindFloat("cropwindow", &cwi);
//    if (cr && cwi == 4) {
//        crop[0] = Clamp(min(cr[0], cr[1]), 0., 1.);
//        crop[1] = Clamp(max(cr[0], cr[1]), 0., 1.);
//        crop[2] = Clamp(min(cr[2], cr[3]), 0., 1.);
//        crop[3] = Clamp(max(cr[2], cr[3]), 0., 1.);
//    }

//    ///////////////////////
//    // Parse denoiser flags

//    // Filter width
//    int wnd_rad = params.FindOneInt("wnd_rad", 10);
    
//    // Damping parameter
//    float k = params.FindOneFloat("k", .1f);
    
//    // Patchsize
//    int ptc_rad = params.FindOneInt("ptc_rad", 3);

//    return new DualFilm(xres, yres, filter, crop, filename, openwin, wnd_rad,
//        k, ptc_rad);
//}


