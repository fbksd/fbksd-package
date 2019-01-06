
/*
 *  Copyright(c) 2011 Fabrice Rousselle.
 * 
 *  You can redistribute and/or modify this file under the terms of the GNU
 *  General Public License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 */


// film/image.cpp*
#include <iostream>
#include <fstream>
#include "smooth.h"
#include "parallel.h"
#include <limits>
#include <vector>
#include <sys/param.h>
#include <string.h>
#include "denoiser.h"

// SmoothFilm Method Definitions
SmoothFilm::SmoothFilm(int xres, int yres, Filter *filt, float gamma)
    : Film(xres, yres), _splatting(false) {
    filter = filt;
    cropWindow[0] = 0.f;
    cropWindow[1] = 1.f;
    cropWindow[2] = 0.f;
    cropWindow[3] = 1.f;
    filename = "";
    // Compute film image extent
    xPixelStart = Ceil2Int(xResolution * cropWindow[0]);
    xPixelCount = max(1, Ceil2Int(xResolution * cropWindow[1]) - xPixelStart);
    yPixelStart = Ceil2Int(yResolution * cropWindow[2]);
    yPixelCount = max(1, Ceil2Int(yResolution * cropWindow[3]) - yPixelStart);

    // Allocate film image storage
    pixels = new BlockedArray<Pixel>(xPixelCount, yPixelCount);
    // Allocate film image variance storage
    variance = new BlockedArray<BoxVariance>(xPixelCount, yPixelCount);

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

    // Allocate subpixel film image storage
    subPixelRes = 4;
    int nSubPix = subPixelRes * xPixelCount * subPixelRes * yPixelCount;
    subPixels.resize(nSubPix);

    // Create the appropriate denoiser
    int nScales = 4;
    _denoiser = new Denoiser(nScales, xPixelCount, yPixelCount, filt, filename,
        gamma, subPixelRes);
}


void SmoothFilm::AddSample(float* sample) {
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

    // WORKAROUND: sometimes image samples are out of range, causing a stack overflow
    if(x0 > x1 || y0 > y1)
    {
        std::cout << "ERROR: out of range sample [" << sample[0] << ", " << sample[1] << "]" << std::endl;
        return;
    }

    // Loop over filter support and add sample to pixel arrays
    float xyz[3] = {sample[2], sample[3], sample[4]};

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

    // Always use AtomicAdd since adaptive sampling might be using large kernels
    bool syncNeeded = true; // (filter->xWidth > 0.5f || filter->yWidth > 0.5f);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            // Evaluate filter value at $(x,y)$ pixel
            int offset = ify[y-y0]*FILTER_TABLE_SIZE + ifx[x-x0];
            float filterWt = filterTable[offset];

            // Update pixel values with filtered sample contribution
            Pixel &pixel = (*pixels)(x - xPixelStart, y - yPixelStart);
            if (!syncNeeded) {
                pixel.Lxyz[0] += filterWt * xyz[0];
                pixel.Lxyz[1] += filterWt * xyz[1];
                pixel.Lxyz[2] += filterWt * xyz[2];
                pixel.weightSum += filterWt;
            }
            else {
                // Safely update _Lxyz_ and _weightSum_ even with concurrency
                AtomicAdd(&pixel.Lxyz[0], filterWt * xyz[0]);
                AtomicAdd(&pixel.Lxyz[1], filterWt * xyz[1]);
                AtomicAdd(&pixel.Lxyz[2], filterWt * xyz[2]);
                AtomicAdd(&pixel.weightSum, filterWt);
            }
        }
    }

    // Add sample to variance estimate
    int x = Floor2Int(sample[0]);
    int y = Floor2Int(sample[1]);
    if (x < 0 || y < 0 || x >= xPixelCount || y >= yPixelCount)
        return;

    float rgb[3] = {sample[2], sample[3], sample[4]};
    BoxVariance &pixvar = (*variance)(x, y);
    AtomicAdd(&pixvar.LrgbSum[0], rgb[0]);
    AtomicAdd(&pixvar.LrgbSum[1], rgb[1]);
    AtomicAdd(&pixvar.LrgbSum[2], rgb[2]);
    AtomicAdd(&pixvar.LrgbSumSqr[0], rgb[0]*rgb[0]);
    AtomicAdd(&pixvar.LrgbSumSqr[1], rgb[1]*rgb[1]);
    AtomicAdd(&pixvar.LrgbSumSqr[2], rgb[2]*rgb[2]);
    AtomicAdd((AtomicInt32*)&pixvar.nSamples, (int32_t)1);
    // Add sample to subpixel buffer
    x = Floor2Int(subPixelRes * sample[0]);
    y = Floor2Int(subPixelRes * sample[1]);
    int pix = x + y * subPixelRes * xPixelCount;
    SubPixel &subpix = subPixels[pix];
    AtomicAdd(&subpix.Lxyz[0], rgb[0]);
    AtomicAdd(&subpix.Lxyz[1], rgb[1]);
    AtomicAdd(&subpix.Lxyz[2], rgb[2]);
    AtomicAdd(&subpix.weightSum, 1);
}


void SmoothFilm::GetSampleExtent(int *xstart, int *xend,
                                int *ystart, int *yend) const {
    *xstart = Floor2Int(xPixelStart + 0.5f - filter->xWidth);
    *xend   = Floor2Int(xPixelStart + 0.5f + xPixelCount  +
                        filter->xWidth);

    *ystart = Floor2Int(yPixelStart + 0.5f - filter->yWidth);
    *yend   = Floor2Int(yPixelStart + 0.5f + yPixelCount +
                        filter->yWidth);
}


void SmoothFilm::GetPixelExtent(int *xstart, int *xend,
                               int *ystart, int *yend) const {
    *xstart = xPixelStart;
    *xend   = xPixelStart + xPixelCount;
    *ystart = yPixelStart;
    *yend   = yPixelStart + yPixelCount;
}


void SmoothFilm::WriteImage(float* img) {
	// Convert image to RGB and compute final pixel values
    vector<float> rgb(3 * xPixelCount * yPixelCount);
	for (int y = 0, pix = 0; y < yPixelCount; ++y) {
		for (int x = 0; x < xPixelCount; ++x, ++pix) {
			Pixel &pixel = (*pixels)(x, y);

			// Convert pixel XYZ color to RGB
//			XYZToRGB(pixel.Lxyz, &rgb[3*pix]);
            memcpy(&rgb[3*pix], pixel.Lxyz, 3*sizeof(float));

			// Normalize pixel with weight sum
			float weightSum = pixel.weightSum;
			if (weightSum != 0.f) {
				float invWt = 1.f / weightSum;
				rgb[3*pix+0] = max(0.f, rgb[3*pix+0] * invWt);
				rgb[3*pix+1] = max(0.f, rgb[3*pix+1] * invWt);
				rgb[3*pix+2] = max(0.f, rgb[3*pix+2] * invWt);
			}
		}
	}
//    ::WriteImage(filename, &rgb[0], NULL, xPixelCount, yPixelCount, xPixelCount, yPixelCount, 0, 0);

    if (!_splatting) {
        _denoiser->SetPixelMeanAndVarianceFinal(subPixels, variance);
        _denoiser->WriteImage(img);
    }
}


void SmoothFilm::UpdateDisplay(int x0, int y0, int x1, int y1,
    float splatScale) {
}


//SmoothFilm *CreateSmoothFilm(const ParamSet &params, Filter *filter) {
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
//    // Read denoiser parameter
//    float gamma = params.FindOneFloat("gamma", .2f);

//    return new SmoothFilm(xres, yres, filter, crop, filename, openwin, gamma);
//}


