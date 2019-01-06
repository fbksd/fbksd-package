/*
    pbrt source code Copyright(c) 1998-2010 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

    pbrt is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.  Note that the text contents of
    the book "Physically Based Rendering" are *not* licensed under the
    GNU GPL.

    pbrt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 */

/////////////////////////////////////////////////////////////////////////////////
// This file is modified from exisiting pbrt codes in order to test the lwrr project.
// Author: Bochang Moon (moonbochang@gmail.com)
/////////////////////////////////////////////////////////////////////////////////

// film/image.cpp*
#include "stdafx.h"
#include "parallel.h"
#include "pbrt.h"
#include <algorithm>
#include <omp.h>

#include "lwr_film.h"

template<class T>
void LWR_Film::initForUpdatingErrorMap(T* MSE_map)
{
	m_nextProcessIdx = 0;
	m_pixelErrors.resize(xPixelCount * yPixelCount);
}

void LWR_Film::generateScramblingInfo(int num1D, int num2D)
{
	int nPix = xPixelCount * yPixelCount;
	m_pixInfo.resize(nPix);
}

void LWR_Film::computeSampleMap(const float* map_MSE, const int numSamplePerIterations)
{	
	const int nPix = xPixelCount * yPixelCount;

	m_pixelErrors.resize(nPix);

	// initialize
#pragma omp parallel for schedule(guided, 4)
	for (int i = 0; i < nPix; ++i) {
		m_pixelErrors[i].m_numSample = 0;
		m_pixelErrors[i].m_idx = i;
		m_pixelErrors[i].m_error = map_MSE[i * 3 + 0];		
	}

	std::vector<PixSPP> tempArrError;
	tempArrError.resize(m_pixelErrors.size());

	std::copy(m_pixelErrors.begin(), m_pixelErrors.end(), tempArrError.begin());	
	std::sort(tempArrError.begin(), tempArrError.end());

	int idxQuantile = nPix * 0.95;				
	double upper_mse = tempArrError[idxQuantile].m_error;		
	long double total_mse = 0.0;

#pragma omp parallel for reduction(+:total_mse)
	for (int i = 0; i < idxQuantile; ++i) 	
		total_mse = total_mse + tempArrError[i].m_error;
	total_mse += upper_mse * (nPix - idxQuantile + 1);

	int nGeneratedSample = 0;

	//
	m_maxSPP = Ceil2Int((long double)numSamplePerIterations * ((long double)upper_mse / total_mse));	

	for (int i = nPix-1; i >= 0; --i) {
		int idx = tempArrError[i].m_idx;
		double mse = min((double)tempArrError[i].m_error, upper_mse);
		int nsample = Ceil2Int((long double)numSamplePerIterations * ((long double)mse / total_mse));		
		m_pixelErrors[idx].m_numSample = nsample;

		nGeneratedSample += nsample;
		if (nGeneratedSample > numSamplePerIterations)
			break;		
	}

	int count = 0;
	for (int i = 0; i < nPix; ++i) {
		if (m_pixelErrors[i].m_numSample > 0) {			
			tempArrError[count].m_idx = i;
			tempArrError[count].m_numSample = m_pixelErrors[i].m_numSample;	
			++count;
		}
	}
	std::copy_n(tempArrError.begin(), count, m_pixelErrors.begin());
	m_pixelErrors.resize(count);
}

void LWR_Film::test_lwrr(int numSamplePerIterations, bool isLastPass, float* img)
{
	++m_iterationCount;

//	Timer solverTimer;
//	solverTimer.Start();

	pLWRR->run_lwrr(numSamplePerIterations, isLastPass);

//	solverTimer.Stop();
//	printf("\nAnalysis Error Elapsed Time = %.1f sec\n", solverTimer.Time());

	if (isLastPass) 
	{
		std::string dbgFileName;
		float* inImg = pLWRR->get_inputImg();
		float* ranks = pLWRR->get_ranks();
		float* optImg = pLWRR->get_optImg();

//		dbgFileName = "in_" + filename;
//		WriteXYZImage(dbgFileName, inImg, 1.f);

//		dbgFileName = "rank_" + filename;
//		WriteXYZImageGrey(dbgFileName, ranks, 1.f / (nDimens + 1));

//		dbgFileName = "adaptive_" + filename;
        WriteXYZImage(dbgFileName, optImg, 1.f, img);

//		dbgFileName = "spp_" + filename;
        //WriteXYZImageGrey(dbgFileName, m_mapSPP, 1.f, img);

		//dbgFileName = "global_width_h_" + filename;
		//WriteXYZImageGrey(dbgFileName, pLWRR->m_mem->_width_img, 1.f);

		//dbgFileName = "depth_" + filename;
//        WriteXYZImageGrey(dbgFileName, pLWRR->m_mem->_depth, 1.f, img);

		//dbgFileName = "normal_" + filename;
//        WriteXYZImage(dbgFileName, pLWRR->m_mem->_normal, 1.f, img);

		//dbgFileName = "texture_" + filename;
//        WriteXYZImage(dbgFileName, pLWRR->m_mem->_texture, 1.f, img);
	}
	else {	
		m_nextProcessIdx = 0;

		float* mse_optImg = pLWRR->get_mse_optImg();
		computeSampleMap(mse_optImg, numSamplePerIterations);	

		printf("MAX spp = %d\n", m_maxSPP);	
	}
}

// ImageFilm Method Definitions
LWR_Film::LWR_Film(int xres, int yres, Filter *filt, float rayScale)
              : Film(xres, yres), m_rayScale(rayScale)
{
    filter = filt;
    cropWindow[0] = 0;
    cropWindow[1] = 1;
    cropWindow[2] = 0;
    cropWindow[3] = 1;
    // Compute film image extent
    xPixelStart = Ceil2Int(xResolution * cropWindow[0]);
    xPixelCount = max(1, Ceil2Int(xResolution * cropWindow[1]) - xPixelStart);
    yPixelStart = Ceil2Int(yResolution * cropWindow[2]);
    yPixelCount = max(1, Ceil2Int(yResolution * cropWindow[3]) - yPixelStart);

    // Allocate film image storage
    pixels = new BlockedArray<Pixel>(xPixelCount, yPixelCount);

	m_nextProcessIdx = 0;
	m_iterationCount = 0;

	int nPix = xPixelCount * yPixelCount;

	// Basic Buffers - initialization
	m_mapSPP =      (int*)calloc(nPix, sizeof(int));
	m_accImg =      (float*)calloc(nPix * 3, sizeof(float));
	m_accImg2 =     (float*)calloc(nPix * 3, sizeof(float));
	m_accNormal =   (float*)calloc(nPix * 3, sizeof(float));
	m_accNormal2 =  (float*)calloc(nPix * 3, sizeof(float));
	m_accTexture =  (float*)calloc(nPix * 3, sizeof(float));
	m_accTexture2 = (float*)calloc(nPix * 3, sizeof(float));
	m_accDepth =    (float*)calloc(nPix, sizeof(float));
	m_accDepth2 =   (float*)calloc(nPix, sizeof(float));

	// Additional
	m_accTextureMoving = m_accTextureMoving2 = NULL;
	m_mapMovingSPP = NULL;	

#ifdef FEATURE_MOTION
	m_accTextureMoving = (float*)calloc(nPix * 3, sizeof(float));		
	m_accTextureMoving2 = (float*)calloc(nPix * 3, sizeof(float));						
	m_mapMovingSPP = (int*)calloc(nPix, sizeof(int));
#endif

	pLWRR = new LWRR(xPixelCount, yPixelCount, nPix);
	pLWRR->init_lwrr(m_accImg, m_accImg2, m_accNormal, m_accNormal2, m_accTexture, m_accTexture2,
		             m_accDepth, m_accDepth2, m_mapSPP, m_mapMovingSPP, m_accTextureMoving, m_accTextureMoving2);
	//

    // Possibly open window for image display
//    if (openWindow || PbrtOptions.openWindow) {
//        Warning("Support for opening image display window not available in this build.");
//    }

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
}

void LWR_Film::initializeGlobalVariables(const int initSPP)
{
	m_pixelErrors.resize(xPixelCount*yPixelCount);
	for (int i = 0; i < xPixelCount*yPixelCount; ++i) {		
		m_pixelErrors[i].m_idx = i;		
		m_pixelErrors[i].m_error = 0.0;
		m_pixelErrors[i].m_numGeneratedSamples = 0;
		m_pixelErrors[i].m_numSample = initSPP;		
	}
	m_maxSPP = initSPP;
	m_globalLock = Mutex::Create();
}

void LWR_Film::AddSampleExtended(float* sample, const int idxPix)
{
    int idx = idxPix;

    float xyz[3], texture[3];
    memcpy(xyz, &sample[COLOR_R], 3*sizeof(float));
    memcpy(texture, &sample[TEXTURE_R], 3*sizeof(float));
//	L.ToRGB(xyz);
//	isect.rho.ToRGB(texture);
    for (int c = 0; c < 3; ++c) {
        AtomicAdd(&m_accImg[idx * 3 + c], xyz[c]);
        AtomicAdd(&m_accImg2[idx * 3 + c], xyz[c] * xyz[c]);
    }

    AtomicAdd((AtomicInt32*)&m_mapSPP[idx], (int32_t)1);

#ifndef FEATURE_MOTION
    for (int c = 0; c < 3; ++c) {
        AtomicAdd(&m_accTexture[idx * 3 + c], texture[c]);
        AtomicAdd(&m_accTexture2[idx * 3 + c], texture[c] * texture[c]);
    }
        AtomicAdd(&m_accNormal[idx * 3 + 0], sample[NORMAL_X]);
        AtomicAdd(&m_accNormal[idx * 3 + 1], sample[NORMAL_Y]);
        AtomicAdd(&m_accNormal[idx * 3 + 2], sample[NORMAL_Z]);

        AtomicAdd(&m_accNormal2[idx * 3 + 0], sample[NORMAL_X]*sample[NORMAL_X]);
        AtomicAdd(&m_accNormal2[idx * 3 + 1], sample[NORMAL_Y]*sample[NORMAL_Y]);
        AtomicAdd(&m_accNormal2[idx * 3 + 2], sample[NORMAL_Z]*sample[NORMAL_Z]);
        AtomicAdd(&m_accDepth[idx], sample[DEPTH]);
        AtomicAdd(&m_accDepth2[idx], sample[DEPTH]*sample[DEPTH]);
//    if (!isect.shadingN.HasNaNs()) {
//        AtomicAdd(&m_accNormal[idx * 3 + 0], isect.shadingN.x);
//        AtomicAdd(&m_accNormal[idx * 3 + 1], isect.shadingN.y);
//        AtomicAdd(&m_accNormal[idx * 3 + 2], isect.shadingN.z);

//        AtomicAdd(&m_accNormal2[idx * 3 + 0], isect.shadingN.x * isect.shadingN.x);
//        AtomicAdd(&m_accNormal2[idx * 3 + 1], isect.shadingN.y * isect.shadingN.y);
//        AtomicAdd(&m_accNormal2[idx * 3 + 2], isect.shadingN.z * isect.shadingN.z);
//    }
//    AtomicAdd(&m_accDepth[idx], isect.depth);
//    AtomicAdd(&m_accDepth2[idx], isect.depth * isect.depth);

#else
    if (isect.isAnimated) {
        AtomicAdd((AtomicInt32*)&m_mapMovingSPP[idx], (int32_t)1);
        for (int c = 0; c < 3; ++c) {
            AtomicAdd(&m_accTextureMoving[idx * 3 + c], texture[c]);
            AtomicAdd(&m_accTextureMoving2[idx * 3 + c], texture[c] * texture[c]);
        }
    }
    else {
        // Still objects
        for (int c = 0; c < 3; ++c) {
            AtomicAdd(&m_accTexture[idx * 3 + c], texture[c]);
            AtomicAdd(&m_accTexture2[idx * 3 + c], texture[c] * texture[c]);
        }
        if (!isect.shadingN.HasNaNs()) {
            AtomicAdd(&m_accNormal[idx * 3 + 0], isect.shadingN.x);
            AtomicAdd(&m_accNormal[idx * 3 + 1], isect.shadingN.y);
            AtomicAdd(&m_accNormal[idx * 3 + 2], isect.shadingN.z);

            AtomicAdd(&m_accNormal2[idx * 3 + 0], isect.shadingN.x * isect.shadingN.x);
            AtomicAdd(&m_accNormal2[idx * 3 + 1], isect.shadingN.y * isect.shadingN.y);
            AtomicAdd(&m_accNormal2[idx * 3 + 2], isect.shadingN.z * isect.shadingN.z);
        }
        AtomicAdd(&m_accDepth[idx], isect.depth);
        AtomicAdd(&m_accDepth2[idx], isect.depth * isect.depth);
    }
#endif
}


void LWR_Film::GetSampleExtent(int *xstart, int *xend,
                                int *ystart, int *yend) const {
    *xstart = Floor2Int(xPixelStart + 0.5f - filter->xWidth);
    *xend   = Floor2Int(xPixelStart + 0.5f + xPixelCount  +
                        filter->xWidth);

    *ystart = Floor2Int(yPixelStart + 0.5f - filter->yWidth);
    *yend   = Floor2Int(yPixelStart + 0.5f + yPixelCount +
                        filter->yWidth);
}


void LWR_Film::GetPixelExtent(int *xstart, int *xend,
                               int *ystart, int *yend) const {
    *xstart = xPixelStart;
    *xend   = xPixelStart + xPixelCount;
    *ystart = yPixelStart;
    *yend   = yPixelStart + yPixelCount;
}

template<class T> 
void LWR_Film::WriteXYZImageGrey(const string fileName, const T *xyz, const float scale, float* img)
{
    // Convert image to RGB and compute final pixel values
//    int nPix = xPixelCount * yPixelCount;
//    float *rgb = new float[3*nPix];

    int offset = 0;
    for (int y = 0; y < yPixelCount; ++y) {
        for (int x = 0; x < xPixelCount; ++x) {
            float v = max(0.f, scale * (float)xyz[offset]);
            img[offset * 3 + 0] = v;
            img[offset * 3 + 1] = v;
            img[offset * 3 + 2] = v;
            ++offset;
        }
    }

    // Write RGB image
//    ::WriteImage(fileName, rgb, NULL, xPixelCount, yPixelCount,
//                 xResolution, yResolution, xPixelStart, yPixelStart);

//    delete[] rgb;
}

void LWR_Film::WriteCropImage(const string fileName, float* _img, int xStart, int yStart, int xCount, int yCount, bool isGrey)
{
	    // Convert image to RGB and compute final pixel values
    int nPix = xCount * yCount;
    float *rgb = new float[3*nPix];

    int offset = 0;

    for (int y = yStart; y < yStart + yCount; ++y) {
        for (int x = xStart; x < xStart + xCount; ++x) {
			int idx = y * xPixelCount + x;
			if (isGrey) {
				rgb[offset * 3 + 0] = max(0.f, (float)_img[idx]);
				rgb[offset * 3 + 1] = max(0.f, (float)_img[idx]);
				rgb[offset * 3 + 2] = max(0.f, (float)_img[idx]);
			}
			else {
				rgb[offset * 3 + 0] = max(0.f, (float)_img[idx * 3 + 0]);
				rgb[offset * 3 + 1] = max(0.f, (float)_img[idx * 3 + 1]);
				rgb[offset * 3 + 2] = max(0.f, (float)_img[idx * 3 + 2]);	
			}
            ++offset;
        }
    }	

    // Write RGB image
//    ::WriteImage(fileName, rgb, NULL, xCount, yCount,
//                 xCount, yCount, 0, 0);
	delete[] rgb;
}

template<class T> 
void LWR_Film::WriteXYZImage(const string fileName, const T *xyz, const float scale, float* img)
{
    // Convert image to RGB and compute final pixel values
    int nPix = xPixelCount * yPixelCount;
//    float *rgb = new float[3*nPix];

    int offset = 0;
    for (int y = 0; y < yPixelCount; ++y) {
        for (int x = 0; x < xPixelCount; ++x) {
            img[offset * 3 + 0] = max(0.f, scale * (float)xyz[offset * 3 + 0]);
            img[offset * 3 + 1] = max(0.f, scale * (float)xyz[offset * 3 + 1]);
            img[offset * 3 + 2] = max(0.f, scale * (float)xyz[offset * 3 + 2]);
            ++offset;
        }
    }	

    // Write RGB image
//    ::WriteImage(fileName, rgb, NULL, xPixelCount, yPixelCount,
//                 xResolution, yResolution, xPixelStart, yPixelStart);

//    delete[] rgb;
}

void LWR_Film::WriteImage(float* img) {
    // Convert image to RGB and compute final pixel values
    int nPix = xPixelCount * yPixelCount;
    float *rgb = new float[3*nPix];
	float *outrgb = new float[3*nPix];
    int offset = 0;
	int numTotalSample = 0;
    for (int y = 0; y < yPixelCount; ++y) {
        for (int x = 0; x < xPixelCount; ++x) {
            // Convert pixel XYZ color to RGB
//            XYZToRGB((*pixels)(x, y).Lxyz, &rgb[3*offset]);
            memcpy(&rgb[3*offset], (*pixels)(x, y).Lxyz, 3*sizeof(float));

            // Normalize pixel with weight sum
            float weightSum = (*pixels)(x, y).weightSum;
			numTotalSample += (*pixels)(x,y).count;
            if (weightSum != 0.f) {
                float invWt = 1.f / weightSum;
                rgb[3*offset  ] = max(0.f, rgb[3*offset  ] * invWt);
                rgb[3*offset+1] = max(0.f, rgb[3*offset+1] * invWt);
                rgb[3*offset+2] = max(0.f, rgb[3*offset+2] * invWt);
            }

            // Add splat value at pixel
            //float splatRGB[3];
            //XYZToRGB((*pixels)(x, y).splatXYZ, splatRGB);
            //rgb[3*offset  ] += splatScale * splatRGB[0];
            //rgb[3*offset+1] += splatScale * splatRGB[1];
            //rgb[3*offset+2] += splatScale * splatRGB[2];
            ++offset;
        }
    }

	float spp = numTotalSample / (float)nPix;
	printf("Average SPP in Write Image = %.1f\n", spp);

    // Write RGB image
//    ::WriteImage(filename, rgb, NULL, xPixelCount, yPixelCount,
//                 xResolution, yResolution, xPixelStart, yPixelStart);
	
	// image final reconstruction
    test_lwrr(0, true, img);

    // Release temporary image memory
    delete[] rgb;
	delete[] outrgb;
}


void LWR_Film::UpdateDisplay(int x0, int y0, int x1, int y1,
    float splatScale) {
}


//LWR_Film *CreateLWRFilm(const ParamSet &params, Filter *filter) {
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
	
//	float rayScale = params.FindOneFloat("rayscale", 0.f);
	
//    return new LWR_Film(xres, yres, filter, crop, filename, openwin, rayScale);
//}



 









