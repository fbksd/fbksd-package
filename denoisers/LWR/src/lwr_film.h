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

#if defined(_MSC_VER)
#pragma once
#endif

#ifndef PBRT_LWR_FILM_H
#define PBRT_LWR_FILM_H

// film/image.h*
#include "pbrt.h"
#include "film.h"
#include "sampler.h"
#include "filter.h"
#include "lwrr.h"

class PixInfo
{
public:
	PixInfo() {	m_numGenerated = 0;	};
	~PixInfo() {}
public:
	int m_numGenerated;
	uint32_t scramble_image[2];
	uint32_t scramble_lens[2];
	uint32_t scramble_time;
	vector<uint32_t> scramble_1D;
	vector<uint32_t> scramble_2D;
};

struct PixSPP 
{
	PixSPP() {}
	PixSPP(int idx, float error) {
		m_idx = idx;
		m_error = error;
	}
	bool operator< (const PixSPP& b) const {
		return (m_error < b.m_error);
	}
	int m_idx;
	float m_error;
	int m_numSample;
	int m_numGeneratedSamples;
};


// ImageFilm Declarations
class LWR_Film : public Film {
public:
	void WriteCropImage(const string fileName, float* _img, int xStart, int yStart, int xCount, int yCount, bool isGrey = false);

	template<class T> 
    void WriteXYZImage(const string fileName, const T *xyz, const float scale, float* img);

	template<class T> 
    void WriteXYZImageGrey(const string fileName, const T *xyz, const float scale, float* img);
	void initializeGlobalVariables(const int initSPP);

public:
    // ImageFilm Public Methods
    LWR_Film(int xres, int yres, Filter *filt, float rayScale);

    ~LWR_Film() 
	{
		free(m_accImg);
		free(m_accImg2);
		free(m_accNormal);
		free(m_accNormal2);
		free(m_accTexture);
		free(m_accTexture2);
		free(m_accDepth);
		free(m_accDepth2);
		free(m_mapSPP);
		//
        delete pixels;
        delete[] filterTable;

		if (m_accTextureMoving)
			free(m_accTextureMoving);
		if (m_accTextureMoving2)
			free(m_accTextureMoving2);
		if (m_mapMovingSPP)
			free(m_mapMovingSPP);	

		Mutex::Destroy(m_globalLock);

		delete pLWRR;
	}


    void AddSampleExtended(float* sample, const int idxPix);

    void GetSampleExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void GetPixelExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void WriteImage(float* img);
    void UpdateDisplay(int x0, int y0, int x1, int y1, float splatScale);

	inline int getXPixelStart() const { return xPixelStart; }
	inline int getYPixelStart() const { return yPixelStart; }
	inline int getXPixelCount() const { return xPixelCount; }
	inline int getYPixelCount() const { return yPixelCount; }

	//

	template<class T> 
	void initForUpdatingErrorMap(T* MSE_map);
    void test_lwrr(int numSamplePerIterations, bool isLastPass = false, float* img = nullptr);
	void computeSampleMap(const float* map_MSE, const int numSamplePerIterations);
	
	// Initialization
	void generateScramblingInfo(int num1D, int num2D);

	// Utility
	inline int getPixelReference() 
	{
		int idxPixel = m_nextProcessIdx++;
		return idxPixel;					
	}
	/////////////////////////////////////////////////////////////////
	void AppendStrToFileName(std::string appendStr) { filename += appendStr; };

private:
    // ImageFilm Private Data
    Filter *filter;
	float *filterTable;
    float cropWindow[4];
    string filename;
    int xPixelStart, yPixelStart, xPixelCount, yPixelCount;

    struct Pixel {
        Pixel() {
            for (int i = 0; i < 3; ++i) {
				Lxyz[i] = 0.f;
				Lxyz2[i] = 0.f;
				//splatXYZ[i] = 0.f;	// for Metropolis sampling
			}
            weightSum = 0.f;
			count = 0;
        }
		// 12B
        float Lxyz[3];

		// 4B
        float weightSum;

		// 12B - for computing variance
		float Lxyz2[3];

		// 12B - Metropolis require this variable. So this should be used when you use Metropolis sampling
        //float splatXYZ[3];

        //float pad;
		// we use count for denoting ray sample count instead of the dummy (pad)
		// 4B
		float count;
    };
    BlockedArray<Pixel> *pixels;

	int m_iterationCount;

	// global variables which shold be shared among threads
	int m_nextProcessIdx;			

	// Input Buffers 
	float* m_accImg;
	float* m_accImg2;
	float* m_accNormal;
	float* m_accNormal2;
	float* m_accTexture;
	float* m_accTexture2;
	float* m_accDepth;
	float* m_accDepth2;
	int* m_mapSPP;
	// Additional feature buffers
	float* m_accTextureMoving;
	float* m_accTextureMoving2;
	int* m_mapMovingSPP;

public:
	Mutex* m_globalLock;
	int m_maxSPP;
	float m_maxDepth;
	int m_samplesPerPixel;

	vector<PixSPP> m_pixelErrors;
	vector<PixInfo> m_pixInfo;

	float m_rayScale;

	// Test LWRR
	LWRR* pLWRR;
};


LWR_Film *CreateLWRFilm(const ParamSet &params, Filter *filter);

#endif 
