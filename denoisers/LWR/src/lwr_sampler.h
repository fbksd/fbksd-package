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

#pragma once

#include "pbrt.h"
#include "sampler.h"
#include "montecarlo.h"
#include "lwr_film.h"
#include <iostream>

class LWR_Sampler : public Sampler {
public:
    LWR_Sampler(int xstart, int xend, int ystart, int yend, int minsamples, int spp, float sopen, float sclose, int nIterations, LWR_Film *film);
    LWR_Sampler(const LWR_Sampler *parent, int xstart, int xend, int ystart, int yend);    
    ~LWR_Sampler();

    Sampler *GetSubSampler(int num, int count);
    int RoundSize(int size) const { return RoundUpPow2(size); }
    void SetMaximumSampleCount(int maxSampleCount) {
        m_maxSPP = maxSampleCount;
        delete[] m_sampleBuf;
        m_sampleBuf = NULL;
    }
    int MaximumSampleCount() 
	{ 
		int maxSPP = 0;
		int xSize = m_myFilm->xResolution;
		for (int y = m_yStartSub; y < m_yEndSub; ++y) {
			for (int x = m_xStartSub; x < m_xEndSub; ++x) {
				int idx = y * xSize + x;
				const PixSPP* pPixel = &(m_myFilm->m_pixelErrors[idx]);
				maxSPP = max(maxSPP, pPixel->m_numSample);
			}
		}		
		return maxSPP; 	
	}

    int GetMoreSamplesWithIdx(std::vector<float> *sample, RNG &rng, int& pixIdx);

    bool ReportResults(Sample *samples, const RayDifferential *rays,
        const Spectrum *Ls, const Intersection *isects, int count);

    int GetIterationCount() { return m_nIterations; }
	int GetInitSPP() { return m_sppInit; }

	// maximum spp for this step
	int m_maxSPP;
	
	//
	int m_startIdx;
	int m_endIdx;

private:
     // Film attributes
    int m_xPixelCount, m_yPixelCount;
    
    // BandwidthSampler private attributes
    int m_nIterations;

    LWR_Film *m_myFilm;

	float* m_sampleBuf;

    // Attributes for initialization phase
    int m_xPos, m_yPos;
    int m_sppInit;

    int m_xStartSub;
	int m_xEndSub;
	int m_yStartSub;
	int m_yEndSub;
    
    void init(const LWR_Sampler * parent, const int minsamples);    

	/////////////////////
	inline void LDShuffleScrambled1D(int nSamples, int nPixel, float *samples, RNG &rng, int nGenerated, uint32_t scramble) {		
		for (int i = 0; i < nSamples * nPixel; ++i)	
			samples[i] = VanDerCorput(i + nGenerated * nSamples, scramble);			
		for (int i = 0; i < nPixel; ++i)
			Shuffle(samples + i * nSamples, nSamples, 1, rng);
		Shuffle(samples, nPixel, nSamples, rng);
	}
	inline void LDShuffleScrambled2D(int nSamples, int nPixel, float *samples, RNG &rng, int nGenerated, uint32_t* scramble) {		
		for (int i = 0; i < nSamples * nPixel; ++i)
			Sample02(i + nGenerated * nSamples, scramble, &samples[2*i]);			
		for (int i = 0; i < nPixel; ++i)
			Shuffle(samples + 2 * i * nSamples, nSamples, 2, rng);
		Shuffle(samples, nPixel, 2 * nSamples, rng);
	}
    void LDPixelSample(int xPos, int yPos, float shutterOpen, float shutterClose, int nPixelSamples, std::vector<float> *samples, float *buf, RNG &rng, PixInfo& pixInfo) {
		// Prepare temporary array pointers for low-discrepancy camera samples
		float *imageSamples = buf; buf += 2 * nPixelSamples;
//		float *lensSamples = buf;  buf += 2 * nPixelSamples;
//		float *timeSamples = buf;  buf += nPixelSamples;

		// Prepare temporary array pointers for low-discrepancy integrator samples
//		uint32_t count1D = samples[0].n1D.size();
//		uint32_t count2D = samples[0].n2D.size();
//		const uint32_t *n1D = count1D > 0 ? &samples[0].n1D[0] : NULL;
//		const uint32_t *n2D = count2D > 0 ? &samples[0].n2D[0] : NULL;
//		float **oneDSamples = ALLOCA(float *, count1D);
//		float **twoDSamples = ALLOCA(float *, count2D);
//		for (uint32_t i = 0; i < count1D; ++i) {
//			oneDSamples[i] = buf;
//			buf += n1D[i] * nPixelSamples;
//		}
//		for (uint32_t i = 0; i < count2D; ++i) {
//			twoDSamples[i] = buf;
//			buf += 2 * n2D[i] * nPixelSamples;
//		}

		if (pixInfo.m_numGenerated == 0) {
			pixInfo.scramble_image[0] = rng.RandomUInt();
			pixInfo.scramble_image[1] = rng.RandomUInt();
//			pixInfo.scramble_lens[0] = rng.RandomUInt();
//			pixInfo.scramble_lens[1] = rng.RandomUInt();
//			pixInfo.scramble_time = rng.RandomUInt();

//			pixInfo.scramble_1D.resize(count1D);
//			pixInfo.scramble_2D.resize(count2D * 2);
//			for (unsigned int i = 0; i < count1D; ++i)
//				pixInfo.scramble_1D[i] = rng.RandomUInt();
//			for (unsigned int i = 0; i < count2D * 2; ++i)
//				pixInfo.scramble_2D[i] = rng.RandomUInt();
		}	

		// Generate low-discrepancy pixel samples
		LDShuffleScrambled2D(1, nPixelSamples, imageSamples, rng, pixInfo.m_numGenerated, pixInfo.scramble_image);
//		LDShuffleScrambled2D(1, nPixelSamples, lensSamples, rng, pixInfo.m_numGenerated, pixInfo.scramble_lens);
//		LDShuffleScrambled1D(1, nPixelSamples, timeSamples, rng, pixInfo.m_numGenerated, pixInfo.scramble_time);
//		for (uint32_t i = 0; i < count1D; ++i)
//			LDShuffleScrambled1D(n1D[i], nPixelSamples, oneDSamples[i], rng, pixInfo.m_numGenerated, pixInfo.scramble_1D[i]);
//		for (uint32_t i = 0; i < count2D; ++i)
//			LDShuffleScrambled2D(n2D[i], nPixelSamples, twoDSamples[i], rng, pixInfo.m_numGenerated, &pixInfo.scramble_2D[i * 2]);

		// Initialize _samples_ with computed sample values
		for (int i = 0; i < nPixelSamples; ++i) {
            samples->push_back(xPos + imageSamples[2*i]);
            samples->push_back(yPos + imageSamples[2*i+1]);
//			samples[i].imageX = xPos + imageSamples[2*i];
//			samples[i].imageY = yPos + imageSamples[2*i+1];
//			samples[i].time = Lerp(timeSamples[i], shutterOpen, shutterClose);
//			samples[i].lensU = lensSamples[2*i];
//			samples[i].lensV = lensSamples[2*i+1];
//			// Copy integrator samples into _samples[i]_
//			for (uint32_t j = 0; j < count1D; ++j) {
//				int startSamp = n1D[j] * i;
//				for (uint32_t k = 0; k < n1D[j]; ++k)
//					samples[i].oneD[j][k] = oneDSamples[j][startSamp+k];
//			}
//			for (uint32_t j = 0; j < count2D; ++j) {
//				int startSamp = 2 * n2D[j] * i;
//				for (uint32_t k = 0; k < 2*n2D[j]; ++k)
//					samples[i].twoD[j][k] = twoDSamples[j][startSamp+k];
//			}
		}

		pixInfo.m_numGenerated += nPixelSamples;
	}

	// Random Sampling
    void RandomPixelSample(int xPos, int yPos, float shutterOpen, float shutterClose, int nPixelSamples, float *samples, float *buf, RNG &rng, PixInfo& pixInfo) {
//		uint32_t count1D = samples[0].n1D.size();
//		uint32_t count2D = samples[0].n2D.size();
//		const uint32_t *n1D = count1D > 0 ? &samples[0].n1D[0] : NULL;
//		const uint32_t *n2D = count2D > 0 ? &samples[0].n2D[0] : NULL;

		// Initialize _samples_ with computed sample values
		for (int i = 0; i < nPixelSamples; ++i) {
            samples[i*SAMPLE_SIZE + IMAGE_X] = xPos + rng.RandomFloat();
            samples[i*SAMPLE_SIZE + IMAGE_Y] = yPos + rng.RandomFloat();
//			samples[i].imageX = xPos + rng.RandomFloat();
//			samples[i].imageY = yPos + rng.RandomFloat();
//			samples[i].time = Lerp(rng.RandomFloat(), shutterOpen, shutterClose);
//			samples[i].lensU = rng.RandomFloat();
//			samples[i].lensV = rng.RandomFloat();
//			// Copy integrator samples into _samples[i]_
//			for (uint32_t j = 0; j < count1D; ++j) {
//				int startSamp = n1D[j] * i;
//				for (uint32_t k = 0; k < n1D[j]; ++k)
//					samples[i].oneD[j][k] = rng.RandomFloat();
//			}
//			for (uint32_t j = 0; j < count2D; ++j) {
//				int startSamp = 2 * n2D[j] * i;
//				for (uint32_t k = 0; k < 2*n2D[j]; ++k)
//					samples[i].twoD[j][k] = rng.RandomFloat();
//			}
		}
		pixInfo.m_numGenerated += nPixelSamples;
	}
};

LWR_Sampler *CreateLWRSampler(const ParamSet &params, Film *film, const Camera *camera);
