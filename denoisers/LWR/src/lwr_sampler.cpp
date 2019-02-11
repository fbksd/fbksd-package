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

#include "stdafx.h"
#include "montecarlo.h"

#include "lwr_film.h"
#include "lwr_sampler.h"

#include <algorithm>
using namespace std;

LWR_Sampler::LWR_Sampler(int xstart, int xend, int ystart, int yend, int minsamples, int spp, float sopen, float sclose, int nIterations, LWR_Film* film)  
	                : Sampler(xstart, xend, ystart, yend, spp, sopen, sclose),
				      m_xPixelCount(film->getXPixelCount()),
					  m_yPixelCount(film->getYPixelCount()),
					  m_nIterations(nIterations),
					  m_myFilm(film)
{
	init(NULL, minsamples);

	m_sampleBuf = NULL;
}

LWR_Sampler::LWR_Sampler(const LWR_Sampler *parent, int xstart, int xend, int ystart, int yend)  
	                 :  Sampler(parent->xPixelStart, parent->xPixelEnd, parent->yPixelStart,
								parent->yPixelEnd, parent->samplesPerPixel, parent->shutterOpen,
								parent->shutterClose),
						m_xPixelCount(parent->m_xPixelCount),
						m_yPixelCount(parent->m_yPixelCount),
						m_nIterations(parent->m_nIterations),
						m_myFilm(parent->m_myFilm)
{

	// avoid -1
	xstart = max(0, xstart);
	ystart = max(0, ystart);
	xend = min(m_xPixelCount, xend);
	yend = min(m_yPixelCount, yend);
	//

    m_xPos = xstart;
    m_yPos = ystart;
	m_xStartSub = xstart;	
	m_yStartSub = ystart;
	m_xEndSub = xend;
	m_yEndSub = yend;

	init(parent, parent->m_sppInit);

	m_sampleBuf = NULL;
}

LWR_Sampler::~LWR_Sampler(void)
{
	if (m_sampleBuf)
		delete[] m_sampleBuf;
}

void LWR_Sampler::init(const LWR_Sampler* parent, const int minsamples)
{
	m_sppInit = minsamples;
	m_maxSPP = minsamples;

	if (parent != NULL) {
		m_xPos = m_xStartSub;
		m_yPos = m_yStartSub;
		m_maxSPP = parent->m_maxSPP;
	}
	else {
		m_xPos = xPixelStart;
		m_yPos = yPixelStart;
	}
}

Sampler *LWR_Sampler::GetSubSampler(int num, int count) {
    int x0, x1, y0, y1;
    ComputeSubWindow(num, count, &x0, &x1, &y0, &y1);
    if (x0 == x1 || y0 == y1) return NULL;
    return new LWR_Sampler(this, x0, x1, y0, y1);
}

int LWR_Sampler::GetMoreSamplesWithIdx(std::vector<float> *samples, RNG &rng, int& pixIdx)
{
	const PixSPP* pPixel = NULL; 
	int idxPixel = -1;
	{
		MutexLock lock(*m_myFilm->m_globalLock);
		idxPixel = m_myFilm->getPixelReference();	
	}
	if (idxPixel >= m_myFilm->m_pixelErrors.size()) 
		return 0;
	pPixel = &(m_myFilm->m_pixelErrors[idxPixel]);
	
	int spp = pPixel->m_numSample;
	PixInfo& pixInfo = m_myFilm->m_pixInfo[pPixel->m_idx];
	pixIdx = pPixel->m_idx;	
	int xPos = pPixel->m_idx % m_myFilm->xResolution;
    int yPos = pPixel->m_idx / m_myFilm->xResolution;  	    
	
#if MY_SAMPLER == LD_SAMPLER
	if (!m_sampleBuf)
		m_sampleBuf = new float[LDPixelSampleFloatsNeeded(nullptr, m_maxSPP)];	
    LDPixelSample(xPos, yPos, shutterOpen, shutterClose, spp, samples, m_sampleBuf, rng, pixInfo);
#elif MY_SAMPLER == RANDOM_SAMPLER
	RandomPixelSample(xPos, yPos, shutterOpen, shutterClose, spp, samples, m_sampleBuf, rng, pixInfo);	
#endif

	return spp;	
}

bool LWR_Sampler::ReportResults(Sample *samples, const RayDifferential *rays, const Spectrum *Ls, const Intersection *isects, int count) {
	return true;
}


//LWR_Sampler *CreateLWRSampler(const ParamSet &params, Film *film, const Camera *camera) {

//	// By default we update 5% of the image on each iteration
//	int minsamples = params.FindOneInt("minsamples", 4);
//	int ns = params.FindOneInt("pixelsamples", 32);
//    int nIterations = params.FindOneInt("niterations", 8);

//    // Initialize common sampler parameters
//    int xstart, xend, ystart, yend;
//    film->GetSampleExtent(&xstart, &xend, &ystart, &yend);
	
//	LWR_Film* myFilm = dynamic_cast<LWR_Film*> (film);

//    if (myFilm == NULL) {
//        Error("CreateBandwidthSampler(): film is not of type 'MyFilm'");
//        return NULL;
//    }

//	// Output the sampler parameters
//    Info("CreateMySampler:\n");
//	Info("   minsamples.....: %d\n", minsamples);
//    Info("   pixelsamples.....: %d\n", ns);
//    Info("   niterations......: %d\n", nIterations);
//    return new LWR_Sampler(xstart, xend, ystart, yend, minsamples, ns,
//						   camera->shutterOpen, camera->shutterClose, nIterations, myFilm);
//}



