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

// renderers/samplerrenderer.cpp*
#include "stdafx.h"
#include "scene.h"
#include "film.h"
#include "volume.h"
#include "sampler.h"
#include "integrator.h"
#include "progressreporter.h"
#include "camera.h"
#include "intersection.h"
#include "rng.h"

#include "lwrr_test/lwr_renderer.h"
#include "lwrr_test/lwr_sampler.h"
#include <time.h>
#include "omp.h"

static uint32_t hash(char *key, uint32_t len)
{
    uint32_t   hash, i;
    for (hash=0, i=0; i<len; ++i) {
        hash += key[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
} 

// SamplerRendererTask Definitions
void LWR_RendererTask::Run() {
    PBRT_STARTED_RENDERTASK(taskNum);
    // Get sub-_Sampler_ for _SamplerRendererTask_
    Sampler *sampler = mainSampler->GetSubSampler(taskNum, taskCount);		

    if (!sampler)
    {
        reporter.Update();
        PBRT_FINISHED_RENDERTASK(taskNum);
        return;
    }

	LWR_Sampler *mySampler = dynamic_cast<LWR_Sampler *> (sampler);
	LWR_Film *myFilm = dynamic_cast<LWR_Film *> (camera->film);		

	int spp = myFilm->m_maxSPP;
	mySampler->m_maxSPP = spp;
	float maxDepth = myFilm->m_maxDepth;
	float avgSample = mySampler->samplesPerPixel;	
	//

    // Declare local variables used for rendering loop
    MemoryArena arena;

    // Allocate space for samples and intersections
    Sample *samples = origSample->Duplicate(spp);
    RayDifferential *rays = new RayDifferential[spp];
    Spectrum *Ls = new Spectrum[spp];
    Spectrum *Ts = new Spectrum[spp];
    Intersection *isects = new Intersection[spp];
	
    // Get samples from _Sampler_ and update image
    int sampleCount;
	int idxPix;
	while ((sampleCount = mySampler->GetMoreSamplesWithIdx(samples, *m_rng, idxPix)) > 0) {	
        // Generate camera rays and compute radiance along rays
        for (int i = 0; i < sampleCount; ++i) {
			 // Reset intersection information
            isects[i] = Intersection();

            // Find camera ray for _sample[i]_
            PBRT_STARTED_GENERATING_CAMERA_RAY(&samples[i]);
            float rayWeight = camera->GenerateRayDifferential(samples[i], &rays[i]);   

			// 
			if (myFilm->m_rayScale > 0.f)
				rays[i].ScaleDifferentials(myFilm->m_rayScale);
			else
				rays[i].ScaleDifferentials(1.f / sqrtf(avgSample));
			//

            PBRT_FINISHED_GENERATING_CAMERA_RAY(&samples[i], &rays[i], rayWeight);

            // Evaluate radiance along camera ray
            PBRT_STARTED_CAMERA_RAY_INTEGRATION(&rays[i], &samples[i]);
			if (rayWeight > 0.f) 
				Ls[i] = rayWeight * renderer->Li(scene, rays[i], &samples[i], *m_rng,
				arena, &isects[i], &Ts[i]);
			else {
				Ls[i] = 0.f;
				Ts[i] = 1.f;
			}

			// Issue warning if unexpected radiance value returned
			if (Ls[i].HasNaNs()) {
				Error("Not-a-number radiance value returned "
					"for image sample.  Setting to black.");
				Ls[i] = Spectrum(0.f);
			}
			else if (Ls[i].y() < -1e-5) {
				Error("Negative luminance value, %f, returned"
					"for image sample.  Setting to black.", Ls[i].y());
				Ls[i] = Spectrum(0.f);
			}
			else if (isinf(Ls[i].y())) {
				Error("Infinite luminance value returned"
					"for image sample.  Setting to black.");
				Ls[i] = Spectrum(0.f);
			}			
			PBRT_FINISHED_CAMERA_RAY_INTEGRATION(&rays[i], &samples[i], &Ls[i]);
        }

        // Report sample results to _Sampler_, add contributions to image
		for (int i = 0; i < sampleCount; ++i) {
			PBRT_STARTED_ADDING_IMAGE_SAMPLE(&samples[i], &rays[i], &Ls[i], &Ts[i]);	
			
			if (rays[i].maxt > maxDepth * 10.f || isects[i].isLightHit) {
				// no hit
				isects[i].depth = 10.f;
			}
			else
				isects[i].depth = rays[i].maxt / maxDepth;		

			isects[i].shadingN = Faceforward(isects[i].shadingN, rays[i].d);						
			Spectrum tex;
			if (isects[i].material)
				isects[i].material->GetKd(isects[i].dg, tex);
			isects[i].rho += tex;

			myFilm->AddSampleExtended(samples[i], Ls[i], isects[i], idxPix);		

			PBRT_FINISHED_ADDING_IMAGE_SAMPLE();
		}
        // Free _MemoryArena_ memory from computing image sample values
        arena.FreeAll();
    }

    // Clean up after _SamplerRendererTask_ is done with its image region
    camera->film->UpdateDisplay(sampler->xPixelStart,
        sampler->yPixelStart, sampler->xPixelEnd+1, sampler->yPixelEnd+1);
    delete sampler;
    delete[] samples;
    delete[] rays;
    delete[] Ls;
    delete[] Ts;
    delete[] isects;
    reporter.Update();
    PBRT_FINISHED_RENDERTASK(taskNum);
}



// SamplerRenderer Method Definitions
LWR_Renderer::LWR_Renderer(Sampler *s, Camera *c,
                                 SurfaceIntegrator *si, VolumeIntegrator *vi,
                                 bool visIds) {
    sampler = s;
    camera = c;
    surfaceIntegrator = si;
    volumeIntegrator = vi;
    visualizeObjectIds = visIds;
}


LWR_Renderer::~LWR_Renderer() {
    delete sampler;
    delete camera;
    delete surfaceIntegrator;
    delete volumeIntegrator;
}

void LWR_Renderer::Render(const Scene *scene) {
    PBRT_FINISHED_PARSING();
    // Allow integrators to do preprocessing for the scene
    PBRT_STARTED_PREPROCESSING();
    surfaceIntegrator->Preprocess(scene, camera, this);
    volumeIntegrator->Preprocess(scene, camera, this);
    PBRT_FINISHED_PREPROCESSING();

    PBRT_STARTED_RENDERING();

    // Allocate and initialize _sample_
    Sample *sample = new Sample(sampler, surfaceIntegrator,
                                volumeIntegrator, scene);

    // Create and launch _SamplerRendererTask_s for rendering image
	LWR_Sampler *mySampler = dynamic_cast<LWR_Sampler *> (sampler);
	LWR_Film *myFilm = dynamic_cast<LWR_Film *> (camera->film);	
	myFilm->initializeGlobalVariables(mySampler->GetInitSPP());

	// 
	uint32_t count1D = sample[0].n1D.size();		
	uint32_t count2D = sample[0].n2D.size();
	myFilm->generateScramblingInfo(count1D, count2D);

	// Compute number of _SamplerRendererTask_s to create for rendering
	int nPixels = camera->film->xResolution * camera->film->yResolution;
	int nTasks = max(32 * NumSystemCores(), nPixels / (16*16));
	nTasks = RoundUpPow2(nTasks);

	// Compute Max Depth
	float maxDepth = 0.f;
	for(int y = 0; y < camera->film->yResolution; y += 2) {
		for(int x = 0; x < camera->film->xResolution; x += 2) {
			CameraSample cam_smp;			
			cam_smp.imageX = x+.5f;
			cam_smp.imageY = y+.5f;
			cam_smp.lensU = cam_smp.lensV = cam_smp.time = 0.f;
			Ray ray;
			camera->GenerateRay(cam_smp, &ray);
			Intersection isect;
			if(scene->Intersect(ray, &isect))
				maxDepth = max(maxDepth, ray.maxt);			
		}
	}
	myFilm->m_maxDepth = maxDepth;
	myFilm->m_samplesPerPixel = mySampler->samplesPerPixel;
	printf("\nInfo. Max Depth = %.1f, target spp = %d\n", maxDepth, mySampler->samplesPerPixel);
	
	//
	vector<RNG> rng_list;
	rng_list.reserve(nTasks);
	for (int i = 0; i < nTasks; ++i) 
		rng_list.push_back( RNG(nTasks-1-i) );					

	ProgressReporter reporter(nTasks, "Rendering");

	vector<Task *> renderTasks;
	
	for (int i = 0; i < nTasks; ++i)
		renderTasks.push_back(new LWR_RendererTask(scene, this, camera,
		reporter, sampler, sample, 
		visualizeObjectIds, 
		nTasks-1-i, nTasks, &rng_list[nTasks-1-i], mySampler->GetInitSPP()));
	EnqueueTasks(renderTasks);
	WaitForAllTasks();
	for (uint32_t i = 0; i < renderTasks.size(); ++i)
		delete renderTasks[i];
	renderTasks.clear();
	reporter.Done();

	int nIterations = mySampler->GetIterationCount();
	int numSamplePerIterations = (mySampler->samplesPerPixel - mySampler->GetInitSPP()) * nPixels / nIterations;
	int nTasksTotal = nTasks * nIterations;

	if (numSamplePerIterations > 0) {
		// Set the progress reporter
		ProgressReporter reporterAdapt(nTasksTotal, "Adaptive Rendering");
		for (int iter = 0; iter < nIterations; ++iter) {
			myFilm->test_lwrr(numSamplePerIterations);		

			for (int i = 0; i < nTasks; ++i) {
				renderTasks.push_back(new LWR_RendererTask(
					scene, this, camera, reporterAdapt, sampler, sample,
					visualizeObjectIds, nTasks-1-i, nTasks, &rng_list[nTasks-1-i], 0));
			}
			EnqueueTasks(renderTasks);
			WaitForAllTasks();

			for (uint32_t i = 0; i < renderTasks.size(); ++i)
				delete renderTasks[i];
			renderTasks.clear();			
		}		
		reporterAdapt.Done();
	}

    PBRT_FINISHED_RENDERING();
    // Clean up after rendering and store final image
    delete sample;
    camera->film->WriteImage();
}


Spectrum LWR_Renderer::Li(const Scene *scene,
        const RayDifferential &ray, const Sample *sample, RNG &rng,
        MemoryArena &arena, Intersection *isect, Spectrum *T) const {
    Assert(ray.time == sample->time);
    Assert(!ray.HasNaNs());
    // Allocate local variables for _isect_ and _T_ if needed
    Spectrum localT;
    if (!T) T = &localT;
    Intersection localIsect;
    if (!isect) isect = &localIsect;
    Spectrum Li = 0.f;
	
    if (scene->Intersect(ray, isect)) {
        Li = surfaceIntegrator->Li(scene, this, ray, *isect, sample,
                                   rng, arena);
	}
    else {
        // Handle ray that doesn't intersect any geometry
        for (uint32_t i = 0; i < scene->lights.size(); ++i)
           Li += scene->lights[i]->Le(ray);

		// BC -- Save background values or lights as textures
		isect->rho = Li;
    }

    Spectrum Lvi = volumeIntegrator->Li(scene, this, ray, sample, rng,
                                        T, arena);

    return *T * Li + Lvi;
}


Spectrum LWR_Renderer::Transmittance(const Scene *scene,
        const RayDifferential &ray, const Sample *sample, RNG &rng,
        MemoryArena &arena) const {
    return volumeIntegrator->Transmittance(scene, this, ray, sample,
                                           rng, arena);
}



