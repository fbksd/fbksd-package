
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

// renderers/sbfrenderer.cpp*
//#include "stdafx.h"
#include "sbfrenderer.h"
#include "scene.h"
#include "film.h"
#include "volume.h"
#include "sampler.h"
#include "integrator.h"
#include "progressreporter.h"
#include "camera.h"
#include "intersection.h"
#include "samplers/sbfsampler.h"
#include "film/sbfimage.h"
#include "imageio.h"

// SBFRendererTask Definitions
void SBFRendererTask::Run() {
    PBRT_STARTED_RENDERTASK(taskNum);

    // Get sub-_Sampler_ for _SamplerRendererTask_
    Sampler *sampler = mainSampler->GetSubSampler(taskNum, taskCount);
    if (!sampler)
    {
        reporter.Update();
        PBRT_FINISHED_RENDERTASK(taskNum);
        return;
    }

    // Declare local variables used for rendering loop
    MemoryArena arena;

    // Allocate space for samples and intersections
    int maxSamples = sampler->MaximumSampleCount();
    Sample *samples = origSample->Duplicate(maxSamples);
    RayDifferential *rays = new RayDifferential[maxSamples];
    Spectrum *Ls = new Spectrum[maxSamples];
    Spectrum *Ts = new Spectrum[maxSamples];
    Intersection *isects = new Intersection[maxSamples];

    // Get samples from _Sampler_ and update image
    int sampleCount;
    while ((sampleCount = sampler->GetMoreSamples(samples, rng)) > 0) {
        // Generate camera rays and compute radiance along rays
        for (int i = 0; i < sampleCount; ++i) {
            // Reset intersection information
            isects[i] = Intersection();

            // Find camera ray for _sample[i]_
            PBRT_STARTED_GENERATING_CAMERA_RAY(&samples[i]);
            float rayWeight = camera->GenerateRayDifferential(samples[i], &rays[i]);
            rays[i].ScaleDifferentials(1.f / sqrtf((float)sampler->samplesPerPixel));
            PBRT_FINISHED_GENERATING_CAMERA_RAY(&samples[i], &rays[i], rayWeight);

            // Evaluate radiance along camera ray
            PBRT_STARTED_CAMERA_RAY_INTEGRATION(&rays[i], &samples[i]);
            if (rayWeight > 0.f)
                Ls[i] = rayWeight * renderer->Li(scene, rays[i], &samples[i], rng,
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
        if (sampler->ReportResults(samples, rays, Ls, isects, sampleCount))
        {
            for (int i = 0; i < sampleCount; ++i)
            {
                PBRT_STARTED_ADDING_IMAGE_SAMPLE(&samples[i], &rays[i], &Ls[i], &Ts[i]);
                isects[i].shadingN = Faceforward(isects[i].shadingN, rays[i].d);
                isects[i].depth = min(rays[i].maxt, maxDepth)/maxDepth;
                camera->film->AddSample(samples[i], Ls[i], isects[i]);
                PBRT_FINISHED_ADDING_IMAGE_SAMPLE();
            }
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
SBFRenderer::SBFRenderer(Sampler *s, Camera *c,
                       SurfaceIntegrator *si, VolumeIntegrator *vi) {
    sampler = s;
    camera = c;
    surfaceIntegrator = si;
    volumeIntegrator = vi;
}


SBFRenderer::~SBFRenderer() {
    delete sampler;
    delete camera;
    delete surfaceIntegrator;
    delete volumeIntegrator;
}


void SBFRenderer::Render(const Scene *scene) {
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

    // Compute number of _SamplerRendererTask_s to create for rendering
    int nPixels = camera->film->xResolution * camera->film->yResolution;
    int nTasks = max(32 * NumSystemCores(), nPixels / (16*16));
    nTasks = RoundUpPow2(nTasks);

    // TODO: We should do some checking mechanism since the renderer will crash
    // if we're not using SBFImageFilm and SBFSampler...
    SBFImageFilm *sbfFilm = dynamic_cast<SBFImageFilm*>(camera->film);    
    SBFSampler *sbfSampler = dynamic_cast<SBFSampler*>(sampler);        
    
    int xs, ys, xe, ye;
    camera->film->GetPixelExtent(&xs, &xe, &ys, &ye);
    float maxDepth = 0.f;
    for(int y = ys; y < ye; y++)
        for(int x = xs; x < xe; x++) {
            CameraSample smp;
            smp.imageX = x+.5f;
            smp.imageY = y+.5f;
            smp.lensU = smp.lensV = smp.time = 0.f;
            Ray ray;
            camera->GenerateRay(smp, &ray);
            Intersection isect;
            scene->Intersect(ray, &isect);
            if(!isinf(ray.maxt))
                maxDepth = max(maxDepth, ray.maxt);
        }

    ProgressReporter reporter(nTasks, "Initial Sampling");    
    vector<Task *> renderTasks;
    vector<RNG> rngs;
    for (int i = 0; i < nTasks; ++i) {
        rngs.push_back(RNG(nTasks-1-i));
    }

    for (int i = 0; i < nTasks; ++i) {
        renderTasks.push_back(new SBFRendererTask(scene, this, camera,
                                                  reporter, sampler, sample, 
                                                  nTasks-1-i, nTasks,
                                                  rngs[i], maxDepth));
    }
    EnqueueTasks(renderTasks);
    WaitForAllTasks();
    for (uint32_t i = 0; i < renderTasks.size(); ++i)
        delete renderTasks[i];  
    reporter.Done();        
    
    if(sbfSampler->GetAdaptiveSPP() > 0.f && sbfSampler->GetIteration() > 0) {
        for(int iter = 0; iter < sbfSampler->GetIteration(); iter++) {
            vector<vector<int> > pixOff;
            vector<vector<int> > pixSmp;
            sbfFilm->GetAdaptPixels(sbfSampler->GetAdaptiveSPP(), pixOff, pixSmp);
            sbfSampler->SetPixelOffset(&pixOff);
            sbfSampler->SetPixelSampleCount(&pixSmp);

            ProgressReporter asReporter(nTasks, "Adaptive Sampling");
            renderTasks.clear();
            for (int i = 0; i < nTasks; ++i)
                renderTasks.push_back(new SBFRendererTask(scene, this, camera,
                            asReporter, sampler, sample, nTasks-1-i, nTasks,
                            rngs[i], maxDepth));
            EnqueueTasks(renderTasks);
            WaitForAllTasks();
            for (uint32_t i = 0; i < renderTasks.size(); ++i)
                delete renderTasks[i];  
            asReporter.Done();
        }
    }
    

    PBRT_FINISHED_RENDERING();
    // Clean up after rendering and store final image
    delete sample;

    camera->film->WriteImage();
}


Spectrum SBFRenderer::Li(const Scene *scene,
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
    if (scene->Intersect(ray, isect))
        Li = surfaceIntegrator->Li(scene, this, ray, *isect, sample,
                                   rng, arena);
    else {
        // Handle ray that doesn't intersect any geometry
        for (uint32_t i = 0; i < scene->lights.size(); ++i)
           Li += scene->lights[i]->Le(ray);
    }
    Spectrum Lvi = volumeIntegrator->Li(scene, this, ray, sample, rng,
                                        T, arena);
    return *T * Li + Lvi;
}


Spectrum SBFRenderer::Transmittance(const Scene *scene,
        const RayDifferential &ray, const Sample *sample, RNG &rng,
        MemoryArena &arena) const {
    return volumeIntegrator->Transmittance(scene, this, ray, sample,
                                           rng, arena);
}


