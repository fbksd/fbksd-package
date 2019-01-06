
Code and data needed to reproduce the results of our paper:

Robust Denoising using Feature and Color Information.

Our implementation is based on the PBRT raytracer of Pharr and Humphreys:

http://www.pbrt.org/

Additional questions are welcome:
roussell@aim.unibe.ch

--------------------------------------------------------------------------------
Directory "pbrt-v2-dfc":

********************************** IMPORTANT ***********************************
The Makefile in the "src" subdirectory is functional. The other configurations
(src/pbrt.vs2008, src/pbrt.vs2010, src/pbrt.vs2012, pbrt.xcodeproj, and SCons)
were neither updated, nor tested.
You will need to first install the CUDA software:
https://developer.nvidia.com/cuda-downloads
********************************************************************************

We modified multiple files from the original PBRT framework:
"src/Makefile"
"src/core/api.cpp"
"src/core/film.h"
"src/core/integrator.h"
"src/core/integrator.cpp"
"src/core/reflection.h"
"src/core/reflection.cpp"
"src/integrators/directlighting.h"
"src/integrators/directlighting.cpp"
"src/integrators/path.h"
"src/integrators/path.cpp"
"src/integrators/photonmap.h"
"src/integrators/photonmap.cpp"
"src/core/sampler.h"
"src/samplers/random.cpp"

The changes to "api.cpp" allow the parsing of our own sampler ("multisampler"),
renderer ("twostages"), and film ("multifilm") classes. The change in "Makefile"
is to compile CUDA files. CUDA is mandatory since we do not provide a CPU
implementation of our non-local means filter variant. Changes to "integrator",
"reflection", "directlighting", "path", and "photonmap" are to enable the
extraction of feature data. Changes to "sampler" and "random" are to explicitely
store the pixel coordinates of each sample.

We added the following files:
"src/core/featurefilter.h"
"src/core/featurefilter.cu"
"src/samplers/multisampler.h"
"src/samplers/multisampler.cpp"
"src/renderers/twostages.h"
"src/renderers/twostages.cpp"
"src/film/multifilm.h"
"src/film/multifilm.cpp"

In theory, one should be able to download the latest version of PBRTv2, and add
our new files, as well as the modified files, to get a functional version of our
implementation. This is how the present code distribution was obtained.

The main entry point to our algorithm is the call:
"MultiFilm::GetSamplingMap"
in "src/film/multifilm.cpp". We suggest tracing from that point to get a
better understanding of the implementation. Here is some high-level information
about each class:

featurefilter: CUDA implementation of our filter. Takes a color buffer, as well
as a set of feature buffers, as input. Can return both the filtered output, as
well as its derivative (needed to perform the SURE computation).

multisampler: a sampler with two distinct stages (init and adaptive). During the
adaptive phase, it uses a sampling map which specifies how many samples should
be drawn per pixel. Sampling is done using low-discrepancy sampling, and we keep
two sets of scrambling data (one per buffer).

twostages: a modified version of PBRT's "sampler" renderer, with specific code
to handle our "multisampler" sampler. When sending samples to the film, our
renderer specifies the target buffer, and passes additional feature data
(normal, position, visibility, etc.).

multifilm: a modified version of PBRT's "image" film using two or more buffers.
This class stores the additional information we need (sample count,
variance-related statistics, feature buffers). The "multifilm" film holds an
instance of the "featurefilter" which is used to perform the actual filtering.

As we extend PBRT's framework, we inherit its scene description system with some
additions. As an example, here is an exerpt from our "pg2013_sibenik.pbrt" scene
showing the options specific to our implementation:

Film "multi"
    "integer wnd_rad"     [10]
    "integer xresolution" [1024] "integer yresolution" [1024]
Sampler "multi" "integer pixelsamples" [16]  "integer niterations" [0]
PixelFilter "box" "float xwidth" [0.5] "float ywidth" [0.5]
Renderer "twostages"

The Film specification instantiates our "multi" film, which will itself
set the "featurefilter" to use a window radius ("wnd_rad") value of 10. The
Sampler specification instantiates our "multisampler" sampler, and requests 16
samples per pixel on average, and 0 adaptive sampling iterations. Note that when
"niterations" is set to 0, our sampler produces a uniform sampling distribution.
The sampler splits the sampling budget evenly across all iterations. For
instance, when using 2 adaptive iterations, the initial sampling pass and the
two adaptive passes each use 1/3 of the sampling budget. We use a "box"
PixelFilter, which is the only type of filter currently supported by our
implementation. Lastly, the Renderer specification instantiates our "twostages"
renderer, which has no parameters.

We also provide the scene description files of all scenes presented in our paper
in the directory "pg2013-scenes", along with all data (models, textures, brdfs)
needed. To render the "conference" scene, one would simply cd to the
"pg2013-scenes" directory in a terminal and type:
../src/bin/pbrt pg2013_conference.pbrt

The program outputs the following files:
conference.exr (the usual PBRT output)
conference_sel.exr (selection map used to weight the three candidate filters)
conference_spp.exr (sample density map)
conference_int.exr (our intermediate filtered output, after the first pass)
conference_flt.exr (our final filtered output, after the second pass)

For instructions on how to compile PBRT, please refer to PBRT's documentation.

--------------------------------------------------------------------------------

The results in the paper and in our supplemental materials used a simple gamma
correction. Here are the gamma values used for each scene:

conference:  2.2
killines:    2.2
sanmiguel20: 2.2
toasters:    2.2
dragonfog:   1.5
plants-dusk: 1.0 (no gamma correction, but 4x scaling to make it brighter)
sibenik:     2.2
yeahright:   1.5


