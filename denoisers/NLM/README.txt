
Code and data needed to reproduce the results of our paper:

Adaptive rendering with non-local means filtering.

Our implementation is based on the PBRT raytracer of Pharr and Humphreys:

http://www.pbrt.org/

Additional questions are welcome:
roussell@aim.unibe.ch

--------------------------------------------------------------------------------
Directory "pbrt-v2-nlm":

********************************** IMPORTANT ***********************************
The Makefile in the "src" subdirectory is functional. The other configurations
(src/pbrt.vs2008, src/pbrt.vs2010, and pbrt.xcodeproj) were neither updated,
nor tested.
You will need to first install the CUDA software:
https://developer.nvidia.com/cuda-downloads
********************************************************************************

We only modified two files from the original PBRT framework:
"src/core/api.cpp"
"src/Makefile"

The changes to "api.cpp" allow the parsing of our own sampler ("dualsampler"),
renderer ("twostages"), and film ("dualfilm") classes. The change in "Makefile"
is to enable OpenMP support and compile CUDA files. If OpenMP is unavailable,
our own processing will not be parallelized. CUDA is mandatory since we do not
provide a CPU implementation of our non-local means filter variant.

We added the following files:
"src/core/kernel2d.h" and "src/core/kernel2d.cpp"
"src/core/nlmkernel.h" and "src/core/nlmkernel.cu"
"src/denoisers/nlmdenoiser.h" and "src/denoisers/nlmdenoiser.cpp"
"src/samplers/dualsampler.h" and "src/samplers/dualsampler.cpp"
"src/renderers/twostages.h" and "src/renderers/twostages.cpp"
"src/film/dualfilm.h" and "src/film/dualfilm.cpp"

In theory, one should be able to download the latest version of PBRTv2, and add
our new files, as well as the two modified files, to get a functional version
of our implementation. This is how the present code distribution was obtained.

The main entry point to our algorithm is the call:
"dualSampler->GetSamplingMaps(nPixelsPerIteration);"
in "src/renderers/twostages.cpp". We suggest tracing from that point to get a
better understanding of the implementation. Here is some high-level information
about each class:

kernel2d: a simple utility class performing the filtering specifically tailored
to our needs.

nlmkernel: CUDA implementation of our non-local means filter variant. Comment
out the line "#define LD_SAMPLING" in "core/nlmkernel.h" to use pure random
sampling instead of low-discrepancy sampling.

nlmdenoiser: this preprocesses the subpixel data according to the pixel filter
specified in the scene description. It then applies our non-local means filter
variant, estimates the residual error, and generates the sampling map that will
be used by the sampler.

dualsampler: a sampler with two distinct stages (init and adaptive). During the
adaptive phase, it uses a sampling map which specifies how many samples should
be drawn per pixel. Sampling is done using low-discrepancy sampling, and we keep
two sets of scrambling data (one per buffer).

twostages: a modified version of PBRT's "sampler" renderer, with specific code
to handle our "dualsampler" sampler. When sending samples to the film, our
renderer specifies the target buffer.

dualfilm: a modified version of PBRT's "image" film using two buffers. This
class stores the additional information we need (sample count, variance-related
statistics, and subpixel grid). It also forwards all calls from our "twostages"
renderer to the "nlmdenoiser". The "dualfilm" film is responsible for
instantiating the "nlmdenoiser".

As we extend PBRT's framework, we inherit its scene description system with some
additions. As an example, here is an exerpt from our "killines_dual.pbrt" scene
showing the options specific to our implementation:

Film "dual"
    "integer wnd_rad"     [10]
    "integer ptc_rad"     [3]
    "float   k"           [0.45]
    "integer xresolution" [1024]
    "integer yresolution" [1024]
Sampler "dual" "integer pixelsamples" [16]  "integer pixelsamplesinit" [4] "integer niterations" [3]
Renderer "twostages"

The Film specification instantiates our "dual" film, which will itself
instatiate our "nlmdenoiser" using a "wnd_rad" value of 10, a "ptc_rad" of 3,
and set "k" to 0.45 (see our paper for a description of these parameters). The
Sampler specification instantiates our "dualsampler" sampler, and requests 16
samples per pixel on average, using 4 samples per pixel in the initial sampling
pass, as well as 3 adaptive sampling iterations. Note that when "niterations" is
set to 0, our sampler produces a uniform sampling distribution. Lastly, the
Renderer specification instantiates our "twostages" renderer, which has no
parameters.

We also provide the scene description files of all scenes presented in our paper
in the directory "sa2012-scenes", along with all data (models, textures, brdfs)
needed. To render the "toasters" scene, one would simply cd to the
"sa2012-scenes" directory in a terminal and type:
../src/bin/pbrt toasters_dual.pbrt

The program outputs the following files:
toasters.exr (the usual PBRT output)
toasters_img.exr (equivalent to PBRT's output)
toasters_flt.exr (our final filtered output)
toasters_bspp.exr (final sample density map)

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


