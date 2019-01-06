
Supplemental materials for our paper:
Adaptive Sampling and Reconstruction using Greedy Error Minimization

We provide both full-resolution of our paper results in the "figures" directory,
and full source code implementation in the "pbrt_mse" directory.

Additional questions are welcome:
roussell@aim.unibe.ch

--------------------------------------------------------------------------------
FIGURES

We provide all images related to the teaser and the Figure 12 of our paper. Here
is a breakdown of what's provided, taking the "toasters" scene as an example:

OUR method:
toasters_g0032bdw_our_img.exr (finest scale)
toasters_g0032bdw_our_flt.exr (adaptive reconstruction)
toasters_g0032bdw_our_smp.exr (sample density map)
toasters_g0032bdw_our_map.exr (scale selection map used in final reconstruction)

OUR-GRD method, ie. OUR method + ground truth statistics:
toasters_g0032bdw_grd_img.exr (finest scale)
toasters_g0032bdw_grd_flt.exr (adaptive reconstruction)
toasters_g0032bdw_grd_smp.exr (sample density map)
toasters_g0032bdw_grd_map.exr (scale selection map used in final reconstruction)

AWR method:
toasters_g0032bdw_awr_img.exr (finest scale)
toasters_g0032bdw_awr_flt.exr (adaptive reconstruction)
toasters_g0032bdw_awr_smp.exr (sample density map)

NAIVE method:
toasters_naive_equaltime.exr

REFERENCE:
toasters_g4096.exr (rendering using 4096 stratified samples per pixel)

All files are provided for the scenes of Figure 12 (killines, plants-dusk,
sibenik, toasters, and yeahright). For the teaser scene (dragonfog), we only
provided the files corresponding to OUR method. The "g0032bdw" in the filename
indicates the use of a Gaussian pixel filter, with 32 samples per pixel on
average, and the use of an adaptive sampler. The "naive_equaltime" images were
produced with more samples per pixel, in order to match the rendering time of
the "g0032bdw_our_flt.exr" images.

In addition to the EXR files, tonemapped PNG files are provided.

--------------------------------------------------------------------------------
PBRT_MSE

We provide a full PBRT implementation, including our changes.
********************************** IMPORTANT ***********************************
The Makefile in the "src" subdirectory is functional. The other configurations
(src/pbrt.vs2008, src/pbrt.vs2010, and pbrt.xcodeproj) were neither updated,
nor tested.
********************************************************************************

We only modified two files from the original PBRT framework:
"src/core/api.cpp"
"src/Makefile"

The changes to "api.cpp" allow the parsing of our own sampler ("bandwidth"),
renderer ("twostages"), and film ("smooth") classes. The change in "Makefile" is
to enable OpenMP support. If OpenMP is unavailable, our own processing will not
be parallelized.

We added the following files:
"src/core/kernel2d.h" and "src/core/kernel2d.cpp"
"src/core/denoiser.h" and "src/core/denoiser.cpp"
"src/samplers/bandwidth.h" and "src/samplers/bandwidth.cpp"
"src/renderers/twostages.h" and "src/renderers/twostages.cpp"
"src/film/smooth.h" and "src/film/smooth.cpp"

In theory, one should be able to download the latest version of PBRTv2, and add
our new files, as well as the two modified files, to get a functional version
of our implementation. This is how the present code distribution was obtained.

The main entry point to our algorithm is the call:
"bwSampler->GetWorstPixels(nPixelsPerIteration);"
in "src/renderers/twostages.cpp". We suggest tracing from that point to get a
better understanding of the implementation. Here is some high-level information
about each class:

kernel2d: a simple utility class performing the filtering specifically tailored
to our needs.

denoiser: this implements our scale selection method, adaptive reconstruction,
and selects pixels to be sampled during the adaptive phase.

bandwidth: a sampler with two distinct stages (init and adaptive). During the
adaptive phase, it receives a list of pixels to sample from, along with the
scale information needed. We perform importance sampling in the screen space
following the scale distribution (typically a gaussian), and random sampling in
all other dimensions. Note that screen samples follow a low-discrepancy
sequence within each given pixel.

twostages: a modified version of PBRT's "sampler" renderer, with specific code
to handle our "bandwidth" sampler.

smooth: a modified version of PBRT's "image" film. This one stores the
additional information we need (sample count, variance-related statistics, and
subpixel grid). It also forwards all calls from our "twostages" renderer to the
"denoiser". The "smooth" film is responsible for instantiating the "denoiser".

As we extend PBRT's framework, we inherit its scene description system with some
additions. As an example, here is an exerpt from our "toaster.pbrt" scene
showing the options specific to our implementation:

Film "smooth"
    "float  gamma"        [0.2]
    "integer xresolution" [1024] "integer yresolution" [1024]
    "string filename"     ["toasters.exr"]
Sampler "bandwidth" "integer pixelsamples" [32] "integer niterations" [8] 
Renderer "twostages"

The Film specification instantiates our "smooth" film, which will itself
instatiate our "denoiser" using a "gamma" value of 0.2 (see our paper for a
description of this gamma parameter). The Sampler specification instantiates
our "bandwidth" sampler, and requests 32 samples per pixel on average, as well
as 8 adaptive sampling iterations. Note that when "niterations" is set to 0, our
sampler produces a uniform sampling distribution. Lastly, the Renderer
specification instantiates our "twostages" renderer, which has no parameters.

We also provide the scene description files of all scenes presented in our paper
in the directory "sa2011-scenes", along with all data (models, textures, brdfs)
needed. To render the "toasters" scene, one would simply cd to the
"sa2011-scenes" directory in a terminal and type:
../src/bin/pbrt toasters.pbrt

For instructions on how to compile PBRT, please refer to PBRT's documentation.

