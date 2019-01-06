SOURCE CODE FOR "A MACHINE LEARNING APPROACH FOR FILTERING MONTE CARLO NOISE"

This package is a C++/CUDA implementation of the Learning Based Filtering 
(LBF) algorithm described in:

N. K. Kalantari, S. Bako, P. Sen, "A Machine Learning Approach for Filtering 
Monte Carlo Noise", ACM Transaction on Graphics, Volume 34, Number 4, August 2015.

More information can also be found on the authors' project webpage:
http://dx.doi.org/10.7919/F4CC0XM4

Initial release implementation by Nima Khademi Kalantari and Steve Bako, 2015.


-------------------------------------------------------------------------
I. LICENSE CONDITIONS

Copyright (c) 2015.  The Regents of the University of California.  
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted for non-commercial purposes provided that the following conditions 
are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation and/or 
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors 
may be used to endorse or promote products derived from this software without 
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
OF THE POSSIBILITY OF SUCH DAMAGE.


-------------------------------------------------------------------------
II. OVERVIEW

The code takes in a pbrt scene with a specified sampling rate that is rendered
with pbrt and sent to our algorithm, LBF. The noisy data is then processed to 
get a filtered output that is close to the ground truth image rendered with many
samples.

The code was written in C++/CUDA 6.5 and tested on both Windows 8 x64 and 
Windows 7 x64 with a GeForce GTX TITAN and GeForce GTX 760 using 
Visual Studio 2012.


-------------------------------------------------------------------------
III. RUNNING THE PACKAGE

Requirements: Nvidia GPU with CUDA 5.5 or above installed.
(https://developer.nvidia.com/cuda-toolkit-archive)

1. Navigate to pbrt-v2-lbf\src\pbrt.vs2012\

2. Open the solution (pbrt.sln) with Visual Studio 2012.

3. Make sure the CUDA compatibility for the LBF project
   meets your card requirements:

   http://en.wikipedia.org/wiki/CUDA#Supported_GPUs
   
   The default is 3.0. Build the solution. See the 
   pbrt README for additional info on how to build pbrt.

4. Run the code through either Visual Studio or the command line using 
   pbrt.exe in the pbrt-v2-lbf\bin\ folder. The general command line 
   arguments from the "scenes" folder are as follows:
   
   ./../bin/pbrt.exe scenename --spp samplenum 
   
   For example, to run the room-path scene from the "scenes" folder with 16 
   samples per pixel, the command is:
   
   ./../bin/pbrt.exe room-path.pbrt --spp 16
   
   The three outputs (noisy image, filtered result, and timing) will be output in the 
   scenes directory. For example, for the room-path command above, the outputs
   would be:
   
   room-path_MC_0016.exr (the 16spp Monte Carlo image)
   room-path_LBF_flt.exr (the filtered result from LBF)
   room-path_timing.txt (time for rendering and filtering)

***********************************************************************   
NOTE: 

- The solution has been set up for CUDA 6.5. If you have a different version
  of CUDA, follow the below steps to set up the project.

	1) Open LBF.vcxproj in text editor

	2) Find the following two lines

	<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 6.5.props" />
	<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 6.5.targets" />

	3) Change "6.5" to your CUDA version

- Make sure the CUDA_PATH environment variable points to your
  CUDA installation folder (e.g., "NVIDIA GPU Computing Toolkit\CUDA\v6.5").

- If the code complains about the "cudart$(Platform)_$(CUDA_Version).dll"
  (e.g., cudart64_65.dll for x64 platform and CUDA 6.5), find the dll in the
  CUDA installation folder (e.g., "NVIDIA GPU Computing Toolkit\CUDA\v6.5\bin") 
  and copy it to the "bin" folder in the root of the project.
  
- If you have a GeForce GTX TITAN or better, you can set the FAST_FILTER
  flag in the LBF\Globals.h to "1", for improved performance. By default,
  this flag is set to "0".
  
- The renderer samples the scene randomly, and thus, each run produces a
  slightly different input noisy and consequently filtered image.

***********************************************************************
   
-------------------------------------------------------------------------
V. CODE DETAILS

We added two additional projects to pbrt. The first is the "SampleWriter" 
project, which is a static class that will hold the raw data from the pbrt.
Since we need to save data that is not output by default from the renderer,
we modified some files in pbrt to save this in the static SampleWriter class.
These modifications can be found by doing a search for the macro "SAVE_SAMPLES",
which surrounds all calls to the SampleWriter. To revert back to original pbrt
without saving any samples or filtering, set "SAVE_SAMPLES" to 0 in "Globals.h" 
of the SampleWriter project.

Note, we utilize the 2-buffer variance as described in Rousselle et al. 2012 and 
Rousselle et al. 2013. Thus, we directly use the multisampler.h and multisampler.cpp
files found in the code for Rousselle et al. 2013. Thus in the scene file the sampler
should be changed to "multi". However, the "lowdiscrepancy" and "random" samplers
are overloaded to revert to the multisampler.

The second project that we added is the "LBF" project which contains our post-process
algorithm. Specifically, it has the feature extractor, which calculates the features 
that will be the inputs to the neural network and which pre-filters the noisy features.
Note, since the features need to be normalized, the feature extractor will read in 
normalization weights from the "FeatureNorm.dat" file given in the scenes directory. 
After extracting the features, the neural network is created which reads in the trained
weights found in "Weights.dat", evaluates the network to get filter weights, and applies
the filter to obtain the final denoised image. To better understand the code, we suggest 
you start with the LBF function in LBF.cpp.

Note that if you have a GeForce GTX TITAN or better, you can set the FAST_FILTER
flag in the LBF\Globals.h to "1", for improved performance. By default,
this flag is set to "0".

-------------------------------------------------------------------------
VI. TEST SCENES

Scenes can be run without modification through the following download from the pbrt 
website:

http://www.pbrt.org/scenes.php

1. Download pbrt-scenes.zip

2. Extract zip file. Move the contents of the pbrt-scenes folder to pbrt-v2-lbf/scenes.

Note that the following common pbrt scenes from the above package were used in our 
training set:

buddha
cornell box (with path-tracing)
dof-dragons
yeahright

Thus, it would be unfair to do a direct comparison on any of these scenes using the 
weights provided. Our system has been trained on 20 scenes including conference, 
dragonfog, etc., which are not part of the default package. For a complete list of 
training scenes, please refer to the supplementary materials.

   
-------------------------------------------------------------------------
VII. TONEMAPPING SETTINGS

We use gamma 2.2 for all scenes unless otherwise noted below.

San Miguel Terrace: 0.5  scale, 2.2 gamma 
San Miguel Balcony: 1.75 scale, 2.2 gamma (for insets)
Teapot Room: 		2.0  scale, 2.2 gamma (for insets)
San Miguel Hallway: 3.5  scale, 2.2 gamma (for insets)


-------------------------------------------------------------------------
VIII. VERSION HISTORY	
	
v1.0 - Initial release   (05/19/2015)

-------------------------------------------------------------------------

If you find any bugs or have comments/questions, please contact 
Nima K. Kalantari (nima@umail.ucsb.edu) or Steve Bako (stevebako@umail.ucsb.edu).

Santa Barbara, California
05/19/2015