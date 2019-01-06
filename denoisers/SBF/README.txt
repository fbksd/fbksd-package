Changelog:
2013/06/29
The adaptive sampling routine is modified so that: 
1. The filtering procedure is tuned so that the adaptive sampling now gives slightly better results.
2. The number of samples per pixel is no longer rounded up to power of 2.
3. Multiple iteration of adaptive sampling is supported.


--

This is the implementation of our paper:
"SURE-based Optimization for Adaptive Sampling and Reconstruction"
The code is based on the PBRT2 renderer by Matt Pharr and Greg Humphreys:
http://www.pbrt.org/
We have also used the wonderful fast math library written by herumi
http://homepage1.nifty.com/herumi/soft/fmath.html

Following files are modified/added from pbrt-v2:

src/
	core/
		api.cpp
		film.h
		intersection.h
		intersection.cpp
		reflection.h
		reflection.cpp
		image.h
		imageio.cpp  -- fix warning
		parallel.cpp -- fix warning
		volume.cpp   -- fix warning
	integrators/
		path.cpp 
			-- We modified the specular materials sampling strategy to better handle 
			   multiple specular vertices light path
	film/
		sbfimage.h
		sbfimage.cpp
	renderer/
		sbfrenderer.h
		sbfrenderer.cpp
	sampler/
		sbfsampler.h
		sbfsampler.cpp
	sbf/
		CrossBilateralFilter.h
		CrossBilateralFilter.cpp
		CrossNLMFilter.h
		ReconstructionFilter.h
		SBFCommon.h
		TwoDArray.h
		VectorNf.h
		fmath.hpp -- fast math library by herumi
		sbf.h
		sbf.cpp -- The main part of the code
	tools/
		exrdiff.cpp -- changed the metric to relative MSE, and output color tempature diff
sbf_scenes/
	two demo scenes, see their .pbrt files for details of parameters.
Makefile
	Changed -O2 to -Ofast, this does not speedup pbrt much, but our code is 
	siginificantly faster, probably because the -ffast-math flag.

Contact us if you have any questions! 
bachi@cmlab.csie.ntu.edu.tw

--

The code is released under the BSD 2-clause license.

Copyright (c) 2012-2013, Tzu-Mao Li
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
