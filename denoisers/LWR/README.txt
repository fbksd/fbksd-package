The implementations we provide are the C/C++ & CUDA codes to test a work
described in the following paper:

"Adaptive Rendering based on Weighted Local Regression" written by 
Bochang Moon, Nathan Carr, Sung-Eui Yoon.

The codes are recently tested in the following environment: 

OS: Windows 7 64bit
GPU: NVIDIA GeForce GTX TITAN 
CUDA version: 5.0 
Development environment: Visual studio 2010


1. Parameter Setting: We are using some pre-defined symbols for some parameters
(e.g., filtering window size), and those values are in "lwrr_setting.h". We
have used the filter size for all the tests, but you may change the values for
your tests. 

2. Extended feature vectors: To test the extended feature vectors (Fig. 1)
where we test our bandwidth selection even for a naively added feature, please
uncomment "#define FEATURE_MOTION" in "lwrr_setting.h". In other tests, the
symbol should be commented.

3. Additional Notes: We have implemented our work using C/C++ & CUDA (lwrr
project), and tested the project on top of pbrt.  If you want to test our work
with your own renderer, you should include the lwrr project.  Also, you should
write some test files (e.g., some files in the lwrr_test folder) to link your
project and the lwrr. 

Issues:
1) In our work, we have used a pixel filter (box filter). If you need other
filters (e.g., Gaussian), you should modify a function
"LWR_Film::AddSampleExtended()" for your tests. 
2) This version does not support some pbrt features (e.g., cropwindow). 

----
If you find some bugs or issues, please send an email to Bochang Moon (moonbochang@gmail.com).


