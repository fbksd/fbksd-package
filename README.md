# FBKSD Workspace Package

## Introduction

This repository contains a collection of renderers, denoisers, and samplers that can be installed in a FBKSD workspace. 

We also provide scenes, but they are hosted in a different location due to github file size limitations.

### Renderers

The included renderers are:

- [pbrt-v2](https://github.com/mmp/pbrt-v2)
- [pbrt-v3](https://github.com/mmp/pbrt-v3)
- [Mitsuba](https://github.com/mitsuba-renderer/mitsuba)
- A custom procedural renderer

The first three were adapted from the originals to be used as FBKSD rendering back-ends.
The fourth one is a custom procedural renderer that allows rendering mathematical expressions.
All renderers are included as git submodules.

### Denoisers

The included denoisers are:

- LWR [1]
- NFOR [2]
- LBF [3]
- RPF [4]
- SBF [5]
- RHF [6]
- NLM [7]
- RDFC [8]
- GEM [9]
- Box
- Gaussian
- Mitchell

### Samplers

The included samplers are:

- Independent
- Stratified
- Sobol
- Low discrepancy


## System Requirements

- fbksd was tested on Ubuntu 18.04, but it should work in any modern Linux distribution with a recent cmake version;
- The file system must support symbolic links: e.g. EXT4;
- NVIDIA graphics card for denoisers that require CUDA (optional)


## Build and Install

### Dependencies

Each renderer has its own set of dependencies. Please refer to the corresponding project pages for more info.

The list below is a good summary:

- Boost
- OpenEXR
- OpenCV
- Eigen
- XERCES
- FFTW
- GLEW
- Python3
- CUDA (Optional)
  - if CUDA is not installed, denoisers that depend on it will not be compiled.
  - If CUDA is present, gcc version 7 and above are not supported (at this time).
    If your default gcc version id 7 or above, set a different version using the `CC` and `CXX` environment variables before running cmake, as in the example below.

After installing the dependencies, build with `cmake` setting the `CMAKE_INSTALL_PREFIX` variable to the your FBKSD workspace location:

```
$ git clone --recurse-submodules https://github.com/fbksd/fbksd-package.git
$ cd fbksd-package
$ export CC=/usr/bin/gcc-6
$ export CXX=/usr/bin/g++-6
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<workspace path> ../
$ cmake install
```

> Note: This repository uses git submodules. Clone with the `--recurse-submodules` flag.


## Install the Scenes

We couldn't include scenes in this repository due to github file size limitations.
So we provide it as a package hosted somewhere else. 

To install the scenes in your FBKSD workspace, just download the compressed folder [scenes.tar.gz](https://drive.google.com/open?id=1bjBWfLF6ub5Ec1Lgnh65l48EKmHEPzpj) and extract it into the workspace folder. If your workspace already has the `scenes` folder in it, merge the two folders.

The workspace structure should look like this:

```
<workspace>/scenes/pbrt-v2/*
<workspace>/scenes/pbrt-v3/*
<workspace>/scenes/mitsuba/*
...
```

If you prefer, instead of extracting the `scenes.tar.gz` folder directly in your workspace folder, you can extract it somewhere else and just create a symlink for it in the workspace.
The `fbksd` CLI has a command option that creates the link for you:

```
$ fbksd scenes --set <path to the scenes folder>
```

---
## References

[1] Moon, B., Carr, N., and Yoon, S. 2014. Adaptive Rendering Based on Weighted Local Regression. ACM Transactions on Graphics 33, 5, 1–14.

[2] Bitterli, B., Rousselle, F., Moon, B., et al. 2016. Nonlinearly Weighted First-order Regression for Denoising Monte Carlo Renderings. Computer Graphics Forum 35, 4, 107–117.

[3] Kalantari, N.K., Bako, S., and Sen, P. 2015. A machine learning approach for filtering Monte Carlo noise. ACM Transactions on Graphics 34, 4, 122:1-122:12.

[4] Sen, P. and Darabi, S. 2012. On filtering the noise from the random parameters in Monte Carlo rendering. ACM Transactions on Graphics 31, 3, 1–15.

[5] Li, T.-M., Wu, Y., and Chuang, Y. 2012. SURE-based optimization for adaptive sampling and reconstruction. ACM Transactions on Graphics 31, 1.

[6] Delbracio, M., Musé, P., Buades, A., Chauvier, J., Phelps, N., and Morel, J.-M. 2014. Boosting monte carlo rendering by ray histogram fusion. ACM Transactions on Graphics 33, 1, 1–15.

[7] Rousselle, F., Knaus, C., and Zwicker, M. 2012. Adaptive rendering with non-local means filtering. ACM Transactions on Graphics 31, 6, 1.

[8] Rousselle, F., Manzi, M., and Zwicker, M. 2013. Robust denoising using feature and color information. Computer Graphics Forum 32, 7, 121–130.

[9] Rousselle, F., Knaus, C., and Zwicker, M. 2011. Adaptive sampling and reconstruction using greedy error minimization. ACM Transactions on Graphics 30, 6, 1.
