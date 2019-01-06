
/*
    Copyright(c) 2012-2013 Tzu-Mao Li
    All rights reserved.

    The code is based on PBRT: http://www.pbrt.org

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


#if defined(_MSC_VER)
#pragma once
#endif

#ifndef SBF_SBF_COMMON_H__
#define SBF_SBF_COMMON_H__

#include "pbrt.h"
#include "VectorNf.h"

enum ESample
{
    IMAGE_X = 0,
    IMAGE_Y,
    COLOR,
    COLOR_R = COLOR,
    COLOR_G,
    COLOR_B,
    NORMAL,
    NORMAL_X = NORMAL,
    NORMAL_Z,
    NORMAL_Y,
    TEXTURE,
    TEXTURE_COLOR_R = TEXTURE,
    TEXTURE_COLOR_G,
    TEXTURE_COLOR_B,
    DEPTH,

    SAMPLE_SIZE
};

const int c_FeatureDim = 7;
typedef VectorNf<c_FeatureDim> Feature; // normal:3d, rho:3d, depth:1d

inline void ComputeSubWindow(int num, int count, int width, int height,
            int *xs, int *xe, int *ys, int *ye) {
    // Determine how many tiles to use in each dimension, _nx_ and _ny_
    int nx = count, ny = 1;
    while ((nx & 0x1) == 0 && 2 * width * ny < height * nx) {
        nx >>= 1;
        ny <<= 1;
    }

    // Compute $x$ and $y$ pixel sample range for sub-window
    int xo = num % nx, yo = num / nx;
    float tx0 = float(xo) / float(nx), tx1 = float(xo+1) / float(nx);
    float ty0 = float(yo) / float(ny), ty1 = float(yo+1) / float(ny);
    *xs = Floor2Int(Lerp(tx0, 0, width));
    *xe   = Floor2Int(Lerp(tx1, 0, width));
    *ys = Floor2Int(Lerp(ty0, 0, height));
    *ye   = Floor2Int(Lerp(ty1, 0, height));

}
 
#endif //#ifndef SBF_SBF_COMMON_H__
