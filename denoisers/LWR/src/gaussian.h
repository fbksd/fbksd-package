
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

#if defined(_MSC_VER)
#pragma once
#endif

#ifndef PBRT_FILTERS_GAUSSIAN_H
#define PBRT_FILTERS_GAUSSIAN_H

// filters/gaussian.h*
#include "filter.h"

// Gaussian Filter Declarations
class GaussianFilter : public Filter {
public:
    // GaussianFilter Public Methods
    GaussianFilter(float xw, float yw, float a)
        : Filter(xw, yw), alpha(a), expX(expf(-alpha * xWidth * xWidth)),
          expY(expf(-alpha * yWidth * yWidth)) { }
    float Evaluate(float x, float y) const;
private:
    // GaussianFilter Private Data
    const float alpha;
    const float expX, expY;

    // GaussianFilter Utility Functions
    float Gaussian(float d, float expv) const {
        return max(0.f, float(expf(-alpha * d * d) - expv));
    }
};


GaussianFilter *CreateGaussianFilter(const ParamSet &ps);

#endif // PBRT_FILTERS_GAUSSIAN_H
