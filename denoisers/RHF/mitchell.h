#ifndef PBRT_FILTERS_MITCHELL_H
#define PBRT_FILTERS_MITCHELL_H

#include <cmath>

class Filter {
public:
    // Filter Interface
    virtual ~Filter(){}
    Filter(float xw, float yw)
        : xWidth(xw), yWidth(yw), invXWidth(1.f/xw), invYWidth(1.f/yw) {
    }
    virtual float Evaluate(float x, float y) const = 0;

    // Filter Public Data
    const float xWidth, yWidth;
    const float invXWidth, invYWidth;
};


// Mitchell Filter Declarations
class MitchellFilter : public Filter {
public:
    // MitchellFilter Public Methods
    MitchellFilter(float b, float c, float xw, float yw)
        : Filter(xw, yw), B(b), C(c) {
    }
    float Evaluate(float x, float y) const;
    float Mitchell1D(float x) const {
        x = std::abs(2.f * x);
        if (x > 1.f)
            return ((-B - 6*C) * x*x*x + (6*B + 30*C) * x*x +
                    (-12*B - 48*C) * x + (8*B + 24*C)) * (1.f/6.f);
        else
            return ((12 - 9*B - 6*C) * x*x*x +
                    (-18 + 12*B + 6*C) * x*x +
                    (6 - 2*B)) * (1.f/6.f);
    }
private:
    const float B, C;
};


#endif // PBRT_FILTERS_MITCHELL_H
