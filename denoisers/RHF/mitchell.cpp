#include "mitchell.h"

// Mitchell Filter Method Definitions
float MitchellFilter::Evaluate(float x, float y) const {
    return Mitchell1D(x * invXWidth) * Mitchell1D(y * invYWidth);
}


//MitchellFilter *CreateMitchellFilter(const ParamSet &ps) {
//    // Find common filter parameters
//    float xw = ps.FindOneFloat("xwidth", 2.f);
//    float yw = ps.FindOneFloat("ywidth", 2.f);
//    float B = ps.FindOneFloat("B", 1.f/3.f);
//    float C = ps.FindOneFloat("C", 1.f/3.f);
//    return new MitchellFilter(B, C, xw, yw);
//}


