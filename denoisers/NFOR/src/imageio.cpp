
/*
    pbrt source code Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

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


// core/imageio.cpp*
#include "imageio.h"
#include "utils.h"

#include <vector>
#include <ImfInputFile.h>
#include <ImfRgbaFile.h>
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <half.h>
using namespace Imf;
using namespace Imath;


static void WriteImageEXR(const string &name, float *pixels,
        float *alpha, int xRes, int yRes,
        int totalXRes, int totalYRes,
        int xOffset, int yOffset) {
    Rgba *hrgba = new Rgba[xRes * yRes];
    for (int i = 0; i < xRes * yRes; ++i)
        hrgba[i] = Rgba(pixels[3*i], pixels[3*i+1], pixels[3*i+2],
                        alpha ? alpha[i]: 1.f);

    Box2i displayWindow(V2i(0,0), V2i(totalXRes-1, totalYRes-1));
    Box2i dataWindow(V2i(xOffset, yOffset), V2i(xOffset + xRes - 1, yOffset + yRes - 1));

    try {
        RgbaOutputFile file(name.c_str(), displayWindow, dataWindow, WRITE_RGBA);
        file.setFrameBuffer(hrgba - xOffset - yOffset * xRes, 1, xRes);
        file.writePixels(yRes);
    }
    catch (const std::exception &e) {
        Error("Unable to write image file \"%s\": %s", name.c_str(),
            e.what());
    }

    delete[] hrgba;
}

void WriteImage(const string &name, float *pixels, float *alpha, int xRes,
                int yRes, int totalXRes, int totalYRes, int xOffset, int yOffset) {
    if (name.size() >= 5) {
        uint32_t suffixOffset = name.size() - 4;
        if (!strcmp(name.c_str() + suffixOffset, ".exr") ||
            !strcmp(name.c_str() + suffixOffset, ".EXR")) {
             WriteImageEXR(name, pixels, alpha, xRes, yRes, totalXRes,
                           totalYRes, xOffset, yOffset);
             return;
        }
    }
    Error("Can't determine image file type from suffix of filename \"%s\"",
          name.c_str());
}

void WriteImageMono(const string &name, float *pixels, float *alpha, int xRes,
                int yRes, int totalXRes, int totalYRes, int xOffset, int yOffset) {
    if (name.size() >= 5) {
        uint32_t suffixOffset = name.size() - 4;
        if (!strcmp(name.c_str() + suffixOffset, ".exr") || !strcmp(name.c_str() + suffixOffset, ".EXR")) {
            size_t numPixels = xRes*yRes;
            std::vector<float> tmp(numPixels*3);
            for(size_t i = 0; i < numPixels; ++i)
            {
                tmp[i*3] = pixels[i*3];
                tmp[i*3 + 1] = pixels[i*3];
                tmp[i*3 + 2] = pixels[i*3];
            }

            WriteImageEXR(name, tmp.data(), alpha, xRes, yRes, totalXRes,
                          totalYRes, xOffset, yOffset);
            return;
        }
    }
    Error("Can't determine image file type from suffix of filename \"%s\"",
          name.c_str());
}
