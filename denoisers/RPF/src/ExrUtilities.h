#ifndef EXRUTILITIES_H
#define EXRUTILITIES_H

#include "Globals.h"
#include <assert.h>

//#define OPENEXR_DLL
#include <ImfRgbaFile.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfArray.h>
#include <Iex.h>
#include <half.h>
#include <algorithm>

FILE* OpenFile(char* fileName, char* type);
float* ImageRead(char* filename, int& width, int& height);
void readRgba1 (const char fileName[], Imf::Array2D<Imf::Rgba> &pixels, int &width, int &height);
void WriteEXRImage(const std::string &name, float *pixels, float *alpha, int xRes, int yRes, int totalXRes, int totalYRes, int xOffset, int yOffset);
void WriteEXRFile(char* fileName, int xRes, int yRes, float* input);
float tonemap(float val);
void convertFromInf(float& val);

// This functio is defined in SampleWriter/Globals.cpp
#ifndef _WINDOWS
int fopen_s(FILE **f, const char *name, const char *mode);
#endif

#endif
