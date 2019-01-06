#ifndef UTILITIES_H
#define UTILITIES_H

template<class T> class Matrix;

#include <stdio.h>
#include <cublas_v2.h>
#include "Globals.h"
//#include "Matrix.h"

//#define OPENEXR_DLL
//#include <ImfRgbaFile.h>
//#include <ImfOutputFile.h>
//#include <ImfInputFile.h>
//#include <ImfChannelList.h>
//#include <ImfStringAttribute.h>
//#include <ImfMatrixAttribute.h>
//#include <ImfArray.h>
//#include <Iex.h>
//#include <half.h>

#include <iostream>
//#include <tchar.h>
#include <algorithm>

FILE* OpenFile(char* fileName, char* type);
void PrintError(cudaError_t err, char* file, int line);
void CublasErrorCheck(cublasStatus_t err);
void PrintAvailableMemory();
void WriteEXRImage(const std::string &name, float *pixels, float *alpha, int xRes, int yRes, int totalXRes, int totalYRes, int xOffset, int yOffset);
void WriteEXRFile(char* fileName, int xRes, int yRes, float* input);

#endif
