#include "Globals.h"

// Initialize values for globals
size_t width = -1;
size_t height = -1;
size_t samplesPerPixel = -1;
size_t sampleLength = -1;
int posIndex = -1;
int colIndex = -1;
int featIndex = -1;
int randIndex = -1;
int numOfFeat = -1;
int numOfRand = -1;
int numOfIterations = -1;
int startSize = -1;
int endSize = -1;
int version = -1;
int blockReduceSize = -1; 
float gaussianDelta = GAUSSIAN_CUT /  GAUSSIAN_LENGTH;  
float gaussianDeltaInv = 1.f / gaussianDelta;
float gaussianTable[GAUSSIAN_LENGTH];
float variance = -1.0f;
float stdIncrease = -1.0f;
float randVarFactor = -1.0f;
int featureStdFactor = -1;
int worldStdFactor = -1;
float startMutual = -1.0f;
float mutualIncrease = -1.0f;
float colorStdFactor = -1.0f;
float reduceFactor = -1.0f;
float alphaFactor = 1.0f;
float textureFactor = 1.0f;
float normFactor = 1.0f;
int* blockSizes = NULL;
bool isThreshold = false;
bool isMean = false;
bool isMedian = false;
bool hasExplicitBlocks = false;


#ifndef _WINDOWS
#include <cstdlib>
#include <cassert>

int fopen_s(FILE **f, const char *name, const char *mode) {
    int ret = 0;
    assert(f);
    *f = fopen(name, mode);
    /* Can't be sure about 1-to-1 mapping of errno and MS' errno_t */
    if (!*f)
        ret = errno;
    return ret;
}
#endif
