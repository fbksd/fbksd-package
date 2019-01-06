#ifndef	RPF_H_INCLUDED
#define RPF_H_INCLUDED

#include <iostream>
#include <vector>
#include <random>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <fstream>
#include "PreComputeGaussian.h"
#include "Globals.h"
#include "CImg.h"
#include "SampleSet.h"

#include "timer.h"

using namespace std;
using namespace cimg_library;

extern SampleSet* samples; 

// External functions used here but defined in MKL_Utils.h
void computeExp(float* x_pfVector, int x_nSize);
void generateRandomNormal(float* x_pfVector, size_t x_nSize, float x_fSigma);

//**** INPUT FUNCTIONS *****//
void initializeData(float* pbrtData, size_t pbrtWidth, size_t pbrtHeight, 
					size_t pbrtSpp, size_t pbrtSampleLength, int posCount, int colorCount, int featureCount, int randomCount, FILE* datafp);

// Read parameters from config file
void parseConfigFile(FILE* fp);

// Read in sample data from file
void parseSampleFile(FILE* fp);

// Set the default parameters (no config file needed)
void setDefaultParameters();

// Make sure that the parameters from the config file make sense
void checkParameters();

//***** RPF FUNCTIONS *****//

// Main RPF algorithm
// Takes in two empty image buffers and a filename. After the algorithm
// is finished, it has the noisy image in origImg and the filtered output
// in rpfImg
void RPF(CImg<float>* rpfImg, CImg<float>* origImg);

// Choose samples in block of size b around pixel at x, y. The maxNumOfSamples are specified
// and the selected samples are stored in neighboringSamples. The samples are also normalized 
// at the end of the function.
void preProcessSamples(int b, size_t maxNumOfSamples, SampleSet* neighboringSamples, int x, int y);

// The pdfs, mutual information, and dependencies are calculated here to obtain the alpha 
// and beta weights
float computeFeatureWeights(int t, SampleSet* neighboringSamples, float* alpha, float* beta);

// The joint bilateral filter is applied with the previously calculated weights to get the color at
// the current pixel. The result is stored in the neighboringSamples set
void filterColorSamples(SampleSet* neighboringSamples, float* alpha, float* beta, float contributionCR, int t);

// Perform boxFiltering on the samples in the "samples" array and save in img. Also requires user to 
// specify whether the box filter is being applied on the original or filtered samples.
void boxFilter(CImg<float>* img, bool calculateOriginal);

#endif
