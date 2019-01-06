#ifndef	FeatureExtractor_H_INCLUDED
#define FeatureExtractor_H_INCLUDED

#include <vector>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Globals.h"

using namespace std;

extern cudaArray* devSampleArray;
extern cudaChannelFormatDesc channelDesc;

class FeatureExtractor {

public:

	FeatureExtractor(float* pbrtPixelData, float* pbrtVarData, float** pixelBuffer, int pbrtWidth, int pbrtHeight, int pbrtSpp, char* inputFolder); 
	~FeatureExtractor();

	int GetFeatureLength();
	float* GetPixelFeatureMu();
	float* GetPixelFeatureSigma();
	float* GetVar();
	float* GetFilteredData();
	float* GetNormMu();
	float* GetNormStd();
	float* GetFeatures();
	
	void PrefilterFeatures(float* pbrtPixelData, float** pixelBuffer);
	void ExtractMAD();
	void ExtractGradients();
	void ExtractFeatureStatistics();
	void ExtractMeanDeviation();
	void ExtractSpp();

private:

	float* HostToDevice(float* data, int size);
	float* DeviceToHost(float* data, int size);

	void CalcMAD(float* MADFeature, int blockSize, int offset);
	void Dilate(float* input);
	void GetData(float* img, int offset);
	void LoadInputToCudaTexture(bool shouldAllocate);
	void NormalizeVar(float* varData);
	void ProcessBufferVar(float* varData, float** pixelBuffer);
	void SamplesToTexture(float4* samplesTexture, float* data, int width, int height);
	void LoadData(float* pbrtPixelData, float* pbrtVarData, float** pixelBuffer);
	void SaveNoisyData(char* filename, float* data, size_t imgWidth, size_t imgHeight, size_t imgSamplesPerPixel);
	void SaveVal(int x, int y, int offset, float val);
	float NormalizeVal(float val, int offset);

	float* var;
	float* filteredData;
	float* pixelFeatureMu; 
	float* pixelFeatureSigma; 

	float4* hostSamplesTexture;

	int numOfSamples;
	int totalSize;
	int width; 
	int height;
	int spp;
	int sampleLength;
	int featureLength;

	float* featMu;
	float* featStd;
	float* filterVar;
	float* colorVar;
	float* features;
	
};

#endif