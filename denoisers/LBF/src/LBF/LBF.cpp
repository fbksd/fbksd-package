#include <iostream>
#include <vector>
#include <random>
#include <assert.h>
#include <fstream>
#include "Globals.h"
#include "CImg.h"
#include "FeatureExtractor.h"
#include "NeuralNetwork.h"
//#include "../core/timer.h"

using namespace std;

void LBF(char* inputFolder, char* sceneName, float* pbrtPixelData, float* pbrtVarData, float** pixelBuffer, int pbrtWidth, 
                      int pbrtHeight, int pbrtSpp, float* img) {
	
//	Timer featureTimer;
//	featureTimer.Start();
	printf("Running Feature Extractor\n");

	// Run feature extractor (Sec. 3.3)
	FeatureExtractor* featureExtractor = new FeatureExtractor(pbrtPixelData, pbrtVarData, pixelBuffer, pbrtWidth, pbrtHeight, pbrtSpp, inputFolder);
	featureExtractor->ExtractMAD();
	featureExtractor->PrefilterFeatures(pbrtPixelData, pixelBuffer);
	featureExtractor->ExtractGradients();
	featureExtractor->ExtractFeatureStatistics();
	featureExtractor->ExtractMeanDeviation();
	featureExtractor->ExtractSpp();
	
	// Record time
//	char fileName[1000];
//	sprintf(fileName, "%s/%s_timing.txt", inputFolder, sceneName);
//	FILE* fp = OpenFile(fileName, "at");
//	featureTimer.Stop();
//	fprintf(fp, "Feature Extraction Time: %f sec\n", featureTimer.Time() + time);
//	featureTimer.Reset();
//	featureTimer.Start();

	// Run neural network and filter
	printf("Running NN and Filter\n");
	float* varData = featureExtractor->GetVar();
	int featureLength = featureExtractor->GetFeatureLength();
	float* featureData = new float[featureLength * pbrtWidth * pbrtHeight];
	memcpy(featureData, featureExtractor->GetFeatures(), featureLength * pbrtWidth * pbrtHeight * sizeof(float));
	NeuralNetwork* network = new NeuralNetwork(inputFolder, pbrtWidth, pbrtHeight, featureLength);
    network->ApplyWeightsAndFilter(inputFolder, sceneName, featureData, varData, img);
	
	// Cleanup
	delete network;
	delete featureExtractor;
	cudaDeviceReset();

	// Record time
//	featureTimer.Stop();
//	fprintf(fp, "NN/Filter Time: %f sec\n", featureTimer.Time());
//	fclose(fp);
}

