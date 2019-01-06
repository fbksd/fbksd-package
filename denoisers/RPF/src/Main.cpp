#include <iostream>
#include <vector>
#include <random>
#include <assert.h>
#include <fstream>
#include "Globals.h"
#include "CImg.h"
#include "SampleSet.h"
#include "ExrUtilities.h"
//#include <direct.h>
#include "../core/timer.h"

using namespace std;

void RPF(CImg<float>* rpfImg, CImg<float>* origImg);

void initializeData(float* pbrtData, size_t pbrtWidth, size_t pbrtHeight, 
					size_t pbrtSpp, size_t pbrtSampleLength, int posCount, int colorCount, int featureCount, int randomCount, FILE* datafp);

//// An array of pointers that point to feature information for each
//// sample in the entire image
//SampleSet* samples = NULL;

//void RPF(char* outputFolder, float* pbrtData, size_t pbrtWidth,
//					  size_t pbrtHeight, size_t pbrtSpp, size_t pbrtSampleLength, int posCount, int colorCount, int featureCount, int randomCount, FILE* datafp) {

//	fprintf(stdout, "Starting RPF\n");
//	fflush(stdout);

//	// Start timer
//	Timer timer;
//	timer.Start();

//	// Get samples from file and add to list of samples
//	initializeData(pbrtData, pbrtWidth, pbrtHeight, pbrtSpp, pbrtSampleLength, posCount, colorCount, featureCount, randomCount, datafp);

//	// Initialize image
//	CImg<float>* rpfImg = new CImg<float>(width, height, 1, 3);
//	CImg<float>* origImg = new CImg<float>(width, height, 1, 3);

//	// Perform random parameter filtering
//	RPF(rpfImg, origImg);

//	// Save filtered image
//	char outputName[1000];
//	sprintf(outputName, "%s_RPF_flt.exr", outputFolder);
//	float* imgData = new float[NUM_OF_COLORS * pbrtWidth * pbrtHeight];
//	WriteEXRFile(outputName, (int) pbrtWidth, (int) pbrtHeight, rpfImg->data());

//	// Save original image
//	sprintf(outputName, "%s_MC_%04d.exr", outputFolder, pbrtSpp);
//	WriteEXRFile(outputName, (int) pbrtWidth, (int) pbrtHeight, origImg->data());

//	// Clean up
//	delete rpfImg;
//	delete origImg;
//	delete samples;
//	delete[] imgData;

//	// Wait for user
//	printf("Finished with RPF\n");

//	// Output runtime
//	timer.Stop();
//	float runtime = (float) timer.Time();
//	fprintf(stdout, "Runtime: %.2lf secs\n\n", runtime);

//}


