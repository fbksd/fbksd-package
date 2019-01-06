#ifndef	SAMPLEWRITER_H_INCLUDED
#define SAMPLEWRITER_H_INCLUDED

#include "CImg.h"
#include "Globals.h"
//#include "../core/timer.h"
#include "../LBF/Globals.h"

using namespace cimg_library;

class SAMPLER_API SampleWriter {

public:

	static void Initialize(size_t width, size_t height, size_t samplesPerPixel);
    static void ProcessData(float* img);
	static bool ShouldSaveFeature(size_t x, size_t y, size_t k);
	static int GetPixelCount(size_t x, size_t y); 

	static size_t GetWidth();
	static size_t GetHeight();
	static size_t GetSamplesPerPixel();
	static size_t GetNumOfSamples();
	static SampleElem GetFeature(size_t x, size_t y, size_t k, OFFSET offset);
	static SampleElem GetFeature(size_t index, bool isTexture2);

	static void SetWidth(size_t width);
	static void SetHeight(size_t height);
	static void SetSamplesPerPixel(size_t samplesPerPixel);
	static void SetPosition(size_t x, size_t y, size_t k, SampleElem position, OFFSET offset);
	static void SetPosition(size_t index, SampleElem position, bool saveSample, int n, size_t x, size_t y, OFFSET offset);
	static void SetColor(size_t x, size_t y, size_t k, SampleElem color, OFFSET offset);
	static void SetColor(size_t index, SampleElem color, bool saveSample, int n, size_t x, size_t y, OFFSET offset);
	static void SetFeature(size_t x, size_t y, size_t k, SampleElem feature, OFFSET offset);
	static void SetFeature(size_t index, SampleElem feature, bool saveSample, int n, size_t x, size_t y, OFFSET offset);

private:

	static size_t GetIndex(size_t x, size_t y, size_t& k, bool& saveSample);

	static void ProcessTexture2Data();
	static void GenerateBufferIndex(int& bufInd, float& bufNormFactor, int n);
    static void SaveRenderTime(char* sceneName, char* inputFolder, char* name);

	static int numOfBuffers;
	static int* featureInd;
	static int* pixelSampleCount;
	static size_t width;
	static size_t height;
	static size_t spp;
	static size_t numOfSamples;
	static SampleElem* pixelData;
	static SampleElem* texture2Data;
	static SampleElem* sampleMean;
	static SampleElem* varData;
	static SampleElem** pixelBuffer;

};

#endif
