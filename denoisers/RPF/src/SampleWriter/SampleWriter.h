#ifndef	SAMPLEWRITER_H_INCLUDED
#define SAMPLEWRITER_H_INCLUDED

#include <iostream>
#include <assert.h>
//#include <Windows.h>
#include <stdio.h>
//#include <direct.h>
#include "Globals.h"
#include "CImg.h"
#include "../Globals.h"

using namespace cimg_library;

class SAMPLER_API SampleWriter {

public:

	static void initialize(size_t width, size_t height, size_t samplesPerPixel);

    static void ProcessData(float* result);
	static void readSamplesFromFile(char* fileName, SampleElem* data);
	static void reconstructImg(SampleElem* data);

	// Getters and Setters
	static SampleElem getPosition(size_t x, size_t y, size_t k, OFFSET offset);
	static SampleElem getPosition(size_t index);
	static SampleElem* getPosition();
	static SampleElem getColor(size_t x, size_t y, size_t k, OFFSET offset);
	static SampleElem getColor(size_t index);
	static SampleElem* getColor();
	static SampleElem getFeature(size_t x, size_t y, size_t k, OFFSET offset);
	static SampleElem getFeature(size_t index);
	static SampleElem* getFeature();
	static SampleElem getRandomParameter(size_t x, size_t y, size_t k, OFFSET offset);
	static SampleElem getRandomParameter(size_t index);
	static SampleElem* getRandomParameter();
	static size_t getWidth();
	static size_t getHeight();
	static size_t getSamplesPerPixel();
	static size_t getNumOfSamples();

	static void setPosition(size_t x, size_t y, size_t k, SampleElem position, OFFSET offset);
	static void setPosition(size_t x, size_t y, size_t k, SampleElem* positions);
	static void setPosition(size_t index, SampleElem position);
	static void setPosition(size_t index, SampleElem* positions);
	static void setColor(size_t x, size_t y, size_t k, SampleElem color, OFFSET offset);
	static void setColor(size_t x, size_t y, size_t k, SampleElem* colors);
	static void setColor(size_t index, SampleElem color);
	static void setColor(size_t index, SampleElem* colors);
	static void setFeature(size_t x, size_t y, size_t k, SampleElem feature, OFFSET offset);
	static void setFeature(size_t x, size_t y, size_t k, SampleElem* features, OFFSET offset, size_t size);
	static void setFeature(size_t index, SampleElem feature);
	static void setFeature(size_t index, SampleElem* features, size_t size);
	static void setRandomParameter(size_t x, size_t y, size_t k, SampleElem randomParameter, OFFSET offset);
	static void setRandomParameter(size_t x, size_t y, size_t k, SampleElem* randomParameters, OFFSET offset, size_t size);
	static void setRandomParameter(size_t index, SampleElem randomParameter);
	static void setRandomParameter(size_t index, SampleElem* randomParameters, size_t size);

	static void setWidth(size_t width);
	static void setHeight(size_t height);
	static void setSamplesPerPixel(size_t samplesPerPixel);

private:

	SampleWriter();
	~SampleWriter();

	static void checkData();
	static size_t getIndex(size_t x, size_t y, size_t k);
	static void orderSamples(SampleElem* data);
	static bool allEqualToZero(SampleElem* data, size_t size);
	static FILE* openDatFile(char* fileName, char* mode);
	static SampleElem tonemap(SampleElem val); 
	static void saveImg(char* fileName, CImg<SampleElem>* img, bool shouldTonemap);
	static void dumpRpfData(char* inputFolder, char* name, SampleElem* fullData, size_t sampleLength, 
							int posCount, int colorCount, int featureCount, int randomCount);
	static size_t removeExtraData(SampleElem*& curData, int& posCount, int& colorCount, int& featureCount, int& randomCount);

	static SampleElem* positions;
	static SampleElem* colors;
	static SampleElem* features;
	static SampleElem* randomParameters;
	static size_t width;
	static size_t height;
	static size_t spp;
	static size_t numOfSamples;
	static bool isInitialized;


};

#endif
