#include "SampleWriter.h"
#include <cassert>

int SampleWriter::numOfBuffers;
int* SampleWriter::featureInd;
int* SampleWriter::pixelSampleCount;
size_t SampleWriter::width;
size_t SampleWriter::height;
size_t SampleWriter::spp;
size_t SampleWriter::numOfSamples;
SampleElem* SampleWriter::pixelData;
SampleElem* SampleWriter::texture2Data;
SampleElem* SampleWriter::sampleMean;
SampleElem* SampleWriter::varData;
SampleElem** SampleWriter::pixelBuffer;

void SampleWriter::Initialize(size_t imgWidth, size_t imgHeight, size_t imgSamplesPerPixel) {

	width = imgWidth; 
	height = imgHeight;
	spp = imgSamplesPerPixel;
	numOfSamples = width * height;

	// Data initialization
	pixelData = new SampleElem[SAMPLE_LENGTH * numOfSamples];
	sampleMean = new SampleElem[SAMPLE_LENGTH * numOfSamples];
	varData = new SampleElem[SAMPLE_LENGTH * numOfSamples];
	texture2Data = new SampleElem[NUM_OF_TEXTURE_2 * numOfSamples * spp];
	memset(pixelData, 0, SAMPLE_LENGTH * numOfSamples * sizeof(SampleElem));
	memset(sampleMean, 0, SAMPLE_LENGTH * numOfSamples * sizeof(SampleElem));
	memset(varData, 0, SAMPLE_LENGTH * numOfSamples * sizeof(SampleElem));
	memset(texture2Data, 0, NUM_OF_TEXTURE_2 * numOfSamples * spp * sizeof(SampleElem));

	numOfBuffers = 2;
	pixelBuffer = new SampleElem*[numOfBuffers];
	for(int i = 0; i < numOfBuffers; i++) {
		pixelBuffer[i] = new SampleElem[SAMPLE_LENGTH * numOfSamples];
		memset(pixelBuffer[i], 0, SAMPLE_LENGTH * numOfSamples * sizeof(SampleElem));
	}

	// Number of samples processed for each pixel
	pixelSampleCount = new int[numOfSamples];
	memset(pixelSampleCount, 0, numOfSamples * sizeof(int));

	featureInd = new int[numOfSamples];
	memset(featureInd, 0, numOfSamples * sizeof(int));

}

void SampleWriter::GenerateBufferIndex(int& bufInd, float& bufNormFactor, int n) {

	bufInd = ((n-1) >= (spp/numOfBuffers));
	bufNormFactor = ((n-1) % (spp/numOfBuffers)) + 1;
	assert(bufNormFactor > 0 && bufNormFactor <= (spp/numOfBuffers));
	assert(bufInd >= 0 && bufInd < numOfBuffers);

}

void SampleWriter::ProcessTexture2Data() {

	assert(numOfBuffers >= 1);
	int imgSize = width * height;
	for(size_t i = 0; i < numOfSamples; i++) {

		int pixelTex2Index = SAMPLE_LENGTH * i + FEATURE + TEXTURE_2_X_OFFSET;
		memset(&pixelData[pixelTex2Index], 0, NUM_OF_TEXTURE_2 * sizeof(SampleElem));
		memset(&sampleMean[pixelTex2Index], 0, NUM_OF_TEXTURE_2 * sizeof(SampleElem));
		memset(&varData[pixelTex2Index], 0, NUM_OF_TEXTURE_2 * sizeof(SampleElem));

		// Update mean and variance (using online algorithm)
		for(int q = 0; q < spp; q++) {

			int tex2Index = NUM_OF_TEXTURE_2 * spp * i + q*NUM_OF_TEXTURE_2;
			for(int k = 0; k < NUM_OF_TEXTURE_2; k++) {

				SampleElem meanDelta = texture2Data[tex2Index + k] - pixelData[pixelTex2Index + k];
				pixelData[pixelTex2Index + k] += meanDelta / (q + 1);
				
				int bufInd;
				float bufNormFactor;
				GenerateBufferIndex(bufInd, bufNormFactor, q + 1);
				int bufferTex2Ind = i + (k + TEXTURE_2_X)*imgSize;
				if(bufNormFactor == 1.0f) {
					pixelBuffer[bufInd][bufferTex2Ind] = 0;
				}
				meanDelta = texture2Data[tex2Index + k] - pixelBuffer[bufInd][bufferTex2Ind];
				pixelBuffer[bufInd][bufferTex2Ind] += meanDelta / bufNormFactor;

				SampleElem varDelta = texture2Data[tex2Index + k] - sampleMean[pixelTex2Index + k];
				sampleMean[pixelTex2Index + k] += varDelta / (q + 1);
				varData[pixelTex2Index + k] += varDelta * (texture2Data[tex2Index + k] - sampleMean[pixelTex2Index + k]);

			}

		}
	}

}

//void SampleWriter::SaveRenderTime(char* fileName, char* inputFolder, char* sceneName) {
 
//	string fileNameStr = string(fileName);
//	int lastslash = fileNameStr.find_last_of("\\");
//	int lastbackslash = fileNameStr.find_last_of("/");
//	lastslash = (lastslash > lastbackslash) ? lastslash : lastbackslash;
//	if (lastslash != std::string::npos){
//		std::string tempFile = fileNameStr.substr(0, lastslash);
//		std::string tempName = fileNameStr.substr(lastslash + 1, std::string::npos);
//		strcpy(inputFolder, tempFile.c_str());
//		strcpy(sceneName, tempName.c_str());
//	} else {
//		sprintf(inputFolder, ".");
//		strcpy(sceneName, fileNameStr.c_str());
//	}

//	char tmpFileName[1000];
//	sprintf(tmpFileName, "%s\\%s_timing.txt", inputFolder, sceneName);
//	FILE* fp;
//	fopen_s(&fp, tmpFileName, "wt");

//	if(!fp) {
//		fprintf(stderr, "Could not open file %s\n", tmpFileName);
//		getchar();
//		exit(-1);
//	}

//	fprintf(fp, "Render Time: %f sec\n", timer.Time());
//	fclose(fp);
//	timer.Reset();
//	timer.Start();
//}

void SampleWriter::ProcessData(float* img) {
	
	// Record time
	char inputFolder[BUFFER_SIZE]; 
	char sceneName[BUFFER_SIZE];
//	SaveRenderTime(fileName, inputFolder, sceneName, timer);

	// Update texture 2 values
	ProcessTexture2Data();
	delete[] texture2Data;
	delete[] sampleMean;
	delete[] pixelSampleCount;
	delete[] featureInd;

	// Run feature extractor and nn/filter
    LBF(inputFolder, sceneName, pixelData, varData, pixelBuffer, width, height, spp, img);
}

bool SampleWriter::ShouldSaveFeature(size_t x, size_t y, size_t k) {
	if(x < 0 || x >= width || y < 0 || y >= height) {
		return false;
	}
	return (featureInd[x + y*width] == k);
}


//********************************* GETTERS/SETTERS ***************************************//


SampleElem SampleWriter::GetFeature(size_t x, size_t y, size_t k, OFFSET offset) {
	bool saveSample;
	size_t index = GetIndex(x, y, k, saveSample);
	if(index != -1) {
		bool isTexture2 = (offset >= TEXTURE_2_X_OFFSET && offset <= TEXTURE_2_Z_OFFSET);
		size_t featOffset;
		if(!isTexture2) {
			featOffset = SAMPLE_LENGTH*index + FEATURE + offset;
		} else {
			featOffset = NUM_OF_TEXTURE_2*index*spp + k*NUM_OF_TEXTURE_2 + (offset - TEXTURE_2_X_OFFSET);
		}
		return GetFeature(featOffset, isTexture2);
	} 

	return 0;
}

SampleElem SampleWriter::GetFeature(size_t index, bool isTexture2) {
	if(!isTexture2) {
		assert(index < SAMPLE_LENGTH * numOfSamples);
		return pixelData[index];
	} else {
		assert(index < NUM_OF_TEXTURE_2 * numOfSamples * spp);
		return texture2Data[index];
	}
}

size_t SampleWriter::GetWidth() {
	return width;
}

size_t SampleWriter::GetHeight() {
	return height;
}

size_t SampleWriter::GetSamplesPerPixel() {
	return spp;
}

size_t SampleWriter::GetNumOfSamples() {
	assert(numOfSamples == width * height * spp);
	return numOfSamples;
}

size_t SampleWriter::GetIndex(size_t x, size_t y, size_t& k, bool& saveSample) {

	saveSample = false;

	if(x >= 0 && x < width && y >= 0 && y < height && height >= 0 && width >= 0 && k >= 0 && k < spp) {
		size_t index = x + y*width;
		pixelSampleCount[index] = MAX(k + 1, pixelSampleCount[index]);
		assert(index < width*height);
		return index;
	}

	return -1;
}

int SampleWriter::GetPixelCount(size_t x, size_t y) {

	if(x < 0 || x >= width || y < 0 || y >= height) {
		return 0;
	}

	size_t index = x + y * width;
	assert(index >= 0 && index < numOfSamples);
	return pixelSampleCount[index];
}

void SampleWriter::SetPosition(size_t x, size_t y, size_t k, SampleElem position, OFFSET offset) {

	bool saveSample;
	size_t index = GetIndex(x, y, k, saveSample);
	
	if(index != -1) {
		SetPosition(SAMPLE_LENGTH * index + POSITION + offset, position, saveSample, pixelSampleCount[index], x, y, offset);
	}

}

void SampleWriter::SetPosition(size_t index, SampleElem position, bool saveSample, int n, size_t x, size_t y, OFFSET offset) {

	assert(index < SAMPLE_LENGTH * numOfSamples);
	assert(n != 0);
	
	// Update mean and variance (using online algorithm)
	SampleElem meanDelta = position - pixelData[index];
	pixelData[index] += meanDelta / n;

	int bufInd;
	float bufNormFactor;
	GenerateBufferIndex(bufInd, bufNormFactor, n);
	size_t bufPosInd = x + y * width + (POSITION + offset) * width * height;
	meanDelta = position - pixelBuffer[bufInd][bufPosInd];
	pixelBuffer[bufInd][bufPosInd] += meanDelta / bufNormFactor;

	SampleElem varDelta = position - sampleMean[index];
	sampleMean[index] += varDelta / n;
	varData[index] += varDelta * (position - sampleMean[index]);

}

void SampleWriter::SetColor(size_t x, size_t y, size_t k, SampleElem color, OFFSET offset) {

	bool saveSample;
	size_t index = GetIndex(x, y, k, saveSample);
	
	if(index != -1) {

		SetColor(SAMPLE_LENGTH * index + COLOR + offset, color, saveSample, pixelSampleCount[index], x, y, offset);

		int colorIndex = spp * NUM_OF_COLORS * index + k*NUM_OF_COLORS + offset;
		assert(colorIndex < (NUM_OF_COLORS * width * height * spp));

	}

}

void SampleWriter::SetColor(size_t index, SampleElem color, bool saveSample, int n, size_t x, size_t y, OFFSET offset) {

	assert(index < SAMPLE_LENGTH * numOfSamples);
	assert(n != 0);
	
	// Update mean and variance (using online algorithm)
	SampleElem meanDelta = color - pixelData[index];
	pixelData[index] += meanDelta / n;

	int bufInd;
	float bufNormFactor;
	GenerateBufferIndex(bufInd, bufNormFactor, n);
	size_t bufColorInd = x + y * width + (COLOR + offset) * width * height;
	meanDelta = color - pixelBuffer[bufInd][bufColorInd];
	pixelBuffer[bufInd][bufColorInd] += meanDelta / bufNormFactor;

	SampleElem varDelta = color - sampleMean[index];
	sampleMean[index] += varDelta / n;
	varData[index] += varDelta * (color - sampleMean[index]);

}

void SampleWriter::SetFeature(size_t x, size_t y, size_t k, SampleElem feature, OFFSET offset) {

	bool saveSample;
	size_t index = GetIndex(x, y, k, saveSample);

	if(index != -1) {

		if(offset == TEXTURE_1_X_OFFSET) {
			featureInd[index]++;
		}

		bool isTexture2 = false;
		if(offset >= TEXTURE_2_X_OFFSET && offset <= TEXTURE_2_Z_OFFSET) {
			isTexture2 = true;
			texture2Data[NUM_OF_TEXTURE_2 * spp * index + k*NUM_OF_TEXTURE_2 + (offset - TEXTURE_2_X_OFFSET)] = feature;
		}

		SetFeature(SAMPLE_LENGTH * index + FEATURE + offset, feature, saveSample, pixelSampleCount[index], x, y, offset);

	}

}

void SampleWriter::SetFeature(size_t index, SampleElem feature, bool saveSample, int n, size_t x, size_t y, OFFSET offset) { 

	assert(index < SAMPLE_LENGTH * numOfSamples);
	assert(n != 0);
	
	// Update mean and variance (using online algorithm)
	SampleElem meanDelta = feature - pixelData[index];
	pixelData[index] += meanDelta / n;

	int bufInd;
	float bufNormFactor;
	GenerateBufferIndex(bufInd, bufNormFactor, n);
	size_t bufFeatInd = x + y * width + (FEATURE + offset) * width * height;
	meanDelta = feature - pixelBuffer[bufInd][bufFeatInd];
	pixelBuffer[bufInd][bufFeatInd] += meanDelta / bufNormFactor;

	SampleElem varDelta = feature - sampleMean[index];
	sampleMean[index] += varDelta / n;
	varData[index] += varDelta * (feature - sampleMean[index]);

}

void SampleWriter::SetWidth(size_t newWidth) {
	width = newWidth;
	numOfSamples = width * height;
}

void SampleWriter::SetHeight(size_t newHeight) {
	height = newHeight;
	numOfSamples = width * height;
}

void SampleWriter::SetSamplesPerPixel(size_t newSamplesPerPixel) {
	spp = newSamplesPerPixel;
}
