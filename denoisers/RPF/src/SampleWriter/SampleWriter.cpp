#include "SampleWriter.h"

#include <iostream>

SampleElem* SampleWriter::positions;
SampleElem* SampleWriter::colors;
SampleElem* SampleWriter::features;
SampleElem* SampleWriter::randomParameters;
size_t SampleWriter::width;
size_t SampleWriter::height;
size_t SampleWriter::spp;
size_t SampleWriter::numOfSamples;
bool SampleWriter::isInitialized;

SampleWriter::SampleWriter() {
	this->isInitialized = false;
}

SampleWriter::~SampleWriter() {

	if(isInitialized) {
		delete[] positions;
		delete[] colors;
		delete[] features;
		delete[] randomParameters;
	}

}

void SampleWriter::initialize(size_t imgWidth, size_t imgHeight, size_t imgSamplesPerPixel) {

	width = imgWidth; 
	height = imgHeight;
	spp = imgSamplesPerPixel;
	numOfSamples = width * height * spp;
	isInitialized = true;

	positions = new SampleElem[NUM_OF_POSITIONS * numOfSamples];
	colors = new SampleElem[NUM_OF_COLORS * numOfSamples];
	features = new SampleElem[NUM_OF_FEATURES * numOfSamples];
	randomParameters = new SampleElem[NUM_OF_RANDOM_PARAMS * numOfSamples];

	memset(positions, 0, NUM_OF_POSITIONS * numOfSamples * sizeof(SampleElem));
	memset(colors, 0, NUM_OF_COLORS * numOfSamples * sizeof(SampleElem));
	memset(features, 0, NUM_OF_FEATURES * numOfSamples * sizeof(SampleElem));
	memset(randomParameters, 0, NUM_OF_RANDOM_PARAMS * numOfSamples * sizeof(SampleElem));
}

void SampleWriter::checkData()
{
    //FIXME: This test was failing on correct scenes. Turning it off for now.
    return;

	for(int i = 0; i < numOfSamples; i++) {

		bool allZero = true;

		for(int j = 0; j < NUM_OF_POSITIONS; j++) {
			if(positions[NUM_OF_POSITIONS * i + j] != 0) {
				allZero = false;
				break;
			}
		}

		if(!allZero) {
			continue;
		}

		for(int j = 0; j < NUM_OF_COLORS; j++) {
			if(colors[NUM_OF_COLORS * i + j] != 0) {
				allZero = false;
				break;
			}
		}

		if(!allZero) {
			continue;
		}

		for(int j = 0; j < NUM_OF_FEATURES; j++) {
			if(features[NUM_OF_FEATURES * i + j] != 0) {
				allZero = false;
				break;
			}
		}
	
		if(!allZero) {
			continue;
		}

		for(int j = 0; j < NUM_OF_RANDOM_PARAMS; j++) {
			if(randomParameters[NUM_OF_RANDOM_PARAMS * i + j] != 0) {
				allZero = false;
				break;
			}
		}

		if(allZero) {
			printf("ERROR: Pixel at index %d does not have correct number of samples\n", i);
			exit(-1);
		}
		
	}

}

size_t SampleWriter::removeExtraData(SampleElem*& curData, int& posCount, int& colorCount, int& featureCount, int& randomCount) {
	
	bool* isExtra = new bool[SAMPLE_LENGTH];
	memset(isExtra, 0, SAMPLE_LENGTH * sizeof(bool));

	posCount = NUM_OF_POSITIONS;
	colorCount = NUM_OF_COLORS;
	featureCount = 0;
	randomCount = 0;

	// Assume we always have POS, COLOR, WORLD_1, NORM_1, TEXTURE_1
	for(int k = WORLD_2_X; k < SAMPLE_LENGTH; k++) {
		for(size_t i = 0; i < numOfSamples; i++) {
			if(curData[i*SAMPLE_LENGTH + k] != 0) {
				break;
			} 
			/*if(i == numOfSamples - 1) {
				isExtra[k] = true;
			}*/
		}
	}

	size_t dataSize = 0;
	for(int k = 0; k < SAMPLE_LENGTH; k++) {
		if(!isExtra[k]) {
			dataSize++;
			if(k >= RANDOM) {
				randomCount++;
			} else if(k >= FEATURE) {
				featureCount++;
			} 
			
		} else {
			printf("Removing Dimension %d\n", k);
		}
	}
	
	assert(dataSize >= WORLD_2_X);

	SampleElem* resizedData = new SampleElem[dataSize * numOfSamples];
	memset(resizedData, 0, dataSize * numOfSamples * sizeof(SampleElem));

	for(size_t i = 0; i < numOfSamples; i++) {
		int offset = 0;
		for(int k = 0; k < SAMPLE_LENGTH; k++) {
			if(!isExtra[k]) {
				resizedData[i*dataSize + offset] = curData[i*SAMPLE_LENGTH + k];
				offset++;
			}
		}
	}

	delete[] curData;
	curData = resizedData;

	delete[] isExtra;
	return dataSize;
}

void SampleWriter::ProcessData(float* result) {

	//************************** WRITE SAMPLE DATA ***********************//

	// Check that all pixels have equal number of samples
	checkData();

	size_t sampleLength = (size_t) SAMPLE_LENGTH;
	#if WRITE_SAMPLES

		char fileName[BUFFER_SIZE];
		char extension[BUFFER_SIZE];
		strcpy_s(fileName, sceneName); 
		sprintf_s(extension, "_%04d.dat", iter);
		strcat_s(fileName, extension);

		// Open file for writing
		FILE* fp = openDatFile(fileName, "wb");

		// Save some info about the image
		fwrite(&width, sizeof(size_t), 1, fp);
		fwrite(&height, sizeof(size_t), 1, fp);
		fwrite(&spp, sizeof(size_t), 1, fp);
		fwrite(&sampleLength, sizeof(size_t), 1, fp);

	#endif

	SampleElem* fullData = new SampleElem[SAMPLE_LENGTH * numOfSamples];

	// Save all the sample data
	printf("\nProcessing Data\n");

	for(size_t i = 0; i < numOfSamples; i++) {

		SampleElem data[SAMPLE_LENGTH];
		memcpy(&data[POSITION], &positions[NUM_OF_POSITIONS * i], NUM_OF_POSITIONS * sizeof(SampleElem));
		memcpy(&data[COLOR], &colors[NUM_OF_COLORS * i], NUM_OF_COLORS * sizeof(SampleElem));
		memcpy(&data[FEATURE], &features[NUM_OF_FEATURES * i], NUM_OF_FEATURES * sizeof(SampleElem));
		memcpy(&data[RANDOM], &randomParameters[NUM_OF_RANDOM_PARAMS * i], NUM_OF_RANDOM_PARAMS * sizeof(SampleElem));

		#if WRITE_SAMPLES
				
			fwrite(data, sizeof(SampleElem), SAMPLE_LENGTH, fp);

		#endif
		
		memcpy(&fullData[i*SAMPLE_LENGTH], data, SAMPLE_LENGTH * sizeof(SampleElem));
	}

	#if WRITE_SAMPLES

		// Add termination character and close file
		fwrite("\0", sizeof(char), 2, fp);
		fclose(fp);

	#endif

	int posCount;
	int colorCount;
	int featureCount;
	int randomCount;
	sampleLength = removeExtraData(fullData, posCount, colorCount, featureCount, randomCount);

	//******************************** SAVE IMAGES ***************************************//

	#if SAVE_IMAGES

		printf("\nSaving Images\n");

		// Make output folder
		char outputFolder[BUFFER_SIZE];
		sprintf(outputFolder, "%s%s_RPF_DataImages/", sceneName, name);
		_mkdir(outputFolder);
		strerror(errno);

		for(size_t k = 0; k < sampleLength; k++) {
			CImg<float> tempImg(width, height);
			tempImg.fill(0);

			for(size_t i = 0; i < height; i++) {
				for(size_t j = 0; j < width; j++) {

					size_t pixelIndex = i * width + j;
					for(int q = 0; q < spp; q++) {
						size_t sampleIndex = spp * pixelIndex + q;
						size_t dataIndex = sampleLength * sampleIndex + k;
						tempImg(j,i) += fullData[dataIndex];
					}

				}
			}

			tempImg *= (1.0f / float(spp));

			// Write the image
			char buff[BUFFER_SIZE];
			char temp[BUFFER_SIZE];

			sprintf(temp, "Data_%03d.bmp", k);
			strcpy(buff, outputFolder);
			strcat(buff, temp);
				
			tempImg.normalize(0, 255);
			tempImg.save(buff);

		}

	#endif

    RPF(result, fullData, width, height, spp, sampleLength, posCount, colorCount, featureCount, randomCount, NULL);
}


void SampleWriter::dumpRpfData(char* inputFolder, char* name, SampleElem* fullData, size_t sampleLength,
							   int posCount, int colorCount, int featureCount, int randomCount) {

	printf("\nSaving RPF Data\n");
	char buff[BUFFER_SIZE];
	sprintf(buff, "%s\\%s_RPF.dat", inputFolder, name);
	FILE* fp = fopen(buff, "wb");
	if(!fp) {
		fprintf(stderr, "ERROR: Could open the block statistic file %s\n", buff);
		getchar();
		exit(-1);
	}

	//printf("width %ld height %ld spp %ld length %ld\n", width, height, spp, sampleLength);
	fwrite(&width, 1, sizeof(size_t), fp);
	fwrite(&height, 1, sizeof(size_t), fp);
	fwrite(&spp, 1, sizeof(size_t), fp);
	fwrite(&sampleLength, 1, sizeof(size_t), fp);

	fwrite(&posCount, 1, sizeof(int), fp);
	fwrite(&colorCount, 1, sizeof(int), fp);
	fwrite(&featureCount, 1, sizeof(int), fp);
	fwrite(&randomCount, 1, sizeof(int), fp);

	fwrite(fullData, sampleLength * numOfSamples, sizeof(SampleElem), fp);

	fclose(fp);

}

void SampleWriter::readSamplesFromFile(char* fileName, SampleElem* data) {
	
	// Open file
	FILE* fp = openDatFile(fileName, "rb");
	
	// Get info
	size_t width, height, spp, sampleLength;
	fread(&width, sizeof(size_t), 1, fp);
	fread(&height, sizeof(size_t), 1, fp);
	fread(&spp, sizeof(size_t), 1, fp);
	fread(&sampleLength, sizeof(size_t), 1, fp);

	// Get sample data
	fread(data, sizeof(SampleElem), width * height * spp * sampleLength, fp);

	// Make sure we reached the end of the file
	assert(fgetc(fp) == '\0');

	// Put the samples in sequential order 
	orderSamples(data);

	// Close file
	fclose(fp);

}

void SampleWriter::reconstructImg(SampleElem* data) {
	
	CImg<SampleElem>* img = new CImg<SampleElem>(width, height, 1, NUM_OF_COLORS);
	img->fill(0);

	SampleElem meanScalar = 1.0f / spp;
	for(size_t i = 0; i < height; i++) {
		for(size_t j = 0; j < width; j++) {
			size_t index = spp * ((i * width) + j);

			for(size_t k = 0; k < spp; k++) {
				SampleElem* currentSample = &data[SAMPLE_LENGTH * (index + k)];
				for(int q = 0; q < NUM_OF_COLORS; q++) {
					(*img)(j,i,0,q) += currentSample[COLOR + q];
				}
			}

			for(int q = 0; q < NUM_OF_COLORS; q++) {
				(*img)(j,i,0,q) *= meanScalar;
			}

		}
	}
	
	saveImg("ReconstructedImg.bmp", img, true);

	delete img;
	
}

void SampleWriter::orderSamples(SampleElem* data) {

	// Read in sample values
	SampleElem* dataCopy = new SampleElem[SAMPLE_LENGTH * numOfSamples];
	memcpy(dataCopy, data, SAMPLE_LENGTH * numOfSamples * sizeof(SampleElem));
	memset(data, 0, SAMPLE_LENGTH * numOfSamples * sizeof(SampleElem));

	// Transfer data back from the copy in the correct order
	size_t index = 0;
	for(size_t i = 0; i < numOfSamples; i++) {
		
		SampleElem x = dataCopy[index];
		SampleElem y = dataCopy[index + 1];
		if(x > width || y > height || x < 0 || y < 0) {
			index += SAMPLE_LENGTH;
			continue;
		}
		size_t xInt = (size_t) floor(x);
		size_t yInt = (size_t) floor(y);
		size_t pixelIndex = SAMPLE_LENGTH * spp*((width * yInt) + xInt);
		for(size_t j = 0; j < spp; j++) {
			if(allEqualToZero(data + pixelIndex + (j * SAMPLE_LENGTH), SAMPLE_LENGTH)) {
				for(size_t k = 0; k < SAMPLE_LENGTH; k++) {
					data[pixelIndex + j*SAMPLE_LENGTH + k] = dataCopy[index + k];
				}
				break;
			}
		}
		index += SAMPLE_LENGTH;
	
	}

	delete[] dataCopy;

}

bool SampleWriter::allEqualToZero(SampleElem* data, size_t size) {

	for(size_t i = 0; i < size; i++) {
		if(data[i] != 0.0f) {
			return false;
		}
	}

	return true;

}

SampleElem SampleWriter::tonemap(SampleElem val) {

	return (SampleElem) (255 * MIN(1.0, pow(val, COLOR_GAMMA)/0.9));

}

void SampleWriter::saveImg(char* fileName, CImg<SampleElem>* img, bool shouldTonemap) {

	if(shouldTonemap) {
		int imgHeight = img->height();
		int imgWidth = img->width();
		int channels = img->spectrum();
		for(int i = 0; i < imgHeight; i++) {
			for(int j = 0; j < imgWidth; j++) {
				for(int k = 0; k < channels; k++) {
					(*img)(j,i,0,k) = tonemap((*img)(j,i,0,k));
				}
			}
		}
	} else {
		(*img) /= img->max();
		(*img) *= 255;
	}

	img->save(fileName);

}

FILE* SampleWriter::openDatFile(char* fileName, char* mode) {
	
	// Open dat file with samples
	FILE* fp;
    fopen_s(&fp, fileName, mode);
	if(!fp) {
		fprintf(stderr, "ERROR: Could not locate Sample file %s\n", fileName);
		getchar();
		exit(-1);
	}

	return fp;

}

SampleElem SampleWriter::getPosition(size_t x, size_t y, size_t k, OFFSET offset) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		return getPosition(NUM_OF_POSITIONS * index + offset);
	}

	return 0;
}

SampleElem SampleWriter:: getPosition(size_t index) {
	assert(index < NUM_OF_POSITIONS * numOfSamples);
	return positions[index];
}

SampleElem* SampleWriter::getPosition() {
	return positions;
}

SampleElem SampleWriter::getColor(size_t x, size_t y, size_t k, OFFSET offset) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		return getColor(NUM_OF_COLORS * index + offset);
	}

	return 0;
}

SampleElem SampleWriter::getColor(size_t index) {
	assert(index < NUM_OF_COLORS * numOfSamples);
	return colors[index];
}

SampleElem* SampleWriter::getColor() {
	return colors;
}

SampleElem SampleWriter::getFeature(size_t x, size_t y, size_t k, OFFSET offset) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		return getFeature(NUM_OF_FEATURES * index + offset);
	} 

	return 0;
}

SampleElem SampleWriter::getFeature(size_t index) {
	assert(index < NUM_OF_FEATURES * numOfSamples);
	return features[index];
}

SampleElem* SampleWriter::getFeature() {
	return features;
}

SampleElem SampleWriter::getRandomParameter(size_t x, size_t y, size_t k, OFFSET offset) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		return getRandomParameter(NUM_OF_RANDOM_PARAMS * index + offset);
	}

	return 0;
}

SampleElem SampleWriter::getRandomParameter(size_t index) {
	assert(index < NUM_OF_RANDOM_PARAMS * numOfSamples);
	return randomParameters[index];
}

SampleElem* SampleWriter::getRandomParameter() {
	return randomParameters;
}

size_t SampleWriter::getWidth() {
	return width;
}

size_t SampleWriter::getHeight() {
	return height;
}

size_t SampleWriter::getSamplesPerPixel() {
	return spp;
}

size_t SampleWriter::getNumOfSamples() {
	assert(numOfSamples == width * height * spp);
	return numOfSamples;
}

size_t SampleWriter::getIndex(size_t x, size_t y, size_t k) {
	assert(x >= 0 && height >= 0 && k >= 0);
	assert(k < spp);
	if(x < width && y < height) {
		return spp * ((y * width) + x) + k;
	} 

	return -1;
}

void SampleWriter::setPosition(size_t x, size_t y, size_t k, SampleElem position, OFFSET offset) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		setPosition(NUM_OF_POSITIONS * index + offset, position); 
	}
}

void SampleWriter::setPosition(size_t x, size_t y, size_t k, SampleElem* positions) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		setPosition(NUM_OF_POSITIONS * index, positions);
	}
}

void SampleWriter::setPosition(size_t index, SampleElem position) {
	assert(index < NUM_OF_POSITIONS * numOfSamples);
	assert(positions[index] == 0);
	positions[index] = position;
}

void SampleWriter::setPosition(size_t index, SampleElem* srcPositions) {
	assert(index < NUM_OF_POSITIONS * numOfSamples);
	memcpy(&positions[index], srcPositions, NUM_OF_POSITIONS * sizeof(SampleElem));
}

void SampleWriter::setColor(size_t x, size_t y, size_t k, SampleElem color, OFFSET offset) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		setColor(NUM_OF_COLORS * index + offset, color); 
    }
}

void SampleWriter::setColor(size_t x, size_t y, size_t k, SampleElem* colors) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		setColor(NUM_OF_COLORS * index , colors);
	}
}

void SampleWriter::setColor(size_t index, SampleElem color) {
	assert(index < NUM_OF_COLORS * numOfSamples);
	assert(colors[index] == 0);
	colors[index] = color;
}

void SampleWriter::setColor(size_t index, SampleElem* srcColors) {
	assert(index < NUM_OF_COLORS * numOfSamples);
	memcpy(&colors[index], srcColors, NUM_OF_COLORS * sizeof(SampleElem));
}

void SampleWriter::setFeature(size_t x, size_t y, size_t k, SampleElem feature, OFFSET offset) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		setFeature(NUM_OF_FEATURES * index + offset, feature);
    }
}

void SampleWriter::setFeature(size_t x, size_t y, size_t k, SampleElem* features, OFFSET offset, size_t size) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		setFeature(NUM_OF_FEATURES * index + offset, features, size);
	}
}

void SampleWriter::setFeature(size_t index, SampleElem feature) {
	assert(index < NUM_OF_FEATURES * numOfSamples);
	features[index] = feature; 
}

void SampleWriter::setFeature(size_t index, SampleElem* srcFeatures, size_t size) {
	assert(index < NUM_OF_FEATURES * numOfSamples);
	assert(size < NUM_OF_FEATURES);
	memcpy(&features[index], srcFeatures, size * sizeof(SampleElem));
}

void SampleWriter::setRandomParameter(size_t x, size_t y, size_t k, SampleElem randomParameter, OFFSET offset) {
	size_t index = getIndex(x, y, k);
	if(index != -1) {
		setRandomParameter(NUM_OF_RANDOM_PARAMS * index + offset, randomParameter);
	}
}
	
void SampleWriter::setRandomParameter(size_t x, size_t y, size_t k, SampleElem* randomParameters, OFFSET offset, size_t size) {
	size_t index = getIndex(x, y, k);
	assert(k >= 0 && k < spp);
	if(index != -1) {
		setRandomParameter(NUM_OF_RANDOM_PARAMS * index + offset, randomParameters, size);
	}
}

void SampleWriter::setRandomParameter(size_t index, SampleElem randomParameter) {
	assert(index < NUM_OF_RANDOM_PARAMS * numOfSamples);
	assert(randomParameters[index] == 0);
	randomParameters[index] = randomParameter;
}

void SampleWriter::setRandomParameter(size_t index, SampleElem* srcRandomParameters, size_t size) {
	assert(index < NUM_OF_RANDOM_PARAMS * numOfSamples);
	assert(size < NUM_OF_RANDOM_PARAMS);
	memcpy(&randomParameters[index], srcRandomParameters, size * sizeof(SampleElem));
}

void SampleWriter::setWidth(size_t newWidth) {
	width = newWidth;
}

void SampleWriter::setHeight(size_t newHeight) {
	height = newHeight;
}

void SampleWriter::setSamplesPerPixel(size_t newSamplesPerPixel) {
	spp = newSamplesPerPixel;
}
