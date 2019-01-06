#include "SampleSet.h"

#define _aligned_malloc(size, alig) \
    malloc(size)

#define _aligned_free(block) \
    free(block)

// Main set of all samples in the image
extern SampleSet* samples; 

SampleSet::SampleSet(size_t size, bool initializeData) { 

	// Initialize fields
	totalSize = size;
	numOfSamples = 0;
	randomValuesInitialized = false;
	
	// Every SampleSet has a cPrime vector that corresponds to the colors to filter for this iteration
	cPrime = (float*) _aligned_malloc(NUM_OF_COLORS * totalSize * sizeof(float), ALIGNMENT);
	memset(cPrime, 0, NUM_OF_COLORS * totalSize *sizeof(float));

	// The small blocks (not the main "samples" block that carries all of the samples) will initialize this data
	if(initializeData) {

		// Allocate memory for all of the member arrays

		//***** DATA/WEIGHTS *****//
		data = (float*) _aligned_malloc(sampleLength * totalSize * sizeof(float), ALIGNMENT);
		pixelColors = (float*) _aligned_malloc(NUM_OF_COLORS * samplesPerPixel * sizeof(float), ALIGNMENT);	
		alpha = new float[NUM_OF_COLORS];
		beta = new float[numOfFeat];

		memset(data, 0, sampleLength * totalSize *sizeof(float));
		memset(pixelColors, 0, NUM_OF_COLORS * samplesPerPixel * sizeof(float));
		memset(alpha, 0, NUM_OF_COLORS * sizeof(float));
		memset(beta, 0, numOfFeat * sizeof(float));


		//***** STATISTICS *****//
		pixelFeatureMu = new float[numOfFeat];
		pixelFeatureSigma = new float[numOfFeat];
		dataMu = (float*) _aligned_malloc(sampleLength * sizeof(float), ALIGNMENT);
		dataSigma = (float*) _aligned_malloc(sampleLength * sizeof(float), ALIGNMENT);
		normalizedData = (float*) _aligned_malloc(sampleLength * totalSize * sizeof(float), ALIGNMENT);

		memset(pixelFeatureMu, 0, numOfFeat * sizeof(float));
		memset(pixelFeatureSigma, 0, numOfFeat * sizeof(float));
		memset(dataMu, 0, sampleLength * sizeof(float));
		memset(dataSigma, 0, sampleLength * sizeof(float));
		memset(normalizedData, 0, sampleLength * totalSize * sizeof(float));


		//***** PROBABILITY DISTRIBUTIONS *****// 
		histograms = new int[sampleLength * totalSize]; 
		binSizes = new int[sampleLength];
		PDFs = (float*) _aligned_malloc(sampleLength * MAX_NUM_OF_BINS * sizeof(float), ALIGNMENT); // OPTIMIZATION: allocate pdfs on the fly to correct size?

		memset(histograms, 0, sampleLength * totalSize * sizeof(int));
		memset(binSizes, 0, sampleLength * sizeof(int));
		memset(PDFs, 0, sampleLength * MAX_NUM_OF_BINS * sizeof(float));
		

		//***** SSE BUFFERS *****// 
		normColorBuffer = (float*) _aligned_malloc(totalSize * sizeof(float), ALIGNMENT);
		normFeatureBuffer = (float*) _aligned_malloc(totalSize * sizeof(float), ALIGNMENT);
		colorBuffer = (float*) _aligned_malloc(totalSize * sizeof(float), ALIGNMENT);
		featureBuffer = (float*) _aligned_malloc(totalSize * sizeof(float), ALIGNMENT);
		tempStorageBuffer = (float*) _aligned_malloc(totalSize * sizeof(float), ALIGNMENT);
		weightsBuffer = (float*) _aligned_malloc(totalSize * sizeof(float), ALIGNMENT);
		tempWeightsBuffer = (float*) _aligned_malloc(totalSize * sizeof(float), ALIGNMENT);
		
		memset(normColorBuffer, 0, totalSize * sizeof(int));
		memset(normFeatureBuffer, 0, totalSize * sizeof(int));
		memset(colorBuffer, 0, totalSize * sizeof(int));
		memset(featureBuffer, 0, totalSize * sizeof(int));
		memset(tempStorageBuffer, 0, totalSize * sizeof(int));
		memset(weightsBuffer, 0, totalSize * sizeof(int));
		memset(tempWeightsBuffer, 0, totalSize * sizeof(int));


	// This is meant for the global "samples" block that holds all of the input samples
	} else {

		// Generate a 2d array where the first index corresponds to the sample and the second index
		// is an array into the sample (see globals.h for how the sample is laid out)
		sampleInfo = new float*[totalSize];
		for(size_t i = 0; i < totalSize; i++) {
			sampleInfo[i] = new float[sampleLength];
			memset(sampleInfo[i], 0, sampleLength * sizeof(float));
		}

		// Holds the filtered color values of the current iteration. At the end of the iteration it clobbers
		// the cPrime vector so that the values can be used for the following iteration
		cDoublePrime = (float*) _aligned_malloc(NUM_OF_COLORS * totalSize * sizeof(float), ALIGNMENT);
		memset(cDoublePrime, 0, NUM_OF_COLORS * totalSize *sizeof(float));
		
	}

	// Indicate which data has been initialized
	additionalDataInitialized = initializeData; 

}

SampleSet::~SampleSet() {
	
	// Clean up

	_aligned_free(cPrime);
	
	// If small block
	if(additionalDataInitialized) {

		//***** DATA/WEIGHTS *****//
		_aligned_free(data);
		_aligned_free(pixelColors);
		delete[] alpha;
		delete[] beta;

		
		//***** STATISTICS *****//
		delete[] pixelFeatureMu;
		delete[] pixelFeatureSigma;
		_aligned_free(dataMu);
		_aligned_free(dataSigma);
		_aligned_free(normalizedData);


		//***** PROBABILITY DISTRIBUTIONS *****//
		delete[] histograms;
		delete[] binSizes;
		_aligned_free(PDFs);
		
		//***** SSE BUFFERS *****//
		_aligned_free(normColorBuffer);
		_aligned_free(normFeatureBuffer);
		_aligned_free(colorBuffer);
		_aligned_free(featureBuffer);
		_aligned_free(tempStorageBuffer);
		_aligned_free(weightsBuffer);
		_aligned_free(tempWeightsBuffer);

	
	// If main sample block
	} else {

		// Clean up for main sample block
		for(size_t i = 0; i < totalSize; i++) {
			delete[] sampleInfo[i];
		}
		delete[] sampleInfo;

		_aligned_free(cDoublePrime);

	}

	if(randomValuesInitialized) {
		
		//***** RANDOM VALUES *****//
		delete[] randomIndexes;
		delete[] integerIndexes;

	}

}

void SampleSet::readSamples(float* pbrtData) {

	// Get sample data
	assert(sampleLength == sampleLength);
	
	// Put the samples in sequential order 
	clampColors(pbrtData);

	size_t dataOffset = 0;
	
	// Loop through the file and propagate the data
	for(size_t i = 0; i < totalSize; i++) {

		float* sampleInfoPtr = sampleInfo[i];

		// Get values from file in the following order
		
		// X_COORD			(Position start)
		// Y_COORD
		// COLOR_1			(Color start)
		// COLOR_2
		// COLOR_3
		// WORLD_1_X		(Feature start)
		// WORLD_1_Y
		// WORLD_1_Z
		// NORM_1_X
		// NORM_1_Y
		// NORM_1_Z
		// TEXTURE_X
		// TEXTURE_Y
		// TEXTURE_Z
		// TEXTURE_W 
		// WORLD_2_X
		// WORLD_2_Y
		// WORLD_2_Z
		// NORM_2_X
		// NORM_2_Y
		// NORM_2_Z
		// U_COORD			(Random start)
		// V_COORD
		// TIME
		// LIGHT_1
		// LIGHT_2
		// TIME

		size_t count = 0;
		size_t index = 0;

		for(size_t j = 0; j < sampleLength; j++) {
			
			sampleInfoPtr[index] = pbrtData[j + dataOffset];
			index++;

		}
		
		dataOffset += sampleLength;
		assert(index == sampleLength);

		// Update the number of samples we have
		numOfSamples++; 

	} 
}

void SampleSet::orderSamples(float* data) {

	// Read in sample values
	float* dataCopy = new float[sampleLength * width * height * samplesPerPixel];
	memcpy(dataCopy, data, sampleLength * width * height * samplesPerPixel * sizeof(float));
	memset(data, 0, sampleLength * width * height * samplesPerPixel * sizeof(float));

	// Transfer data back from the copy in the correct order
	size_t index = 0;
	for(size_t i = 0; i < width * height * samplesPerPixel; i++) {
		
		float x = dataCopy[index];
		float y = dataCopy[index + 1];

		if(x > width || y > height || x < 0 || y < 0) {
			index += sampleLength;
			continue;
		}
		size_t xInt = (size_t) floor(x);
		size_t yInt = (size_t) floor(y);

		size_t pixelIndex = sampleLength * samplesPerPixel*((width * yInt) + xInt);
		for(size_t j = 0; j < samplesPerPixel; j++) {
			if(allEqualToZero(data + pixelIndex + (j * sampleLength), sampleLength)) {
				for(size_t k = 0; k < sampleLength; k++) {
					if(k >= COLOR_1 && k <= COLOR_3) {
						data[pixelIndex + j*sampleLength + k] = MIN(dataCopy[index + k], MAX_SAMP_VAL);
					} else {
						data[pixelIndex + j*sampleLength + k] = dataCopy[index + k];
					}
				}
				break;
			}
			if(j == (samplesPerPixel - 1)) {
				assert(0);
			}
		}
		index += sampleLength;
	
	}

	delete[] dataCopy;

}


void SampleSet::clampColors(float* data) {

	size_t index = 0;
	for(size_t i = 0; i < width * height * samplesPerPixel; i++) {
		
		float x = data[index];
		float y = data[index + 1];

		if(x > width || y > height || x < 0 || y < 0) {
			index += sampleLength;
			continue;
		}
		size_t xInt = (int) floor(x);
		size_t yInt = (int) floor(y);

		size_t pixelIndex = size_t(sampleLength) * size_t(samplesPerPixel) * size_t((width * yInt) + xInt);
		for(size_t j = 0; j < samplesPerPixel; j++) {
			for(int k = COLOR_1; k <= COLOR_3; k++) {
				data[pixelIndex + j*sampleLength + k] = MIN(data[pixelIndex + j*sampleLength + k], MAX_SAMP_VAL);
			}
		}
		index += sampleLength;
	
	}

}

bool SampleSet::allEqualToZero(float* data, size_t size) {

	for(size_t i = 0; i < size; i++) {
		if(data[i] != 0.0f) {
			return false;
		}
	}

	return true;

}

void SampleSet::initializeCPrime() { 

	// Initialize the colors for the first iteration
	for(size_t i = 0; i < totalSize; i++) {
		float* currentSample = sampleInfo[i];
		cPrime[i] = currentSample[COLOR_1]; 
		cPrime[i + totalSize] = currentSample[COLOR_2];
		cPrime[i + 2*totalSize] = currentSample[COLOR_3];
	}

} 

void SampleSet::copyData(SampleSet* src, int x, int y, size_t size) {

	// Copy data from the source at position (x, y) to the first size
	// elements of "this" data array
	size_t srcTotalSize = src->getTotalSize();
	for(size_t i = 0; i < size; i++) {

		// Get index of source
		size_t index = samplesPerPixel * ((width * y) + x) + i;
		
		// Copy from the source to the ith index
		copyData(src, index, i);

	}

}

void SampleSet::copyData(SampleSet* src, size_t srcIndex, size_t destIndex) { 

	// Copy data from source sampleInfo array to "this" data array
	float* srcSample = src->getSampleInfo(srcIndex);
	for(size_t i = 0; i < sampleLength; i++) {
		data[destIndex + i*totalSize] = srcSample[i];
	}

	// Copy cPrime colors over from source
	size_t srcSize = src->getNumOfSamples();
	float* srcCPrime = src->getCPrime();
	for(int i = 0; i < NUM_OF_COLORS; i++) {
		cPrime[destIndex + i*totalSize] =  MIN(MAX_SAMP_VAL, srcCPrime[srcIndex + i*srcSize]); //srcCPrime[srcIndex + i*srcSize];//
	}

	// Update the sample count
	numOfSamples++;

} 

void SampleSet::updateColor(SampleSet* src, int x, int y, size_t size) {

	// Loop through the first size elements of the source pixelColors array
	// and update "this" cDoublePrime
	size_t srcSize = src->getNumOfSamples();
	float* srcPixelColors = src->getPixelColors();
	for(size_t i = 0; i < size; i++) {

		// Update cDoublePrime
		size_t index = samplesPerPixel * ((width * y) + x) + i;
		cDoublePrime[index] = srcPixelColors[NUM_OF_COLORS*i];
		cDoublePrime[index + totalSize] = srcPixelColors[NUM_OF_COLORS*i + 1];
		cDoublePrime[index + 2*totalSize] = srcPixelColors[NUM_OF_COLORS*i + 2];

	} 
	
} 

void SampleSet::generateRandomValues(size_t size, float sigma) { 

	// Make sure the random values haven't been initialized
	assert(!randomValuesInitialized);

	// Allocate the random arrays
	randomIndexes = new float[NUM_OF_POSITIONS * size * RANDOM_SET_SIZE * sizeof(float)];
	integerIndexes = new int[3*size];
	randomValuesInitialized = true;
	
	// Initialize the arrays to zero
	memset(randomIndexes, 0, 2*size*RANDOM_SET_SIZE*sizeof(float));
	memset(integerIndexes, 0, 3*size*sizeof(int));

	// Use Intel MKL to generate random values
	generateRandomNormal(randomIndexes, NUM_OF_POSITIONS * size * RANDOM_SET_SIZE, sigma);
	
	// Set a pointer to the newly generated random indexes
	currentRandomIndexes = randomIndexes;

} 

void SampleSet::generateRandomValues(size_t size) {

	// Find a random offset into the normal random values
	size_t index = size_t(2 * RAND * (RANDOM_SET_SIZE - 1) * size); 
	currentRandomIndexes = randomIndexes + index;

}

int* SampleSet::generateIndexes(int b, int x, int y, size_t size, size_t& numOfSamplesUsed) {

	// Initialize variables
	size_t randomValueIndex = 0;
	int delta = (int) floor(b/2); // Half of box size
	int lowerX = x - delta; // Lower bounds on x and y
	int lowerY = y - delta;

	// Allocate memory for counting number of samples to use at each index
	int* indexCounts = new int[totalSize]; 
	memset(indexCounts, 0, totalSize*sizeof(int));
	  
	for(size_t i = 0; i < size; i++) {

		// Get a random x and y value from the list
		int randX = int(currentRandomIndexes[randomValueIndex]);
		randomValueIndex++;
		int randY = int(currentRandomIndexes[randomValueIndex]);
		randomValueIndex++;
	
		// If the coordinates found are within the bounds of the box
		if(abs(randX) <= delta && abs(randY) <= delta) {

			// Get corresponding index of (x, y) coordinate
			size_t index = (delta + randX)*b + (delta + randY);
			assert(index >= 0 && index < totalSize);

			// Increase the count of this index (clamping it to the number of samples per pixel)
			indexCounts[index] = MIN((int) samplesPerPixel, indexCounts[index] + 1);
				
		}
	}

	// Avoid adding the samples at the current pixel twice
	size_t index = delta * b + delta;
	indexCounts[index] = 0; 

	// Initialize variables
	int maxNumberOfSamples = b*b; 
	size_t integerIndex = 0;
	numOfSamplesUsed = 0;
	
	for(int i = 0; i < maxNumberOfSamples; i++) {

		// If the current index has a nonzero count
		int currentCount = indexCounts[i];
		if(currentCount) {

			// Find the x and y coordinates
			int xCoord = i / b + lowerX;
			int yCoord = i % b + lowerY;

			// If they are within the bounds of the image
			if(xCoord >= 0 && xCoord < width && yCoord >= 0 && yCoord < height) {

				// Add them to the list
				integerIndexes[integerIndex++] = xCoord;
				integerIndexes[integerIndex++] = yCoord;
				integerIndexes[integerIndex++] = currentCount;

				// Update the number of pixel locations used
				numOfSamplesUsed++;

			}

		}
	}

	// Sanity check
	assert(randomValueIndex == NUM_OF_POSITIONS * size);
	assert(integerIndex == 3 * numOfSamplesUsed);
	
	// Clean up
	delete[] indexCounts;

	// Return valid indexes and counts
	return integerIndexes;

}


void SampleSet::calculatePixelStatistics() {
	
	// Make sure this is getting called only when the SampleSet has local (pixel) data
	assert(numOfSamples == samplesPerPixel);

	// Reset the values from the previous iteration
	memset(pixelFeatureMu, 0, numOfFeat * sizeof(float)); 
	memset(pixelFeatureSigma, 0, numOfFeat * sizeof(float));

	// Calculate mean and standard deviation for the pixel
	for(int k = 0; k < numOfFeat; k++) {
		
		//***** MEAN *****//

		size_t featureOffset = (featIndex + k) * totalSize;
		float fInv = 1.0f / numOfSamples;
		for(size_t i = 0; i < numOfSamples; i++) {

			pixelFeatureMu[k] += data[featureOffset + i];

		}

		// Finish calculating the mean
		pixelFeatureMu[k] *= fInv;


		//***** STANDARD DEVIATION *****//
		
		// Standard deviation with bias corrected estimator
		fInv = 1.0f / (numOfSamples - 1); 
		for(size_t i = 0; i < numOfSamples; i++) {

			float temp = data[featureOffset + i] - pixelFeatureMu[k];
			pixelFeatureSigma[k] += temp * temp;

		}

		// Finish calculating the standard deviation
		pixelFeatureSigma[k] = sqrtf(fInv * pixelFeatureSigma[k]);

	}

}

void SampleSet::checkFeatures(float* sampleFeatures, bool& keepSample) {
	
	// Boolean corresponding to whether we should keep the sample with these features
	keepSample = true;

	// Loop through all the features and check if this sample is an outlier
	for(int k = 0; k < numOfFeat; k++) {
		
		// Initialize variables
		float delta = fabs(sampleFeatures[k] - pixelFeatureMu[k]);
		bool outlier = false;
		int sigmaFactor = featureStdFactor; 

		if(k >= WORLD_2_X_OFFSET) { 
			continue;
		}

		// If we are dealing with world position we allow for a higher standard deviation factor
		if(k < 3) { 
			sigmaFactor = worldStdFactor;  
		}

		// If this sample is a constant we shouldn't throw it away
		if((delta < 0.1f) && (pixelFeatureSigma[k] < 0.1f)) { 

			outlier = false;
		
		// If this sample is way over the standard deviation
		} else if(delta > (sigmaFactor * pixelFeatureSigma[k])) {
		
			// Then we classify it as an outlier
			outlier = true;
		
		}

		// If we classified this sample as an outlier
		if(outlier) {

			// We throw it away and finish
			keepSample = false;
			return;

		}

	}

}


void SampleSet::calculateStatistics() {
	
	// Calculates the mean for each entry across all the samples for the entire block
	calculateMean();

	// Calculates the standard deviation for each entry across all the samples for the entire block
	calculateStandardDeviation();

	// Normalize each sample with mean and standard deviation calculated above
	normalizeSamples();

}

void SampleSet::calculateMean() {

	#if ENABLE_SSE
	
		// USE SSE

		// Calculate means with SSE
		for(size_t i = 0; i < sampleLength; i++) {

			dataMu[i] = calculateMeanSSE(data + i*totalSize, numOfSamples);

		}

	#else 
	
		// NOT SSE

		// Calculate means (long way)
		
		// Loop through all samples
		memset(dataMu, 0, numOfFeat*sizeof(float)); 
		
		for(size_t i = 0; i < sampleLength; i++) {
			for(size_t j = 0; j < numOfSamples; j++) {
				dataMu[i] += data[i*totalSize + j];
			}
		}
	
		// Calculate the averages
		float scalar = 1.0f / numOfSamples;
		for(size_t i = 0; i < sampleLength; i++) {
			dataMu[i] *= scalar;
		}

	#endif

}

void SampleSet::calculateStandardDeviation() {

	#if ENABLE_SSE

		// USE SSE

		// Calculate standard deviation with SSE
		for(size_t i = 0; i < sampleLength; i++) {
		
			dataSigma[i] = calculateStdSSE(data + i*totalSize, dataMu[i], numOfSamples);
		
		}
	
	#else

		// NOT SSE
		// Calculate standard deviation (long way)
	
		// Loop through all samples
		memset(dataSigma, 0, numOfFeat*sizeof(float));
		for(size_t i = 0; i < sampleLength; i++) {
			for(size_t j = 0; j < numOfSamples; j++) {
				dataSigma[i] += pow((data[i*totalSize + j] - dataMu[i]), 2);
			}
		}
	
		// Finish computing the standard deviation
		float scalar = 1.0f / (numOfSamples - 1); 
		for(size_t i = 0; i < sampleLength; i++) {
			dataSigma[i] = sqrt(scalar * dataSigma[i]);
		}
		
	#endif

}

void SampleSet::normalizeSamples() {

	#if ENABLE_SSE

		// Use SSE

		// Normalize each sample using SSE
		for(size_t i = 0; i < sampleLength; i++) {

			normalizeSSE(data + i*totalSize, dataMu[i], dataSigma[i], normalizedData + i*totalSize, numOfSamples);

		}

	#else

		// Normalize (long way)
		float* sigmaPtr = dataSigma;
		float* muPtr = dataMu;
		for(size_t i = 0; i < sampleLength; i++) {
			float* dataPtr = data + i*totalSize;
			float* normPtr = normalizedData + i*totalSize;
			float stdInv;
			if(*sigmaPtr < EPSILON) {
				stdInv = 1.0f / EPSILON;
			} else {
				stdInv = 1.0f / *sigmaPtr;
			}
			for(size_t j = 0; j < numOfSamples; j++) {
				*normPtr = ((*dataPtr) - (*muPtr)) * stdInv;
				normPtr++;
				dataPtr++;
			}
			sigmaPtr++;
			muPtr++;
		}
	
	#endif

}

void SampleSet::initializePDFs() {

	// Compute pdfs
	for(size_t i = 0; i < sampleLength; i++) {

		// Quantize the normalized data into integer bins (and save the number of bins in binSizes)
		binSizes[i] = quantizeVector(normalizedData + i*totalSize, histograms + i*totalSize, numOfSamples);
		assert(binSizes[i] > 0 && binSizes[i] <= MAX_NUM_OF_BINS);

		// Calculate the pdf from the histogram
		computePDF(histograms + i*totalSize, PDFs + i*MAX_NUM_OF_BINS, binSizes[i]);
	
	}

}

void SampleSet::computePDF(int* quantizedSrc, float* pdf, size_t size) {

	// Start fresh
	memset(pdf, 0, size * sizeof(float));

	// Count how many fall into each bin
	for(size_t i = 0; i < numOfSamples; i++) {
		size_t index = quantizedSrc[i];
		assert(index < size);
		pdf[index]++;
	} 

	// Finish calculating the pdf by dividing by the total number of samples
	float fInv = 1.0f / numOfSamples;
	for(size_t i = 0; i < size; i++) {
		assert(pdf[i] <= numOfSamples);

		// Avoid quantization error
		if(pdf[i] == numOfSamples) { 
			pdf[i] = 1.0f;
			break;
		} 

		pdf[i] *= fInv;
		assert(pdf[i] >= 0.0f && pdf[i] <= 1.0f);
	} 

}

void SampleSet::computeJointPDF(int* quantizedP1, int* quantizedP2, float* jointPdf, int p1Size, int p2Size) {

	// Count how many fall into each bin
	for(size_t i = 0; i < numOfSamples; i++) {
		int index = quantizedP1[i]*p2Size + quantizedP2[i];
		jointPdf[index]++;
	}

	// Finish calculating the joint pdf by dividing by the total number of samples
	float fInv = 1.0f / numOfSamples;
	for(int i = 0; i < p1Size; i++) {
		for(int j = 0; j < p2Size; j++) {
			int index = i*p2Size + j;
			assert(jointPdf[index] <= numOfSamples);

			// Avoid quantization error
			if(jointPdf[index] == numOfSamples) { 
				jointPdf[index] = 1.0f;
				break;
			} 

			jointPdf[index] *= fInv;
			assert(jointPdf[index] >= 0 && jointPdf[index] <= 1.0f);
		}
	} 

}

float SampleSet::calculateDependency(int specifier1, int specifier2, int index1, int index2, int jointIndex) {

		// Get the specified pdfs and their sizes
	float* pdfX = PDFs + (specifier1 + index1) * MAX_NUM_OF_BINS;
	float* pdfY = PDFs + (specifier2 + index2) * MAX_NUM_OF_BINS; 
	int pdfSizeX = binSizes[specifier1 + index1];
	int pdfSizeY = binSizes[specifier2 + index2];
	float* pdfXY = new float[pdfSizeX * pdfSizeY];
	memset(pdfXY, 0, pdfSizeX * pdfSizeY * sizeof(float));
	computeJointPDF(histograms + (specifier1 + index1)*totalSize, histograms + (specifier2 + index2)*totalSize, pdfXY, pdfSizeX, pdfSizeY);

	// Calculate the mutual information between the two random variables
	float mutualInfo = getMutualInformation(pdfX, pdfY, pdfXY, pdfSizeX, pdfSizeY);

	assert(mutualInfo > -0.1);
	mutualInfo = MAX(mutualInfo, 0.0f);

	delete[] pdfXY;

	return mutualInfo;

}

void SampleSet::spikeRemoval(float colorStdScalar) { 

	// Initialize variables
	float avgScalar = 1.0f / samplesPerPixel;
	float stdScalar = 1.0f / (samplesPerPixel - 1);
	size_t imageSize = width * height;
	size_t index = 0;

	// Initialize standard deviation variables for each color channel
	float stdR = 0; 
	float stdG = 0;
	float stdB = 0; 

	// Loop through each pixel
	for(size_t i = 0; i < imageSize; i++) {

		// Initialize mean variables for each color channel
		float muR = 0;
		float muG = 0;
		float muB = 0;

		// Find the corresponding index
		index = samplesPerPixel*i;

		// Calculate mean color of each pixel
		for(size_t k = 0; k < samplesPerPixel; k++) {

			// Add colors from each channel
			muR += cPrime[index];
			muG += cPrime[index + totalSize];
			muB += cPrime[index + 2*totalSize];

			// Go to the next sample
			index++;

		} 

		// Finish calculating the mean
		muR *= avgScalar; 
		muG *= avgScalar;
		muB *= avgScalar;

		// Reset index
		index = samplesPerPixel*i;
		stdR = 0; 
		stdG = 0;
		stdB = 0; 

		// Calculate standard deviation of color at each pixel
		for(size_t k = 0; k < samplesPerPixel; k++) {

			// Calculate the difference from the mean for each channel
			float diffR = cPrime[index] - muR;
			float diffG = cPrime[index + totalSize] - muG;
			float diffB = cPrime[index + 2*totalSize] - muB;

			// Calculate the square of the difference and add it to the running sum
			stdR += diffR * diffR;
			stdG += diffG * diffG;
			stdB += diffB * diffB;

			// Go to the next sample
			index++;

		}

		// Finish calculating standard deviation
		stdR = sqrt(stdR * stdScalar); 
		stdG = sqrt(stdG * stdScalar);
		stdB = sqrt(stdB * stdScalar);

		// Calculate max pixel 
		float maxR = muR + colorStdScalar*stdR;
		float maxG = muG + colorStdScalar*stdG;
		float maxB = muB + colorStdScalar*stdB;

		// Reset index
		index = samplesPerPixel*i;

		// Clamp pixel values
		for(size_t k = 0; k < samplesPerPixel; k++) {

			// Clamp red channel
			if(cPrime[index] > maxR) { 
				cPrime[index] = maxR; 
			}

			// Clamp green channel
			if(cPrime[index + totalSize] > maxG) { 
				cPrime[index + totalSize] = maxG; 
			}

			// Clamp blue channel
			if(cPrime[index + 2*totalSize] > maxB) {
				cPrime[index + 2*totalSize] = maxB; 
			}

			// Go to the next sample
			index++;

		}

	}

}

void SampleSet::setColors() {

	// Update cPrime with the values from cDoublePrime
	for(size_t i = 0; i < totalSize; i++) {

		// Loop through each color channel
		for(int k = 0; k < NUM_OF_COLORS; k++) {

			cPrime[i + k*totalSize] = cDoublePrime[i + k*totalSize];
		
		}
				
	} 
	
} 

size_t SampleSet::getTotalSize() {
	return totalSize;
}

size_t SampleSet::getNumOfSamples() {
	return numOfSamples;
}

void SampleSet::getColor(size_t index, float& c0, float& c1, float& c2) {

	assert(index < numOfSamples);
	float* currentSample = sampleInfo[index];
	c0 = currentSample[COLOR_1];
	c1 = currentSample[COLOR_2];
	c2 = currentSample[COLOR_3];

} 

void SampleSet::getCPrime(size_t index, float& c0, float& c1, float& c2) {

	assert(index < numOfSamples);
	c0 = cPrime[index];
	c1 = cPrime[index + totalSize];
	c2 = cPrime[index + 2*totalSize];

} 

void SampleSet::getFeature(size_t featureOffset, size_t sampleIndex, float& x) {

	assert(sampleIndex < numOfSamples);
	assert(featureOffset < numOfFeat);
	x = data[(featIndex + featureOffset)*totalSize + sampleIndex];

}

void SampleSet::getNormColor(size_t index, float& c0, float& c1, float& c2) {

	assert(index < numOfSamples);
	c0 = normalizedData[(COLOR_1 * totalSize) + index];
	c1 = normalizedData[(COLOR_2 * totalSize) + index];
	c2 = normalizedData[(COLOR_3 * totalSize) + index];

}

void SampleSet::getNormFeature(size_t featureOffset, size_t sampleIndex, float& x) {

	assert(sampleIndex < numOfSamples);
	assert(featureOffset < numOfFeat);
	x = normalizedData[(featIndex + featureOffset)*totalSize + sampleIndex];

} 

void SampleSet::setPixelColors(size_t index, float c0, float c1, float c2) {

	assert(index < samplesPerPixel);
	pixelColors[NUM_OF_COLORS*index] = c0;
	pixelColors[NUM_OF_COLORS*index + 1] = c1;
	pixelColors[NUM_OF_COLORS*index + 2] = c2;
}

float* SampleSet::getCPrime(size_t index) {

	assert(index < NUM_OF_COLORS);
	return cPrime + (index * totalSize);
	
} 

float* SampleSet::getNormColor(size_t index) {

	assert(index < NUM_OF_COLORS);
	return normalizedData + ((colIndex + index) * totalSize);

} 

float* SampleSet::getNormFeature(size_t index) {

	assert(index < numOfFeat);
	return normalizedData + ((featIndex + index) * totalSize);

} 

void SampleSet::setNumOfSamples(size_t numOfSamples) {
	this->numOfSamples = numOfSamples;
}

float* SampleSet::getNormColorBuffer() {
	return normColorBuffer;
}

float* SampleSet::getNormFeatureBuffer() {
	return normFeatureBuffer;
}

float* SampleSet::getColorBuffer() {
	return colorBuffer;
}

float* SampleSet::getFeatureBuffer() {
	return featureBuffer;
}

float* SampleSet::getTempStorageBuffer() {
	return tempStorageBuffer;
}

float* SampleSet::getWeightsBuffer() {
	return weightsBuffer;
}

float* SampleSet::getTempWeightsBuffer() {
	return tempWeightsBuffer;
}

float* SampleSet::getAlpha() {
	return alpha;
}

float* SampleSet::getBeta() {
	return beta;
}

float* SampleSet::getCPrime() {
	return cPrime;
}

float* SampleSet::getSampleInfo(size_t index) {
	assert(index < numOfSamples);
	return sampleInfo[index];
}

float* SampleSet::getPixelColors() {
	return pixelColors;
}

float* SampleSet::getFeatures(size_t index) {
	assert(index < numOfSamples);
	return sampleInfo[index] + featIndex;
}
