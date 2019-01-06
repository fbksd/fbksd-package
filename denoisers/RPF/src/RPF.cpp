#include "RPF.h"
#include "ExrUtilities.h"

// An array of pointers that point to feature information for each
// sample in the entire image
SampleSet* samples = NULL;

void initializeData(float* pbrtData, size_t pbrtWidth, size_t pbrtHeight, 
					size_t pbrtSpp, size_t pbrtSampleLength, int posCount, int colorCount, int featureCount, int randomCount, FILE* datafp) {

	setDefaultParameters();
	
	width = pbrtWidth;
	height = pbrtHeight;
	samplesPerPixel = pbrtSpp;
	sampleLength = pbrtSampleLength;

	posIndex = 0;
	colIndex = posCount;
	featIndex = colIndex + colorCount;
	randIndex = featIndex + featureCount;
	numOfFeat = featureCount;
	numOfRand = randomCount;
	assert(randIndex + randomCount == sampleLength);

	// Make sure values make sense
	assert(width > 0);	
	assert(height > 0);
	assert(samplesPerPixel > 0);
	assert(sampleLength > 0);

	// Initialize a container for all of the samples
	size_t totalSize = width * height * samplesPerPixel;
	samples = new SampleSet(totalSize, false);
	
	// Read the samples to sampleset
	samples->readSamples(pbrtData);
	
	// Sanity check
	assert(samples->getNumOfSamples() == totalSize); // Make sure we added everything

	// Initialize cPrime with all the sample colors
	samples->initializeCPrime();

}

// ALGORITHM 1 FROM TECHNICAL REPORT
void RPF(CImg<float>* rpfImg, CImg<float>* origImg) { 

	// Set up the desired number of threads for multithreading (see globals.h)
	omp_set_num_threads(NUM_OF_THREADS);

	// Initialize gaussian random number generator
	PreComputeGaussian::initializeTable();
	int threadCount = omp_get_max_threads(); 

	// Variables used for updating the user of program progress
	size_t totalSize = width * height;
	int printDelta = height / 10;
	
	// Main filtering loop
	for(int t = 0; t < numOfIterations; t++) {

		// Update user
		std::cout << "Beginning Iteration #" << t + 1 << endl;

		// Initialize sample variables
		int b = blockSizes[t]; 
		assert(b > 0);
	
		size_t maxNumOfSamples;
		if(t < 2) {

			// For the first iterations use fewer samples
			maxNumOfSamples = (b * b * samplesPerPixel) / 3; 

		} else {

			// For the later iterations use more samples
			maxNumOfSamples = (b * b * samplesPerPixel) / 2;

		}

		// Standard deviation for choosing normal random numbers
		float sigmaP = MAX(1.0f, ((float) (b - 1)) / randVarFactor); 
		size_t numOfRandomValues = maxNumOfSamples; 
		size_t blockSize = samplesPerPixel * b * b;
		SampleSet* neighborList[NUM_OF_THREADS]; 
		
		// Initialize a block for each thread for the current iteration
		for(int k = 0; k < threadCount; k++) {

			// Allocate new set
			SampleSet* currentNeighbor = new SampleSet(blockSize, true);

			// Generate normal random numbers ahead of time
			currentNeighbor->generateRandomValues(numOfRandomValues, sigmaP);
			
			// Save the set to be accessed later
			neighborList[k] = currentNeighbor;

		}

		// Update user
		std::cout << "Box size of " << b << endl;

		// For monitoring program progress
		int amountCompleted = 0;

		// Multithreaded portion
		// Loop through each pixel
        #pragma omp parallel
		{
            #pragma omp for schedule(dynamic) nowait
			for(int i = 0; i < height; i++) { 
				for(int j = 0; j < width; j++) {

					// Get this thread's number and retrieve the allocated block
                    int threadNum = omp_get_thread_num();
					SampleSet* neighboringSamples = neighborList[threadNum];

					// Copy the data over from the current pixel to the current block
					neighboringSamples->copyData(samples, j, i, samplesPerPixel); 
					assert(neighboringSamples->getNumOfSamples() == samplesPerPixel);

					// Add only a portion of the samples in the region to the block (clustering)
					// Also calculates the block statistics
					preProcessSamples(b, maxNumOfSamples, neighboringSamples, j, i);

					// Retrieve the alpha (color) and beta (feature) weights
					float* alpha = neighboringSamples->getAlpha();
					float* beta = neighboringSamples->getBeta();
					
					// Compute the values for the weights for the bilateral filter
					// Also this initializes the histograms and pdfs used for mutual information
					float contributionCR = computeFeatureWeights(t, neighboringSamples, alpha, beta);
					
					// Perform the actual filtering to get the final colors of the current pixel
					// for the current iteration
					filterColorSamples(neighboringSamples, alpha, beta, contributionCR, t);

					// Save the pixel colors of the current pixel to the buffer for the main 
					// set (cDoublePrime)
					samples->updateColor(neighboringSamples, j, i, samplesPerPixel);

					// Reset the block to have no samples (so the next thread can use it)
					neighboringSamples->setNumOfSamples(0);
				
					// Used for updating the user of program progress
					amountCompleted++;	

				}

				// Update the user
				if(i % printDelta == 0) {
					printf("Percentage Complete: %2.2f \n", (float(amountCompleted) / totalSize) * 100);
					fflush(stdout);
				}

			}
		} 

		// Save the pixel colors for all the pixels calculated above into the main set 
		// containing all the data for the entire image (set cPrime to cDoublePrime)
		samples->setColors(); 

		// Remove the remaining noise based on outliers from local pixel statistics
		// (The standard deviation for classifying outliers is reduced every iteration)
		float colorStdScalar = colorStdFactor / pow(reduceFactor, t); 
		colorStdScalar = MAX(0.5f, colorStdScalar);
		
		// Filters the speckles based off the type of filter specified in the cfg file.
		// Can use mean, median, or threshold to eliminate speckles.
		samples->spikeRemoval(colorStdScalar); 
		 
		// Delete the sets for all the threads (new sizes will be allocated during
		// the next iteration)
		for(int k = 0; k < threadCount; k++) {
			delete neighborList[k];
		}

	}  

	// Samples in image have been filtered. Use box filter to compute final pixel values
	boxFilter(rpfImg, false); 
	boxFilter(origImg, true);

	// Clean up
	if(blockSizes) {
		delete[] blockSizes;
	}

}

void RPF(float* result, float* pbrtData, size_t pbrtWidth,
                      size_t pbrtHeight, size_t pbrtSpp, size_t pbrtSampleLength, int posCount, int colorCount, int featureCount, int randomCount, FILE* datafp) {

    fprintf(stdout, "Starting RPF\n");
    fflush(stdout);

    // Start timer
//    Timer timer;
//    timer.Start();

    // Get samples from file and add to list of samples
    initializeData(pbrtData, pbrtWidth, pbrtHeight, pbrtSpp, pbrtSampleLength, posCount, colorCount, featureCount, randomCount, datafp);

    // Initialize image
    CImg<float>* rpfImg = new CImg<float>(width, height, 1, 3);
    CImg<float>* origImg = new CImg<float>(width, height, 1, 3);

    // Perform random parameter filtering
    RPF(rpfImg, origImg);

    int numPixels = width * height;
    float* resultData = rpfImg->data();
    for(int c = 0; c < 3; ++c)
    for(int p = 0; p < numPixels; ++p)
    {
        float v = resultData[c*numPixels + p];
        result[p*3 + c] = v;
    }

    // Save filtered image
//    char outputName[1000];
//    sprintf(outputName, "%s_RPF_flt.exr", outputFolder);
//    float* imgData = new float[NUM_OF_COLORS * pbrtWidth * pbrtHeight];
//    WriteEXRFile(outputName, (int) pbrtWidth, (int) pbrtHeight, rpfImg->data());

    // Save original image
//    sprintf(outputName, "%s_MC_%04d.exr", outputFolder, pbrtSpp);
//    WriteEXRFile(outputName, (int) pbrtWidth, (int) pbrtHeight, origImg->data());

    // Clean up
    delete rpfImg;
    delete origImg;
    delete samples;
//    delete[] imgData;

    // Wait for user
    printf("Finished with RPF\n");

    // Output runtime
//    timer.Stop();
//    float runtime = (float) timer.Time();
//    fprintf(stdout, "Runtime: %.2lf secs\n\n", runtime);
}

// ALGORITHM 2 FROM TECHNICAL REPORT
void preProcessSamples(int b, size_t maxNumOfSamples, SampleSet* neighboringSamples, int x, int y) {

	// Calculate statistics for pixel samples
	neighboringSamples->calculatePixelStatistics();

	// Initialize variables
	size_t numOfSamplesToAdd = maxNumOfSamples;
	size_t neighborIndex = samplesPerPixel; // Offset since we copied the current pixel data already
	size_t numOfSamplesUsed = 0;
	size_t randomValueIndex = 0;
	bool keepSample;

	// Choose normal random numbers (finds random position in precomputed random number array)
	neighboringSamples->generateRandomValues(numOfSamplesToAdd);

	// Get array of x and y coordinates as well as counts
	int* randomIndexes = neighboringSamples->generateIndexes(b, x, y, numOfSamplesToAdd, numOfSamplesUsed);

	// Add samples to neighborhood based on the number of potential coordinates found
	for(size_t q = 0; q < numOfSamplesUsed; q++) {

		// Initialize variables
		keepSample = true;
		int j = -1;
		int i = -1;

		// Get random sample data
		j = randomIndexes[randomValueIndex]; // X coordinate
		randomValueIndex++;
		i = randomIndexes[randomValueIndex]; // Y coordinate
		randomValueIndex++;
		int n = randomIndexes[randomValueIndex]; // Count
		randomValueIndex++;
		
		// If the count is zero we continue
		assert(n >= 0);
		if(n == 0) {
			continue;
		}
	
		// Make sure we have valid coordinates
		assert((j >= 0) && (i >= 0));
		
		// If we land on the current pixel then we skip to the next one
		if(j == x && i == y) { 
			continue;
		}

		// Index into the data of the random coordinate
		size_t index = samplesPerPixel * ((i * width) + j);

		// Perform clustering
		// Loop through the number of samples at this coordinate
		for(int l = 0; l < n; l++) {

			// Get the feature array of the current sample
			float* sampleFeatures = samples->getFeatures(index);

			// Check if the features are outliers based on pixel statistics
			neighboringSamples->checkFeatures(sampleFeatures, keepSample);
	
			// If sample is not an outlier then save it
			if(keepSample) {

				// Save the data from the main set to this block at neighborIndex
				neighboringSamples->copyData(samples, index, neighborIndex);
				neighborIndex++;

			} 

			// Go to the next sample at this coordinate
			index++;
		}

	}

	// Make sure we processed all of the samples
	assert(neighborIndex == neighboringSamples->getNumOfSamples());
	
	// Neighborhood is now ready for statistical analysis

	// Compute mean and standard deviation and create a normalized vector by removing 
	// the mean and dividing by the standard deviation for each entry in the data array
	
	neighboringSamples->calculateStatistics();
	
}

// ALGORITHM 3 FROM TECHNICAL REPORT
float computeFeatureWeights(int t, SampleSet* neighboringSamples, float* alpha, float* beta) {

	// Compute all pdfs before calculating mutual information
	neighboringSamples->initializePDFs();

	// Initialize variables
	int jointIndex = 0; // Index into the joint pdfs
	float dependencyCR = 0.0f; // Dependency between color and random parameters
	float dependencyCP = 0.0f; // Dependency between color and position
	float dependencyCF = 0.0f; // Dependency between color and feature
	float contributionCR = 0.0f; 
	float* dependencyListCF = new float[numOfFeat]; 
	memset(dependencyListCF, 0, numOfFeat * sizeof(float));
	float channelScalar = 1.0f / 3.0f; 

	// Compute the dependencies for the colors using the samples in the neighboring samples
	for(int k = 0; k < NUM_OF_COLORS; k++) {

		// Compute dependency of color on random parameters
		float tempDependencyCR = 0.0f; 
		for(int l = 0; l < numOfRand; l++) {

			// Calculate dependency of the kth color channel and the lth random parameter
			tempDependencyCR += neighboringSamples->calculateDependency(colIndex, randIndex, k, l, jointIndex);
			jointIndex++;

		}
	
		// Compute dependency of color on position
		float tempDependencyCP = 0.0f;
		for(int l = 0; l < NUM_OF_POSITIONS; l++) {
			
			// Calculate dependency of the kth color channel and the lth position (x and y)
			tempDependencyCP += neighboringSamples->calculateDependency(colIndex, posIndex, k, l, jointIndex);
			jointIndex++;

		}

		// Compute dependency of color on features
		float tempDependencyCF = 0.0f;
		for(int l = 0; l < numOfFeat; l++) {

			// Calculate dependency of the kth color channel and the lth feature
			float temp = neighboringSamples->calculateDependency(colIndex, featIndex, k, l, jointIndex); 
			tempDependencyCF += temp;
			dependencyListCF[l] += temp;
			jointIndex++;

		}

		// Add contribution from each color channel
		dependencyCR += tempDependencyCR;
		dependencyCP += tempDependencyCP;
		dependencyCF += tempDependencyCF;

		float tempContributionCR = tempDependencyCR / (tempDependencyCR + tempDependencyCP + EPSILON);
		
		// Compute alpha (color) weight
		float tempAlpha = MAX(1.0f - (alphaFactor * (1.0f + (startMutual + mutualIncrease * t)) * tempContributionCR), 0.0f); 
		assert(tempAlpha >= 0 && tempAlpha <= 1);
		alpha[k] = tempAlpha;

	}

	contributionCR = (dependencyCR * channelScalar) / ((dependencyCR * channelScalar) + (dependencyCP * channelScalar) + EPSILON); 

	// Compute the dependencies for the scene features using neighboring samples
	for(int k = 0; k < numOfFeat; k++) {

		// Compute dependency between features and random parameters
		float dependencyFR = 0.0f;
		for(int l = 0; l < numOfRand; l++) {

			// Calculate dependency of the kth feature and the lth random parameter
			float tempVal = 0;
			tempVal = neighboringSamples->calculateDependency(featIndex, randIndex, k, l, jointIndex);
			dependencyFR += tempVal;
			jointIndex++;

		}
	
		// Compute dependency between features and position
		float dependencyFP = 0.0f;
		for(int l = 0; l < NUM_OF_POSITIONS; l++) {

			// Calculate dependency of the kth feature and the lth position (x and y)
			dependencyFP += neighboringSamples->calculateDependency(featIndex, posIndex, k, l, jointIndex);
			jointIndex++;

		}

		float dependencyCF_K = dependencyListCF[k] * channelScalar; 

		float contributionFR = dependencyFR / (dependencyFR + dependencyFP + EPSILON);
		
		float contributionCF = 0.0f;
		contributionCF = dependencyCF_K / (EPSILON + (dependencyCR * channelScalar) + (dependencyCP * channelScalar) + (dependencyCF * channelScalar));

		// Compute beta (feature) weigtht
		assert(contributionCF == contributionCF); // assert not NAN
		float tempBeta = contributionCF * MAX(1.0f - ((1 + (startMutual + mutualIncrease * t))*contributionFR), 0.0f); 
		beta[k] = tempBeta;

	}

	// Boost texture weights
	int textureIndex = TEXTURE_1_X - featIndex;
	for(int i = 0; i < 3; i++) {
		beta[textureIndex + i] *= textureFactor;
	}

	int normIndex = NORM_1_X - featIndex;
	for(int i = 0; i < 3; i++) {
		beta[normIndex + i] *= normFactor;
	}

	// Clean up
	delete[] dependencyListCF;

	return contributionCR;

}

// ALGORITHM 4 FROM TECHNICAL REPORT
void filterColorSamples(SampleSet* neighboringSamples, float* alpha, float* beta, float contributionCR, int t) {
	
	assert(samplesPerPixel != 0);
	assert(variance > 0);
	float currentVariance = (float) (1 + stdIncrease * t) / variance; 
	float featureSigma = currentVariance * pow(1 - contributionCR, 2);
	float colorSigma = featureSigma;
	size_t pixelSize = samplesPerPixel;
	size_t neighborSize = neighboringSamples->getNumOfSamples();

	#if ENABLE_SSE

		// USE SSE
		
		// Grab aligned memory buffers for SSE
		float* pixelNormColor = neighboringSamples->getNormColorBuffer(); // OPTIMIZATION: For speed, maybe initialize every time?
		float* pixelNormFeature = neighboringSamples->getNormFeatureBuffer();
		float* colorTerm = neighboringSamples->getColorBuffer();
		float* featureTerm = neighboringSamples->getFeatureBuffer();
		float* tempStorage = neighboringSamples->getTempStorageBuffer();;
		float* weights = neighboringSamples->getWeightsBuffer();
		float* tempWeights = neighboringSamples->getTempWeightsBuffer();

		// For each sample at the current pixel
		for(size_t i = 0; i < pixelSize; i++) {

			// Reset some of the buffers
			memset(weights, 0, neighborSize*sizeof(float));
			memset(colorTerm, 0, neighborSize*sizeof(float));
			memset(featureTerm, 0, neighborSize*sizeof(float));
			
			// Calculate the color term for the bilateral filter
			float cDoublePrime[NUM_OF_COLORS]= {0.0f, 0.0f, 0.0f};
			float c0 = 0, c1 = 0, c2 = 0;
			neighboringSamples->getNormColor(i, c0, c1, c2); // Normalized color
			float pixelNormColorTemp[NUM_OF_COLORS] = {c0, c1, c2}; 
			for(int k = 0; k < NUM_OF_COLORS; k++) { // For each color channel

				// Fill up buffer with norm color value of this pixel sample at the current channel
				for(size_t q = 0; q < neighborSize; q++) {
					pixelNormColor[q] = pixelNormColorTemp[k]; 
				}

				// Calculate the sum of the squared differences between the norm colors of the samples and the norm pixel color
				subtractSquareScaleSSE(pixelNormColor, neighboringSamples->getNormColor(k), tempStorage, alpha[k], neighborSize);
				sumSSE(colorTerm, tempStorage, colorTerm, neighborSize);
			}
			
			// Multiply by 1 / sigmaC term from paper (section 6)
			multByValSSE(colorTerm, colorTerm, colorSigma, neighborSize); 
			
			// For all the features
			for(int k = 0; k < numOfFeat; k++) {
				float normFeatureTemp = 0.0f;
				neighboringSamples->getNormFeature(k, i, normFeatureTemp); 
				
				// Fill up buffer with norm feature value of this pixel sample
				for(size_t q = 0; q < neighborSize; q++) {
					pixelNormFeature[q] = normFeatureTemp;
				}

				// Calculate the sum of the squared differences between the norm features of the samples and the norm pixel color
				subtractSquareScaleSSE(pixelNormFeature, neighboringSamples->getNormFeature(k), tempStorage, beta[k], neighborSize);
				sumSSE(featureTerm, tempStorage, featureTerm, neighborSize);
			}

			// Multiply by 1 / sigmaF term from paper (section 6)
			multByValSSE(featureTerm, featureTerm, featureSigma, neighborSize);

			// Find exp(colorTerm) and exp(featureTerm)
			computeExp(colorTerm, neighborSize);
			computeExp(featureTerm, neighborSize);

			// Multiply the exp(colorTerm) and exp(featureTerm) terms and add them to weights
			multSSE(colorTerm, featureTerm, tempWeights, neighborSize);
			sumSSE(weights, tempWeights, weights, neighborSize);
		
			// Loop through all color channels
			for(int k = 0; k < NUM_OF_COLORS; k++) {

				// Multiply colors (cPrime) by weights and save them in cDoublePrime
				float* neighborColor = neighboringSamples->getCPrime(k) ;
				multSSE(neighborColor, tempWeights, tempStorage, neighborSize); 
				cDoublePrime[k] += sumAllElementsSSE(tempStorage, neighborSize);	
		
			} 

			// Add up all of the weights 
			float finalWeight = sumAllElementsSSE(weights, neighborSize);
		
			// Calculate final color
			if(finalWeight != 0) { // TODO: Make sure weight is never zero
				float cDoublePrimeScalar = 1.0f / finalWeight;
				for(int k = 0; k < NUM_OF_COLORS; k++) {
					cDoublePrime[k] *= cDoublePrimeScalar;
				} 
			}

			// Save the current filtered pixel sample color 
			neighboringSamples->setPixelColors(i, cDoublePrime[0], cDoublePrime[1], cDoublePrime[2]); 
		
		} 
		
	#else  

		// NOT SSE
		// Filter the colors of samples in pixel using bilateral filter
		for(size_t i = 0; i < pixelSize; i++) {

			// Initialize variables and grab sample from pixel
			float w = 0.0f;
			float cDoublePrime[NUM_OF_COLORS]= {0.0f, 0.0f, 0.0f};
			float normC1 = 0, normC2 = 0, normC3 = 0;
			neighboringSamples->getNormColor(i, normC1, normC2, normC3);
			float pixelColor[NUM_OF_COLORS] = {normC1, normC2, normC3};
		
			for(size_t j = 0; j < neighborSize; j++) {

				// Calculate the color term of the weight
				float colorTerm = 0.0f;
				neighboringSamples->getNormColor(j, normC1, normC2, normC3);
				float neighborColor[NUM_OF_COLORS] = {normC1, normC2, normC3};
				for(int k = 0; k < NUM_OF_COLORS; k++) {

					float currentAlpha = alpha[k];
					colorTerm += currentAlpha * pow(pixelColor[k] - neighborColor[k], 2);

				}
				colorTerm = PreComputeGaussian::gaussianDistance(colorScalar*colorTerm);
			
				// Calculate the feature term of the weight
				float featureTerm = 0.0f;
				for(int k = 0; k < numOfFeat; k++) {

					float currentBeta = beta[k];
					float pixelF = 0, neighborF = 0;
					neighboringSamples->getNormFeature(k, i, pixelF);
					neighboringSamples->getNormFeature(k, j, neighborF);
					featureTerm += currentBeta * pow(pixelF - neighborF, 2);

				}
				featureTerm = PreComputeGaussian::gaussianDistance(featureScalar*featureTerm);
					
				// Calculate the weight and color for this sample
				float wij = colorTerm * featureTerm; 
				float c1 = 0, c2 = 0, c3 = 0;
				neighboringSamples->getCPrime(j, c1, c2, c3);
				float colorTemp[NUM_OF_COLORS] = {c1, c2, c3};
				for(int k = 0; k < NUM_OF_COLORS; k++) {
					cDoublePrime[k] += wij * colorTemp[k]; 	
				} 

				w += wij;

			}

			// Calculate final color
			assert(w != 0); 
			for(int k = 0; k < NUM_OF_COLORS; k++) {
				cDoublePrime[k] /= w;
			}

			neighboringSamples->setPixelColors(i, cDoublePrime[0], cDoublePrime[1], cDoublePrime[2]);

		} 

	#endif 

}

//***** HELPER FUNCTIONS *****//

void setDefaultParameters() {

	// Default variables
	variance = 0.05f;			// Variance for filter. Adjust from 0.002 (less noisy scenes) to 0.02 (more noisy scenes)
	featureStdFactor = 3.0f;	// For clustering features other than world coordinates (standard deviation threshold)
	worldStdFactor = 50.0f;		// For clustering world coordinates	(standard deviation threshold)
	mutualIncrease = 0.05f;		// Used in the filter weight calculation (is multiplied by current iteration)
	alphaFactor = 4.0f;			// Used in the alpha weight calculation (10 is default)
	textureFactor = 25.0f;		// Scalar to influence the filtering weights for the texture term in the bilateral filter
	normFactor = 10.0f;			// Scalar to influence the filtering weights for the normal term in the bilateral filter
	colorStdFactor = 0.5f;		// Number of standard deviations to characterize a spike in color
	reduceFactor = 2.0f;		// Factor that reduces the colorStdFactor (1 means it doesn't get reduced for each iteration)
	randVarFactor = 4.0f;		// Variance for randomly selecting samples in the neighborhood of each pixel	
	numOfIterations = 4;		// Number of iterations of the algorithm
	stdIncrease = 0.05f;		// Increase of variance for each iteration (0 is default so that it doesn't change)
	startMutual = 0.3f;			// Used for calculating the filter weights (alpha and beta)
	 
	// Array specifying the block sizes to use for each iteration
	blockSizes = new int[numOfIterations];
	assert(numOfIterations == 4);
	blockSizes[0] = 45;
	blockSizes[1] = 30;
	blockSizes[2] = 15;
	blockSizes[3] = 5;

	// Deprecated Terms
	hasExplicitBlocks = true;
	startSize = -1;	
	endSize = -1; 

}

void parseConfigFile(FILE* fp) {

	// Initialize variables
	char strName[100];
	float val = 0;
	hasExplicitBlocks = false;
	variance = -1.0f;
	stdIncrease = -1.0f;
	randVarFactor = -1.0f;
	featureStdFactor = -1;
	worldStdFactor = -1;
	startMutual = -1.0f;
	mutualIncrease = -1.0f;
	colorStdFactor = -1.0f;
	reduceFactor = -1.0f;
	alphaFactor = 1.0f;
	textureFactor = 1.0f;
	normFactor = 1.0f;
	numOfIterations = -1;
	startSize = -1;
	endSize = -1; 

	// While we aren't at the end of the file
	while(!feof(fp)) {

		// Read line from config file
		fscanf(fp, "%s\t%f\n", strName, &val);

		// See which variable is matched (see globals.h for a
		// description of each variable)
		if(!strcmp("STD2", strName)) {

			 variance = val;

		} else if(!strcmp("RepeatTime", strName)) {

			numOfIterations = (int) val;

		} else if(!strcmp("BlockStart", strName)) {

			startSize = (int) val;

		} else if(!strcmp("BlockEnd", strName)) {

			endSize = (int) val;

		} else if(!strcmp("BlockReduce", strName)) {

			blockReduceSize = (int) val;

		} else if(!strcmp("StdIncrease", strName)) {

			stdIncrease = val;

		} else if(!strcmp("CorrespondingFactor", strName)) {

			featureStdFactor = (int) val;

		} else if(!strcmp("VarFactor", strName)) {

			randVarFactor = val;

		} else if(!strcmp("StartFactor", strName)) {

			colorStdFactor = val;

		} else if(!strcmp("DenoiseReduceFactor", strName)) {

			reduceFactor = val;

		} else if(!strcmp("StartMutual", strName)) {

			startMutual = val;

		} else if(!strcmp("MutualIncrease", strName))	{

			mutualIncrease = val;

		} else if(!strcmp("WorldFactor", strName)) {

			worldStdFactor = (int) val;

		} else if(!strcmp("AlphaFactor", strName)) {

			alphaFactor = val;

		} else if(!strcmp("TextureFactor", strName)) {

			textureFactor = val;

		} else if(!strcmp("NormFactor", strName)) {

			normFactor = val;

		} else if(!strcmp("HasBlockSizes", strName)) {

			hasExplicitBlocks = (bool) val;

		} 

	}

	// Array specifying the block sizes to use for each iteration
	blockSizes = new int[numOfIterations];
	memset(blockSizes, 0, numOfIterations * sizeof(int));
	
	// If blocks are explicitly named in config file
	if(hasExplicitBlocks) {
		
		// Reset file to beginning
		rewind(fp);
		int blockNumber = 0;
		int size = 0;
		
		// Read block files from config file
		while(!feof(fp)) {

			// Read line from config file
			fscanf(fp, "%s\t%d\t%d\n", strName, &blockNumber, &size);
			if(!strcmp("BLOCK", strName)) {
				if(blockNumber < numOfIterations) {
					blockSizes[blockNumber] = size;
				} else {
					fprintf(stderr, "Error: Config file has too many block sizes");
					getchar();
					exit(-1);
				}
			}

		}

		// Make sure all iterations have a block size
		for(int i = 0; i < numOfIterations; i++) {
			if(blockSizes[i] == 0) {
				fprintf(stderr, "Error: Config file doesn't have all block sizes");
				getchar();
				exit(-1);
			}
		}
	
	} else {

		// If blocks aren't specified then revert to default form of config file
		// where blocks are reduced by a set amount each iteration
		for(int i = 0; i < numOfIterations; i++) {

			blockSizes[i] = MAX(startSize + i*blockReduceSize, endSize);
		
		}

	}

}

void checkParameters() {

	// Error check all the parameters 
	// Most checks are to see that the parameter is non-negative.

	if(variance <= 0.0f) {

		fprintf(stdout, "ERROR: Variance is either not specified or is negative (or zero). Please check config file\n");

	} 
	
	if(numOfIterations <= 0) {

		fprintf(stdout, "ERROR: Number of iterations is either not specified or is negative (or zero). Please check config file\n");

	}

	if(!hasExplicitBlocks) {

		if(startSize <= 0) {

			fprintf(stdout, "ERROR: Block start size is either not specified or is negative (or zero). Please check config file\n");

		} 
	
		if(endSize <= 0) {

			fprintf(stdout, "ERROR: Block end size is either not specified or is negative (or zero). Please check config file\n");

		} 
	
		if(blockReduceSize >= 0) {

			fprintf(stdout, "ERROR: Block reduce size is either not specified or is positive (or zero). Please check config file\n");

		} 

	}
	
	if(stdIncrease < 0.0f) {

		fprintf(stdout, "ERROR: Standard deviation increase (STD_INCREASE) is either not specified or is negative. Please check config file\n");

	} 
	
	if(featureStdFactor <= 0) {

		fprintf(stdout, "ERROR: Corresponding factor is either not specified or is negative (or zero). Please check config file\n");

	} 
	
	if(randVarFactor <= 0.0f) {

		fprintf(stdout, "ERROR: Var Factor is either not specified or is negative (or zero). Please check config file\n");

	} 
	
	if(colorStdFactor <= 0.0f) {

		fprintf(stdout, "ERROR: Color Standard factor (START_FACTOR) is either not specified or is negative (or zero). Please check config file\n");

	} 
	
	if(reduceFactor < 0.0f) {

		fprintf(stdout, "ERROR: Reduce factor is either not specified or is negative. Please check config file\n");

	} 
	
	if(startMutual < 0.0f) {

		fprintf(stdout, "ERROR: Start mutual is either not specified or is negative. Please check config file\n");

	} 
	
	if(mutualIncrease < 0.0f)	{

		fprintf(stdout, "ERROR: Mutual increase is either not specified or is negative. Please check config file\n");

	} 
	
	if(worldStdFactor <= 0) {

		fprintf(stdout, "ERROR: World Standard factor is either not specified or is negative (or zero). Please check config file\n");

	}

}

uchar convertDoubleToByte(double value) {

	// Set value to be positive
	if(value < 0) {
		value = -value;
	} 
	
	// Clamp value to 1
	if(value > 1) {
		value = 1;
	}

	// Return byte (color)
	return (uchar) ((value)*255 + 0.5);

}

void boxFilter(CImg<float>* img, bool calculateOriginal) {

	// Loop through all the pixels in the image
	for(size_t i = 0; i < height; i++) {
		for(size_t j = 0; j < width; j++) {

			// Box filter samples in pixel P to compute final pixel value

			// Initialize variables
			float color[3] = {0.0f, 0.0f, 0.0f};

			// The index for the data of the current pixel
			size_t index = samplesPerPixel * ((width * i) + j);

			// Go through all the samples for the current pixel
			for(size_t k = 0; k < samplesPerPixel; k++) {

				// Initialize variables
				float c1 = 0, c2 = 0, c3 = 0;

				// Save the original (Monte Carlo) output
				if(calculateOriginal) {

					samples->getColor(index + k, c1, c2, c3);

				// Save the filtered output
				} else {

					samples->getCPrime(index + k, c1, c2, c3);

				}
				
				// Add to the running sum 
				color[0] += c1;
				color[1] += c2;
				color[2] += c3;

			}
			
			// Calculate the average color for each channel and save it
			assert(samplesPerPixel != 0);
			float scalar = 1.0f / samplesPerPixel;
			for(int k = 0; k < NUM_OF_COLORS; k++) {

				color[k] *= scalar;
				(*img)(j,i,0,k) = color[k];
				
			}

		}
	} 

}


