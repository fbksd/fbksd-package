#ifndef	SAMPLESET_H_INCLUDED
#define SAMPLESET_H_INCLUDED

#include <vector>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <omp.h>
#include "MutualInformationWrapper.h"
#include "SSE_Wrapper.h"

using namespace std;

void generateRandomNormal(float* x_pfVector, size_t x_nSize, float x_fSigma);

class SampleSet {

public:

	// Constructors and destructor
	SampleSet(size_t size, bool initializeData); 
	~SampleSet();

	// Read the samples from the file and add the data to 
	// the appropriate arrays
	void readSamples(FILE* fp);
	void readSamples(float* pbrtData);
	void orderSamples(float* data);

	// Initialize cPrime
	void initializeCPrime();
	
	// Calculate mean, std, and normalize samples
	void calculateStatistics();

	void clampColors(float* data);

	// Copy size feature vectors from source to this set at pixel coordinates x, y
	void copyData(SampleSet* src, int x, int y, size_t size); 

	// Copy one feature vector from src at srcIndex and save in this set at destIndex
	void copyData(SampleSet* src, size_t srcIndex, size_t destIndex);

	// Check if any parts of the feature vector are outliers. If an outlier is detected,
	// then keepSample is false
	void checkFeatures(float* sampleFeatures, bool& keepSample);

	// Normalize all data from each sample by subtracting the mean and dividing 
	// by the standard deviation
	void normalizeSamples();

	// Initialize the pdfs and joint pdfs from histograms of the quantized sample data
	void initializePDFs();

	// Calculates the mutual information between the arrays specified by specifier 1 and 2
	// and at indexes 1 and 2
	float calculateDependency(int specifier1, int specifier2, int index1, int index2, int jointIndex);
		
	// Update cDoublePrime with the src pixel values
	void updateColor(SampleSet* src, int x, int y, size_t size);
	
	// Generate size gaussian random values with std sigma and save in randomIndexes
	void generateRandomValues(size_t size, float sigma);

	// Chooses a random offset into the randomIndexes array to use as random values
	void generateRandomValues(size_t size);

	// Choose random samples that fall in the box of size b centered at pixel coordinate x, y.
	// The function chooses size random samples and the usable ones are count. This count is
	// stored in numOfSamplesUsed
	int* generateIndexes(int b, int x, int y, size_t size, size_t& numOfSamplesUsed);
			
	// Find the mean and standard deviation of the feature vector for the pixel samples
	void calculatePixelStatistics();

	// Remove the color spikes in each channel that are above colorStdScalar std deviations above 
	// the mean of the pixel samples
	void spikeRemoval(float colorStdScalar); 

	// Loop through all samples and have them update their colors
	void setColors();

	// Getters and Setters (See comments of member data below)
	size_t getTotalSize();
	size_t getNumOfSamples();
	void setNumOfSamples(size_t numOfSamples);
	void getColor(size_t index, float& c0, float& c1, float& c2); 
	void getCPrime(size_t index, float& c0, float& c1, float& c2);
	void getFeature(size_t featureOffset, size_t sampleIndex, float& x);
	void getNormColor(size_t index, float& c0, float& c1, float& c2);
	void getNormFeature(size_t featureOffset, size_t sampleIndex, float& x);
	float* getCPrime(size_t index);
	float* getSampleInfo(size_t index);
	float* getCPrime();
	float* getNormColor(size_t index);
	float* getNormFeature(size_t index); 
	float* getPixelColors();
	float* getFeatures(size_t index);
	float* getNormColorBuffer();
	float* getNormFeatureBuffer();
	float* getColorBuffer();
	float* getFeatureBuffer();
	float* getTempStorageBuffer();
	float* getWeightsBuffer();
	float* getTempWeightsBuffer();
	float* getAlpha();
	float* getBeta();
	void setPixelColors(size_t index, float c0, float c1, float c2);

private:

	bool allEqualToZero(float* data, size_t size);

	// Calculate the feature mean for all the samples in this set
	void calculateMean();

	// Calculate the standard deviation for all the samples in this set
	void calculateStandardDeviation();

	// Take the histogram (quantizedSource) with size bins and generate a pdf
	void computePDF(int* quantizedSource, float* pdf, size_t size);

	// Take the two histograms (quantizedP1 and quantizedP2) of size p1Size and p2Size
	// respectively and generate a jointPdf
	void computeJointPDF(int* quantizedP1, int* quantizedP2, float* jointPdf, int p1Size, int p2Size);

	// Find the median of the data array
	float findMedian(float* data);

	// Data has all the sample data for the set stored in the following way sample1_Feature1 sample2_Feature1
	// ... sampleTotalSize_Feature1, sample1_Feature2, etc... Ie. All the data is grouped by features rather 
	// than samples. This array is of size SAMPLE_LENGTH * totalSize
	float* data;

	// A 2D array that has the same info as the data array but organized in a different way. 
	// The grouping here is by sample rather than by feature, so each pointer points to the 
	// entire sample vector for each sample
	float** sampleInfo;

	// The mean for each feature. This array is of SAMPLE_LENGTH
	float* dataMu;

	// The std for each feature. This array is of SAMPLE_LENGTH
	float* dataSigma;

	// The above data array but subtracted by the set's mean and divided by the set's std
	float* normalizedData;

	// An array of pdfs. Each pdf is separated by MAX_NUM_OF_BINS. The array is of size
	// MAX_NUM_OF_BINS * SAMPLE_LENGTH
	float* PDFs;

	// Quantized version of the data array. This array is of size SAMPLE_LENGTH * totalSize
	int* histograms;

	// These buffers are used for filtering the colors. They are aligned and used for SSE.
	// They correspond to the normalized colors and features, the un-normalized colors and
	// features, weights, and temporary storage
	float* normColorBuffer; 
	float* normFeatureBuffer;
	float* colorBuffer;
	float* featureBuffer;
	float* tempStorageBuffer;
	float* weightsBuffer;
	float* tempWeightsBuffer;

	// The mean of only the FEATURE part of the sample vector for the pixel samples 
	float* pixelFeatureMu;

	// The std of only the FEATURE part of the sample vector for the pixel samples
	float* pixelFeatureSigma; 
	
	// The alpha array is NUM_OF_COLORS long and has the color weights
	float* alpha;

	// The beta array is NUM_OF_FEATURES long and has the feature weights
	float* beta;

	// Refers to the number of bins for the histograms for the data
	// Order: First three are for color, next two are for position,
	// next NUM_OF_RANDOM_PARAMETERS are for the random parameters,
	// and the next NUM_OF_FEATURES are for the features
	int* binSizes; 

	// Number of samples in the set
	size_t numOfSamples;

	// Max number of samples that the set can hold
	size_t totalSize;

	// Booleans specifying which data was initialized. This is because the 
	// global sample set ("samples") which holds all the sample data doesn't
	// need to initialize most of these arrays and therefore these arrays
	// don't need to be de-allocated in the destructor. However, at each
	// pixel the sample sets need this data so to differentiate the two
	// cases these booleans are used and the appropriate arrays are de-allocated
	bool additionalDataInitialized;
	bool randomValuesInitialized; // Arrays for random numbers initialized

	// A large allocation of randomly generated gaussian numbers
	float* randomIndexes;

	// The random values for this iteration
	float* currentRandomIndexes;
	
	// Holds the count of samples in the block at each x,y coordinate in the form x, y, count
	int* integerIndexes;

	// This array is NUM_OF_COLORS * samplesPerPixel long. It holds the pixel colors for
	// each sample for the pixel samples (the center pixel of this set)
	float* pixelColors;

	// CPrime refers to the final colors at the end of the iteration and for the start of the following
	// iteration. CDoublePrime is a temporary storage where new colors are stored during a given iteration. 
	// At the end of the iteration, cPrime is updated with the values from cDoublePrime.
	// See paper for more info on how these are used
	float* cPrime;
	float* cDoublePrime; 
	
};

#endif