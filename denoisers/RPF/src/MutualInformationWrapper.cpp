#include "MutualInformationWrapper.h"

float getMutualInformation(float* pdfX, float* pdfY, float* pdfXY, int size1, int size2) {

	// Initialize mutual information
	float muInf = 0.0f;

	float* tempJointData = new float[size1 * size2];
	memset(tempJointData, 0, size1 * size2 * sizeof(float));
	int tempIndex = 0;

	// Loop through the x pdf
	for(int i = 0; i < size1; i++) {
		
		// No point in checking y pdfs if x pdf has zero probability here
		if(pdfX[i] == 0) {
			tempIndex += size2;
			continue;
		}
		
		// Loop through the y pdf
		for(int j = 0; j < size2; j++) {

			float joint = pdfXY[(i * size2) + j]; 

			tempJointData[tempIndex++] = joint;

			// If we have nonzero probabilities then calculate mutual information
			if(joint != 0 && pdfY[j] != 0) {
				
				muInf += joint * logf(joint / (pdfX[i] * pdfY[j]));
			
			}
		}
	} 

	assert(tempIndex == (size1 * size2));

	// Check for NaNs and infinite numbers
	assert(muInf == muInf);
	assert(muInf != numeric_limits<float>::infinity());

	delete[] tempJointData;

	return muInf;
	
} 

int quantizeVector(float* src, int* dest, size_t len) {
	
	// Quantize the source into integer bins
	int numOfBins = 0;
	copyvecdata(src, len, dest, numOfBins);
	numOfBins = MAX(1, numOfBins);
	assert(numOfBins <= MAX_NUM_OF_BINS);
	return numOfBins;

}

template <class T> void copyvecdata(T * srcdata, size_t len, int * desdata, int& size) {

	// Argument checking
	if(!srcdata || !desdata) {
		printf("NULL pointers in copyvecdata()!\n");
		return;
	} 

	
	// Assign temporary max and min values
	size_t i;
	int minn,maxx;
	if (srcdata[0] > 0) {
		maxx = minn = int(srcdata[0]+0.5);
	} else {
		maxx = minn = int(srcdata[0]-0.5);
	}

	// Quantize the data and update min and max
	int tmp;
	for (i = 0; i < len; i++) {
		tmp = (srcdata[i] > 0) ? (int)(srcdata[i] + 0.5):(int)(srcdata[i] - 0.5); //round to integers
		minn = MIN(minn, tmp);
		maxx = MAX(maxx, tmp);
		desdata[i] = tmp;
	}

	// Make the vector data begin from 0 (i.e. 1st state)
	for(i=0; i<len; i++) {
		desdata[i] -= minn; 
	}

	// Find the number of bins
	size = (maxx-minn+1);
	
	// If the number of bins is greater than the max than separte the data into
	// the max number of bins
	if(size > MAX_NUM_OF_BINS) {
		float delta = float(size) / float(MAX_NUM_OF_BINS - 1);
		float deltaInv = 1.0f / delta;
		for(int i = 0; i < len; i++) {
			int temp = int(desdata[i]*deltaInv);
			assert(temp < MAX_NUM_OF_BINS);
			assert(temp >= 0);
			desdata[i] = temp;
		}
		size = MAX_NUM_OF_BINS;
	} 

	return;

}
