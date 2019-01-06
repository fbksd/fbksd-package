#include "PreComputeGaussian.h"

PreComputeGaussian::PreComputeGaussian() {

	initializeTable();

}

void PreComputeGaussian::initializeTable() {
	
	// Populate table of exponentials starting from zero, incremented by
	// gaussianDelta and of length GAUSSIAN_LENGTH
	float _fVal = 0;
	for(int i = 0 ; i < GAUSSIAN_LENGTH; i++) {
		gaussianTable[i] = expf(-_fVal);
		_fVal += gaussianDelta;
	}

}

float PreComputeGaussian::gaussianDistance(float normalizedDistance) {

	// Make value positive
	if(normalizedDistance < 0) {
		normalizedDistance *= -1;
	}

	// Lookup value of exp(-normalizedDistance), where normalizedDistance
	// distance divided by std
	float expVal = 0;
	if(normalizedDistance < GAUSSIAN_CUT) {

		normalizedDistance *= gaussianDeltaInv;
		expVal = gaussianTable[(int) normalizedDistance];

		return expVal;
	}

	return 0;
}

void computeExp(float* data, size_t size) {

	// Do size number of lookups
	for(size_t i = 0 ; i < size ; i++) {

		// Look up exp ^ currentElement in precomputed table
		data[i] = PreComputeGaussian::gaussianDistance(data[i]);
		
	}

}

void generateRandomNormal(float* data, size_t size, float sigma){

	// Using the Box-Muller method to generate gaussian random variable from random uniform variable
	for(size_t i = 0; i < size; i++){
		float u = float(rand())/RAND_MAX + EPSILON; // EPSILON is added to avoid NAN error when taking log of s
		float v = float(rand())/RAND_MAX;

		u = u <= 1 ? u : 1.0f; // making sure the maximum remains 1

		data[i] = (float) (sqrtf( -2 * logf(u) ) * cos(2 * M_PI * v) * sigma);
	}

}
