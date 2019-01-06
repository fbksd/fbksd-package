#include "SSE_Wrapper.h"

#if ENABLE_SSE

float calculateMeanSSE(float* src, size_t length) {

	// Arg checking
	assert(length >= 4);

	// Calculate how many groups of four there are
	// and how many extras there are
	size_t loopSize = length / 4;
	size_t numOfExtras = length % 4;
	
	// Group the data into groups of four
    v4sf* data = (v4sf*) src;

	// Initialize the sum to zero
	v4sf cumulativeSum = _mm_set_ps1(0.0f);

	// Add up all of the elements
    for(size_t i = 0; i < loopSize; i++ ) {

		cumulativeSum = _mm_add_ps(*data, cumulativeSum);            

		data++; // Move on to the next four

    }
	
	// Sum up the four entries to get the total sum so far
	float sum = cumulativeSum.m128_f32[0] + cumulativeSum.m128_f32[1] + cumulativeSum.m128_f32[2] + cumulativeSum.m128_f32[3];

	// Add in the extras
	for(size_t i = 0; i < numOfExtras; i++) {

		sum += (*data).m128_f32[i];

	}

	// Divide by the length to calculate the mean
	float mean = sum / length;

	// Return the mean of the vector
	return mean;

}

float calculateStdSSE(float* src, float mean, size_t length) {

	// Arg checking
	assert(length >= 4);

	// Calculate how many groups of four there are
	// and how many extras there are
	size_t loopSize = length / 4;
	size_t numOfExtras = length % 4;
	
	// Group the data into groups of four
    v4sf* data = (v4sf*) src;

	// Set up 4 copies of the mean 
	v4sf mu = _mm_set_ps1(mean);

	// Temporary variable to hold the intermediate results
	// when calculating the standard deviation
	v4sf temp;

	// Cumulative sum of the squared differences
	v4sf cumulativeSum = _mm_set_ps1(0.0f);

	// Add up all of the elements
    for(size_t i = 0; i < loopSize; i++ ) {
         
		// Sum of squared differences
		temp = _mm_sub_ps(*data, mu);
        temp = _mm_mul_ps(temp, temp);
		cumulativeSum = _mm_add_ps(cumulativeSum, temp);

		data++; // Move on to the next four

    }
	
	// Add up the total so far
	float squaredSum = cumulativeSum.m128_f32[0] + cumulativeSum.m128_f32[1] + cumulativeSum.m128_f32[2] + cumulativeSum.m128_f32[3];

	// Add in the extras
	for(size_t i = 0; i < numOfExtras; i++) {

		squaredSum += pow(((*data).m128_f32[i] - mean), 2);

	}

	// Calculate the variance
	float variance = squaredSum / (length - 1); 

	// Return the square root of the variance (standard deviation)
	return sqrtf(variance);

}

void normalizeSSE(float* src, float mean, float std, float* result, size_t length) {

	// Arg checking
	assert(length >= 4);

	// Calculate how many groups of four there are
	// and how many extras there are
	size_t loopSize = length / 4;
	size_t numOfExtras = length % 4;
	float stdInv;
	if(fabs(std) > EPSILON) {
		stdInv = 1.0f / std;
	} else {
		stdInv = 1.0f / EPSILON;
	}
	
	// Group the data into groups of four
    v4sf* data = (v4sf*) src;
	v4sf* dest = (v4sf*) result;

	// Set up 4 copies of the mean and standard deviation
	v4sf mu = _mm_set_ps1(mean);
	v4sf sigmaInv = _mm_set_ps1(stdInv);

	// Add up all of the elements
    for(size_t i = 0; i < loopSize; i++ ) {
         
		*dest = _mm_sub_ps(*data, mu);
		*dest = _mm_mul_ps(*dest, sigmaInv);

		data++; // Move on ot the next four
		dest++;

    }

	// Add in the extras
	for(size_t i = 0; i < numOfExtras; i++) {

		(*dest).m128_f32[i] = ((*data).m128_f32[i] - mean) * stdInv;

	}

}

void subtractSquareScaleSSE(float* data1, float* data2, float* result, float alpha, size_t length) {

	// Calculate how many groups of four there are
	// and how many extras there are
	size_t loopSize = length / 4;
	size_t numOfExtras = length % 4;

	// Group the data into groups of four
    v4sf* d1 = (v4sf*) data1;
	v4sf* d2 = (v4sf*) data2;
	v4sf* dest = (v4sf*) result;
	v4sf scalar = _mm_set_ps1(alpha);
	v4sf temp;

	// Filter the colors of samples in pixel using bilateral filter
	for(size_t i = 0; i < loopSize; i++) {
 
		temp = _mm_sub_ps(*d1, *d2);
		temp = _mm_mul_ps(temp, temp);
		*dest = _mm_mul_ps(temp, scalar);
		
		dest++; // Move on to the next four
		d1++;
		d2++;

	}
		
	// Add in the extras
	for(size_t i = 0; i < numOfExtras; i++) {

		(*dest).m128_f32[i] = alpha * pow(((*d1).m128_f32[i] - (*d2).m128_f32[i]), 2);

	} 

}

void sumSSE(float* data1, float* data2, float* result, size_t length) {

	// Calculate how many groups of four there are
	// and how many extras there are
	size_t loopSize = length / 4;
	size_t numOfExtras = length % 4;

	// Group the data into groups of four
    v4sf* d1 = (v4sf*) data1;
	v4sf* d2 = (v4sf*) data2;
	v4sf* dest = (v4sf*) result;

	// Filter the colors of samples in pixel using bilateral filter
	for(size_t i = 0; i < loopSize; i++) {
 
		*dest = _mm_add_ps(*d1, *d2);

		dest++; // Move on to the next four
		d1++;
		d2++;
		
	}
		
	// Add in the extras
	for(size_t i = 0; i < numOfExtras; i++) {

		(*dest).m128_f32[i] = (*d1).m128_f32[i] + (*d2).m128_f32[i];

	}

}

void multByValSSE(float* data, float* result, float alpha, size_t length) {

	// Calculate how many groups of four there are
	// and how many extras there are
	size_t loopSize = length / 4;
	size_t numOfExtras = length % 4;

	// Group the data into groups of four
    v4sf* d = (v4sf*) data;
	v4sf* dest = (v4sf*) result;
	v4sf scalar = _mm_set_ps1(alpha);
	
	// Filter the colors of samples in pixel using bilateral filter
	for(size_t i = 0; i < loopSize; i++) {
 
		*dest = _mm_mul_ps(*d, scalar);
		
		dest++; // Move on to the next four
		d++;

	}
		
	// Add in the extras
	for(size_t i = 0; i < numOfExtras; i++) {

		(*dest).m128_f32[i] = alpha * (*d).m128_f32[i];

	}

}

void multSSE(float* data1, float* data2, float* result, size_t length) {

	// Calculate how many groups of four there are
	// and how many extras there are
	size_t loopSize = length / 4;
	size_t numOfExtras = length % 4;

	// Group the data into groups of four
    v4sf* d1 = (v4sf*) data1;
	v4sf* d2 = (v4sf*) data2;
	v4sf* dest = (v4sf*) result;
	
	// Filter the colors of samples in pixel using bilateral filter
	for(size_t i = 0; i < loopSize; i++) {
 
		*dest = _mm_mul_ps(*d1, *d2);
		
		dest++; // Move on to the next four
		d1++;
		d2++;

	}
		
	// Add in the extras
	for(size_t i = 0; i < numOfExtras; i++) {

		(*dest).m128_f32[i] = (*d1).m128_f32[i] * (*d2).m128_f32[i];

	}

}

float sumAllElementsSSE(float* data, size_t length) {
	
	// Calculate how many groups of four there are
	// and how many extras there are
	size_t loopSize = length / 4;
	size_t numOfExtras = length % 4;

	// Group the data into groups of four
    v4sf* d = (v4sf*) data;
	v4sf temp = _mm_set1_ps(0.0f);
	
	// Filter the colors of samples in pixel using bilateral filter
	for(size_t i = 0; i < loopSize; i++) {
 
		temp = _mm_add_ps(*d, temp); 
		
		d++; // Move on to the next four
	}
	
	float result = temp.m128_f32[0] + temp.m128_f32[1] + temp.m128_f32[2] + temp.m128_f32[3];

	// Add in the extras
	for(size_t i = 0; i < numOfExtras; i++) {

		result += (*d).m128_f32[i];

	}

	return result;

}

#endif
