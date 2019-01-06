#ifndef	PRECOMPUTEGAUSSIAN_H_INCLUDED
#define PRECOMPUTEGAUSSIAN_H_INCLUDED

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "Globals.h"

class PreComputeGaussian {

public:

	// Initialize table in constructor
	PreComputeGaussian();

	// Precompute exponentials and store them in a table for filtering step
	static void initializeTable();

	// Lookup precomputed exponential (corresponding to the normalizedDistance)
	// from the table. The normalizedDistance is calculated as the distance divided 
	// by the std
	static float gaussianDistance(float x_fDistanceDivByStd);

};

void computeExp(float* data, size_t size);
void generateRandomNormal(float* data, size_t size, float sigma);

#endif