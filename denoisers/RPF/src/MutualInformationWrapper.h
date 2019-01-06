#ifndef	MUTUALINFORMATIONWRAPPER_H_INCLUDED
#define MUTUALINFORMATIONWRAPPER_H_INCLUDED

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
//#include <windows.h>
#include "Globals.h"

// Finds the mutual information of X and Y Random Variables with pdf X of size1 and pdf Y of size2 and joint pdf
// of size1 * size2
float getMutualInformation(float* pdfX, float* pdfY, float* pdfXY, int size1, int size2); 

// Quantize the source vector into an integer vector and return the number of bins
int quantizeVector(float* src, int* dest, size_t len);

// Copy the source data of length (len) into the destination and save the number of integer bins in size
template <class T> void copyvecdata(T * srcdata, size_t len, int * desdata, int& size); 

#endif
