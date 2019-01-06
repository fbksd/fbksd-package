#ifndef	MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <stdio.h>
#include <math.h>
#include <stddef.h> 
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "Utilities.h"
#include "CImg.h"

using namespace std;
using namespace cimg_library;

// Removed copy constructor and equal operators to control the cudamalloc and cudamemcpy's.

template<class T>
class Matrix {

public:

	//***** CONSTRUCTORS AND DESTRUCTOR *****// 

	Matrix();
	Matrix(int width, int height);
	Matrix(int width, int height, int depth);
	Matrix(const Matrix<T>& A);
	~Matrix();


	//***** OVERLOADED OPERATORS *****//
	
	void operator*=(T alpha);
	void operator=(Matrix<T> B);
	
	//***** UTILITY FUNCTIONS *****//

	Matrix<T> crop(size_t cropTop, size_t cropBottom, size_t cropLeft, size_t cropRight);
	Matrix<T> crop(size_t cropTop, size_t cropBottom, size_t cropLeft, size_t cropRight, size_t cropFront, size_t cropBack);
	Matrix<T> tonemap();

	void SetEqualTo(Matrix<T>& B);
	void SetToZero();

	// Initialize all elements of matrix to specified val
	void initializeToVal(T val);

	// Initialize elements of matrix to be random values between the min and max
	void initializeToRandom(T randMin, T randMax);

	void display(char* title);
	void display(char* title, int depthInd);

	void save(char* filename, bool shouldTonemap);
	void save(char* filename, int depthInd);
	void saveToEXR(char* filename) ;
	void saveToFile(char* filename);
	void saveToFile(char* filename, int depthInd);

	void DeviceToHost(Matrix<T>& hostMat);
	void HostToDevice(Matrix<T>& deviceMat);
	void DeviceToHost();
	void HostToDevice();

	void Reshape(int width, int height);
	void Reshape(int width, int height, int depth);

	//***** GETTERS AND SETTERS *****//
	
	__host__ __device__ int getWidth() const;
	__host__ __device__ int getHeight() const;
	__host__ __device__ int getDepth() const;
	__host__ __device__ int getStride() const;
	__host__ __device__ T* getElements() const;
	
	bool getIsCudaMat() const;
	int getIndex(int x, int y);
	T getElement(int x, int y);
	T getElement(int index);
	void setWidth(int width);
	void setHeight(int height);
	void setDepth(int depth);
	void setIsCudaMat(bool isCudaMat);
	void setElement(int index, T element);
	void setElements(T* elements);
	void AllocateData(bool isCudaMat);

private:

	//***** DATA *****//

	int width;
	int height;
	int depth;
	T* elements;
	bool isCudaMat;

};


template<class T>
Matrix<T>::Matrix() {
	this->width = 0;
	this->height = 0;
	this->depth = 0;
	this->elements = NULL;
	this->isCudaMat = false;

}

template<class T>
Matrix<T>::Matrix(int width, int height) {

	this->width = width;
	this->height = height;
	this->depth = 1;
	this->elements = NULL;
	this->isCudaMat = false;

}


template<class T>
Matrix<T>::Matrix(int width, int height, int depth) {

	this->width = width;
	this->height = height;
	this->depth = depth;
	this->elements = NULL;
	this->isCudaMat = false;

}

template<class T>
void Matrix<T>::AllocateData(bool isCudaMat) 
{
	assert(this->elements == NULL);
	this->isCudaMat = isCudaMat;

	// Allocate memory and transfer elements over
	if(!isCudaMat) 
	{
		int size = this->width * this->height * this->depth;
		this->elements = new T[size];
	} 
	else 
	{
		int size = this->width * this->height * this->depth * sizeof(T);
		GpuErrorCheck(cudaMalloc(&(this->elements), size)); 
	}

}

template<class T>
void Matrix<T>::SetToZero() 
{
	assert(this->elements != NULL);
	
	if(isCudaMat)
	{
		int size = this->width * this->height * this->depth * sizeof(T);
		GpuErrorCheck(cudaMemset(this->elements, 0, size));
	}
	else if(!isCudaMat)
	{
		int size = this->width * this->height * this->depth;
		memset(this->elements, 0, size);
	}
}

template<class T>
void Matrix<T>::Reshape(int width, int height) 
{
	assert(width * height == this->width * this->height * this->depth);

	this->width = width;
	this->height = height;
	this->depth = 1;
}

template<class T>
void Matrix<T>::Reshape(int width, int height, int depth) 
{
	assert(width * height * depth == this->width * this->height * this->depth);

	this->width = width;
	this->height = height;
	this->depth = depth;
}


template<class T>
Matrix<T>::Matrix(const Matrix<T>& A) 
{

	assert(!A.getIsCudaMat());
	this->width = A.getWidth();
	this->height = A.getHeight();
	this->depth = A.getDepth();
	this->isCudaMat = A.getIsCudaMat();
	this->elements = NULL;

	if(A.getElements() != NULL) 
	{
		int size = this->width * this->height * this->depth;
		this->elements = new T[size];
		memcpy(this->elements, A.getElements(), this->width * this->height * this->depth * sizeof(T));
	} 

}

template<class T>
void Matrix<T>::operator=(Matrix<T> B) 
{
	assert(!B.getIsCudaMat());
	if(this == &B) {
		return;
	}
	
	if(this->width != B.getWidth() || this->height != B.getHeight() || this->depth != B.getDepth()) 
	{

		this->~Matrix();
		this->width = B.getWidth(); 
		this->height = B.getHeight();
		this->depth = B.getDepth();
		this->isCudaMat = B.getIsCudaMat();
	}

	if (B.elements != NULL)
	{
		this->elements = new T[this->width * this->height * this->depth];
		memcpy(this->elements, B.getElements(), this->width * this->height * this->depth * sizeof(T)); 
	}

}

template<class T>
Matrix<T>::~Matrix() 
{
//    if(elements != NULL)
//    {
//        if(!isCudaMat)
//        {
//            delete[] elements;
//            elements = NULL;
//        }
//        else if (isCudaMat)
//        {
//            // I was getting CUDA ERROR: invalid device pointer in this line.
//            // I had to comment the destructor as a workaround.
//            GpuErrorCheck(cudaFree(this->elements));
//            elements = NULL;
//        }
//        else
//        {
//            elements = NULL;
//        }
//    }
}

template<class T>
Matrix<T> Matrix<T>::crop(size_t cropTop, size_t cropBottom, size_t cropLeft, size_t cropRight) 
{

	assert(!isCudaMat);
	assert(this->width > (cropLeft + cropRight));
	assert(this->height > (cropTop + cropBottom));
	size_t cropWidth = this->width - (cropLeft + cropRight);
	size_t cropHeight = this->height - (cropTop + cropBottom);
	size_t cropDepth = this->depth;
	Matrix<T> croppedMat(cropWidth, cropHeight, cropDepth);
	croppedMat.AllocateData(false);
	T* croppedElements = croppedMat.getElements();

	size_t startX = cropLeft;
	size_t startY = cropTop;
	size_t startZ = 0;
	size_t endX = this->width - cropRight;
	size_t endY = this->height - cropBottom;
	size_t endZ = this->depth;

	size_t index = 0;
	for (size_t k = startZ; k < endZ; k++) {
		for(size_t i = startY; i < endY; i++) {
			for(size_t j = startX; j < endX; j++) {
				size_t pixelIndex = k * this->width * this->height + i * this->width + j;
				croppedElements[index] = this->elements[pixelIndex];
				index++;
			}
		}
	}
	assert(index == (croppedMat.getWidth() * croppedMat.getHeight() * croppedMat.getDepth()));

	return croppedMat;

}

template<class T>
Matrix<T> Matrix<T>::crop(size_t cropTop, size_t cropBottom, size_t cropLeft, size_t cropRight, size_t cropFront, size_t cropBack) 
{

	assert(!isCudaMat);
	assert(this->width > (cropLeft + cropRight));
	assert(this->height > (cropTop + cropBottom));
	assert(this->depth > (cropFront + cropBack));

	size_t cropWidth = this->width - (cropLeft + cropRight);
	size_t cropHeight = this->height - (cropTop + cropBottom);
	size_t cropDepth = this->depth - (cropFront + cropBack);

	Matrix<T> croppedMat(cropWidth, cropHeight, cropDepth);
	croppedMat.AllocateData(false);
	T* croppedElements = croppedMat.getElements();

	size_t startX = cropLeft;
	size_t startY = cropTop;
	size_t startZ = cropFront;
	size_t endX = this->width - cropRight;
	size_t endY = this->height - cropBottom;
	size_t endZ = this->depth - cropBack;

	size_t index = 0;
	for (size_t k = startZ; k < endZ; k++) {
		for(size_t i = startY; i < endY; i++) {
			for(size_t j = startX; j < endX; j++) {
				size_t pixelIndex = k * this->width * this->height + i * this->width + j;
				croppedElements[index] = this->elements[pixelIndex];
				index++;
			}
		}
	}
	assert(index == (croppedMat.getWidth() * croppedMat.getHeight() * croppedMat.getDepth()));

	return croppedMat;

}

template<class T>
void Matrix<T>::SetEqualTo(Matrix<T>& B) 
{

	assert(!isCudaMat && !B.getIsCudaMat());

	if(this == &B) 
	{
		return ;
	}

	if(this->width != B.getWidth() || this->height != B.getHeight() || this->depth != B.getDepth()) 
	{
		this->~Matrix();
		this->width = B.getWidth(); 
		this->height = B.getHeight();
		this->depth = B.getDepth();
		this->isCudaMat = B.getIsCudaMat();
		this->elements = new T[this->width * this->height * this->depth];
	}

	memcpy(this->elements, B.getElements(), this->width * this->height * this->depth * sizeof(T)); 

}


template<class T>
void Matrix<T>::initializeToVal(T val) {

	assert(!isCudaMat);

	for(int i = 0; i < this->height; i++) 
	{
		for(int j = 0; j < this->width; j++)
		{
			for(int k = 0; k < this->depth; k++)
			{
				int index = k * this->width * this->height + i * this->width + j;
				this->elements[index] = val;
			}
		}
	}
}

template<class T>
void Matrix<T>::initializeToRandom(T randMin, T randMax) {

	assert(!isCudaMat);
	for(int i = 0; i < this->height; i++) {
		for(int j = 0; j < this->width; j++) {

			int index = i * this->width + j;
			this->elements[index] = randMin + (randMax - randMin) * rand()/T(RAND_MAX);

		}
	}
}

template<class T>
Matrix<T> Matrix<T>::tonemap() 
{

	assert(!isCudaMat);
	Matrix<T> tonemapped(*this);
	T* tonemappedElements = tonemapped.getElements();
	for(int i = 0; i < this->width * this->height * this->depth; i++) {
		tonemappedElements[i] = 255 * MIN(1.0f, powf(this->elements[i], 1.0f / COLOR_GAMMA));
	}

	return tonemapped;

}

template<class T>
void Matrix<T>::save(char* filename, bool shouldTonemap) 
{

	assert(!isCudaMat);
	CImg<T> img(this->width, this->height, 1, this->depth);
	
	if(shouldTonemap) 
	{
		Matrix<T> temp = this->tonemap();
		memcpy(img.data(), temp.getElements(), this->width * this->height * this->depth * sizeof(T));
	} 
	else 
	{
		memcpy(img.data(), this->elements, this->width * this->height * this->depth * sizeof(T));
		img *= 255;
	}

	img.min(255);
	img.save(filename);


}



template<class T>
void Matrix<T>::save(char* filename, int depthInd) 
{

	assert(!isCudaMat);
	assert(depthInd < this->depth);

	CImg<T> img(this->width, this->height);
	
	memcpy(img.data(), this->elements + this->width * this->height * depthInd, this->width * this->height * sizeof(T));
	
	img *= 255;
	img.min(255);
	//img.normalize(0, 255);
	img.save(filename);


}

template<class T>
void Matrix<T>::saveToEXR(char* filename) 
{

	assert(!isCudaMat);

//	WriteEXRFile(filename, width, height, this->elements);


}


template<class T>
void Matrix<T>::saveToFile(char* fileName) 
{

	assert(!isCudaMat);

    FILE* fp = OpenFile(fileName, "wb");

	fwrite(&this->width, sizeof(int), 1, fp);
	fwrite(&this->height, sizeof(int), 1, fp);
	fwrite(&this->depth, sizeof(int), 1, fp);
	fwrite(this->elements, sizeof(T), this->width * this->height * this->depth, fp);

	fclose(fp);

}

template<class T>
void Matrix<T>::saveToFile(char* fileName, int depthInd) 
{

	assert(!isCudaMat);
	assert(depthInd < this->depth);

	FILE* fp = OpenFile(fileName, "wb");

	int depth = 1;

	fwrite(&this->width, sizeof(int), 1, fp);
	fwrite(&this->height, sizeof(int), 1, fp);
	fwrite(&depth, sizeof(int), 1, fp);
	fwrite(this->elements + this->width * this->height * depthInd, sizeof(T), this->width * this->height, fp);

	fclose(fp);

}

template<class T>
int Matrix<T>::getWidth() const {
	return width;
}

template<class T>
int Matrix<T>::getHeight() const {
	return height;
}

template<class T>
int Matrix<T>::getDepth() const {
	return depth;
}

template<class T>
int Matrix<T>::getIndex(int x, int y) {
	assert(x >= 0 && x < width);
	assert(y >= 0 && y < height);
	return y * width + x;
}

template<class T>
T* Matrix<T>::getElements() const {
	return elements;
}

template<class T>
T Matrix<T>::getElement(int x, int y) {
	assert(this->depth == 1);
	int index = getIndex(x, y);
	assert(index < width * height);
	return getElement(index);
}


template<class T>
T Matrix<T>::getElement(int index) {
	assert(!isCudaMat);
	return elements[index];
}

template<class T>
bool Matrix<T>::getIsCudaMat() const {
	return isCudaMat;
}

template<class T>
void Matrix<T>::setWidth(int width) {
	this->width = width;
}

template<class T>
void Matrix<T>::setHeight(int height) {
	this->height = height;
}


template<class T>
void Matrix<T>::setDepth(int depth) {
	this->depth = depth;
}

template<class T>
void Matrix<T>::setElement(int index, T element) {
    assert(!isCudaMat/* && !isMemManaged*/);
	this->elements[index] = element;
}

template<class T>
void Matrix<T>::setElements(T* elements) {
	this->~Matrix();
	this->elements = elements;
} 

template<class T>
void Matrix<T>::setIsCudaMat(bool isCudaMat) {
	this->isCudaMat = isCudaMat;
}

template<class T>
void Matrix<T>::DeviceToHost(Matrix<T>& hostMat) 
{

	assert(isCudaMat);
	assert(hostMat.getWidth() == this->width && hostMat.getHeight() == this->height && hostMat.getDepth() == this->depth);
	GpuErrorCheck(cudaMemcpy(hostMat.getElements(), this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyDeviceToHost));

}

template<class T>
void Matrix<T>::HostToDevice(Matrix<T>& deviceMat) 
{

	assert(!isCudaMat);
	assert(deviceMat.getWidth() == this->width && deviceMat.getHeight() == this->height && deviceMat.getDepth() == this->depth);
	GpuErrorCheck(cudaMemcpy(deviceMat.getElements(), this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyHostToDevice));

}

template<class T>
void Matrix<T>::DeviceToHost() 
{
	assert(isCudaMat);
	
	T* destElements = new T[this->width * this->height * this->depth];
	GpuErrorCheck(cudaMemcpy(destElements, this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyDeviceToHost));
	
	GpuErrorCheck(cudaFree(this->elements));
	this->elements = destElements;
	this->isCudaMat = false;

}

template<class T>
void Matrix<T>::HostToDevice() {

	assert(!isCudaMat);
	
	T* destElements;
	GpuErrorCheck(cudaMalloc(&(destElements), this->width * this->height * this->depth * sizeof(T))); 
	GpuErrorCheck(cudaMemcpy(destElements, this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyHostToDevice));
	
	delete[] this->elements;
	this->elements = destElements;
	this->isCudaMat = true;

}

template<class T>
void Matrix<T>::display(char* title) 
{

	CImg<T> img(this->width, this->height, 1, this->depth);
	if(!isCudaMat) 
		memcpy(img.data(), this->elements, this->width * this->height * this->depth * sizeof(T));
	else
		cudaMemcpy(img.data(), this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyDeviceToHost);

	img.display(title);

}

template<class T>
void Matrix<T>::display(char* title, int depthInd) 
{

	CImg<T> img(this->width, this->height);
	if(!isCudaMat) 
		memcpy(img.data(), this->elements + depthInd * this->width * this->height, this->width * this->height * sizeof(T));
	else
		cudaMemcpy(img.data(), this->elements + depthInd * this->width * this->height, this->width * this->height * sizeof(T), cudaMemcpyDeviceToHost);

	img.display(title);

}

#endif





