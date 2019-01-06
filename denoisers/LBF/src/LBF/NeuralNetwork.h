#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Matrix.h"
//#include "../core/timer.h"
#include "Globals.h"
#include <cublas_v2.h>

extern cudaArray* devSampleArray;
extern cudaChannelFormatDesc channelDesc;

class NeuralNetwork {

public:

	NeuralNetwork(void);
	~NeuralNetwork(void);

	NeuralNetwork(char* inputFolder, int width, int height, int featureLength);

    void ApplyWeightsAndFilter(char* inputFolder, char* sceneName, float* featureData, float* varData, float* img);
	
private:

	int numOfLayers;
	int* layerSizes;
	int width;
	int height;
	int featureLength;

	FUNC* activationFuncsPerLayer;
	cublasHandle_t handle;

	int blockSize;
	int halfBlock;

	Matrix<float>* devWeights;
	Matrix<float>* hostWeights;

	void FeedForward(Matrix<float>& input, Matrix<float>* devAct);
	void LoadNodeData(Matrix<float>* devData, Matrix<float>* hostData, char* fileName);
	void SpikeRemoval(Matrix<float>& devFilteredImg, int h, int w);

	// Helper cublas function
	void MatMultAB(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& B, Matrix<float>& C);

};


#endif

