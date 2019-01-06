#ifndef CUDAWRAPPERS_H
#define CUDAWRAPPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "Globals.h"
#include "Matrix.h"

cudaChannelFormatDesc channelDesc;
cudaArray* devSampleArray;
texture<float4, cudaTextureType2DLayered> textures;

__global__ void CudaBoxFilter(float* dst, float* src, int halfBlock, int width, int height, int numOfChannels, int xOffset, int yOffset, int srcOffset);
__global__ void CudaDistance(float* distance, float* alpha, float* variances, int width, int height, int numChannels, int indTex, int dx, int dy, TERMTYPE type, float inputAlpha);
__global__ void CudaGetWeights(float* dst, float* src, int width, int height);
__global__ void CudaApplyWeights(float* dst, float* weights, float* totalWeight, int width, int height, int numOfChannels, int j, int i, int filterOffset);
__global__ void CudaNormalize(float* filteredData, float* weights, int width, int height, int numOfChannels);
__global__ void CudaApplyWeightsToVar(float* dst, float* src, float* weights, int width, int height, int nchannels, int j, int i);

#if FAST_FILTER
__global__ void CudaFilterImg(float* filteredImg, float* alpha, float* variances, int halfBlock, int width, int height);
#endif

__global__ void CudaComputeHaarWavelet(float* img, int width, int height, float* imgDWT, int blockSize);
__global__ void CudaCalcGradients(float* dst, int width, int height, int nchannels, int offset, float avgNorm, float stdNorm);
__global__ void CudaCalcMeanDeviation(float* dst, int width, int height, int nchannels, int blockDelta, int offset, float cmAvgNorm, float cmStdNorm);
__global__ void CudaCalcFeatureStatistics(float* dstMu, float* dstSigma, float* srcSigma, int width, int height, int nchannels, int blockDelta, int offset, int sigmaOffset, int stride, float muAvgNorm, float muStdNorm, float sigmaAvgNorm, float sigmaStdNorm);

__global__ void CudaCalcBlockMean(float* pixelColorMu, int width, int height, int blockDelta, float* mu);
__global__ void CudaCalcBlockStd(float* pixelColorMu, int width, int height, int blockDelta, float* mu, float* sigma);
__global__ void CudaReplaceSpikeWithMedian(float* pixelColorMu, float* spikeRemoved, int width, int height, int blockDelta, float spikeFactor, float* mu, float* sigma);

__global__ void CudaScaleVariance(float* dst, float* smpVar, float* smpVarFlt, float* bufVarFlt, int width, int height, int numOfChannels, int offset);
__global__ void CudaCalcBufVar(float* dst, float* buf0, float* buf1, int width, int height, int numOfChannels);
__global__ void CudaAddBiasApplyActivation(ActivationFunc devFunc, float* a, float* b, int width, int height, bool isLastLayer);


// Sigmoid inline funcs
__device__ float sigmoidFunc(float x) { return 1.0f / (1.0f + __expf(MAX(MIN(-x, MAX_EXP_VAL), MIN_EXP_VAL))); }
__device__ float sigmoidDerivFunc(float x) { return x * (1.0f - x); }

// Rectified linear inline funcs
__device__ float rectifiedLinearFunc(float x) { return __logf(1.0f + __expf(MAX(MIN(x, MAX_EXP_VAL), MIN_EXP_VAL))); }
__device__ float rectifiedLinearDerivFunc(float x) { return ((__expf(MAX(MIN(x, MAX_EXP_VAL), MIN_EXP_VAL))-1)/__expf(MAX(MIN(x, MAX_EXP_VAL), MIN_EXP_VAL))); } 


// Cuda doesn't support device function pointers because "__device__" functions are expanded inline by the compiler
__device__ ActivationFunc sigmoidFuncPtr = sigmoidFunc;
__device__ ActivationFunc sigmoidDerivFuncPtr = sigmoidDerivFunc;
__device__ ActivationFunc rectifiedLinearFuncPtr = rectifiedLinearFunc;
__device__ ActivationFunc rectifiedLinearDerivFuncPtr = rectifiedLinearDerivFunc;


ActivationFunc* FindActivationFunc(FUNC funcName) {

	ActivationFunc* currentFunc = NULL;
	switch(funcName) {

		case SIGMOID:
			currentFunc = &sigmoidFuncPtr;
			break;
		case SIGMOID_DERIV:
			currentFunc = &sigmoidDerivFuncPtr;
			break;
		case REC_LINEAR:
			currentFunc = &rectifiedLinearFuncPtr;
			break;
		case REC_LINEAR_DERIV:
			currentFunc = &rectifiedLinearDerivFuncPtr;
			break;
		default:
			fprintf(stderr, "ERROR: couldn't find matching activation function!\n");
	}

	assert(currentFunc != NULL);
	return currentFunc;

}

void AllocateGpuMemory(float *& dst, int size) {

	GpuErrorCheck(cudaMalloc(&(dst), size * sizeof(float))); 
	GpuErrorCheck(cudaMemset(dst, 0, size * sizeof(float)));

}

void AddBiasApplyActivation(FUNC func, Matrix<float>& A, Matrix<float>& B, bool isLastLayer) 
{
	// 65535 is the max num of blocks for cuda
	assert(A.getWidth() < 65535 * 1024);
	assert(A.getHeight() < 65535);
	assert(A.getIsCudaMat() == true && B.getIsCudaMat() == true);
	assert(A.getHeight() == B.getHeight());
	assert(A.getDepth() == 1 && B.getDepth() == 1);

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(1024, 1);

	int xSize = (A.getWidth() + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (A.getHeight() + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = (int) pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = (int) pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	ActivationFunc* hostFunc = FindActivationFunc(func);
	ActivationFunc devFunc;
	GpuErrorCheck(cudaMemcpyFromSymbol(&devFunc, *hostFunc, sizeof(ActivationFunc)));

	CudaAddBiasApplyActivation<<<numOfBlocks, numOfThreadsPerBlock>>>(devFunc, A.getElements(), B.getElements(), A.getWidth(), A.getHeight(), false /*isLastLayer*/);
	CUDA_ERROR_CHECK;

}

void FilterImage(float* filteredImg, float* alpha, float* variances, int halfBlock, int width, int height) 
{
	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = (int) pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = (int) pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	// set texture parameters
    textures.addressMode[0] = cudaAddressModeClamp;
    textures.addressMode[1] = cudaAddressModeClamp;
    textures.filterMode = cudaFilterModePoint;
    textures.normalized = false;

	float* totalWeights;
	float* currentWeight;
	float* tempWeights;
	AllocateGpuMemory(totalWeights, width * height);
	AllocateGpuMemory(currentWeight, width * height);
	AllocateGpuMemory(tempWeights, width * height);

    // Bind the array to the texture
	GpuErrorCheck(cudaBindTextureToArray(textures, devSampleArray, channelDesc));

	#if FAST_FILTER

		CudaFilterImg<<<numOfBlocks, numOfThreadsPerBlock>>>(filteredImg, alpha, variances, halfBlock, width, height);

	#else
		
		int imgSize = width * height;
		for(int dy = -halfBlock; dy <= halfBlock; dy++) 
		{
			for(int dx = -halfBlock; dx <= halfBlock; dx++) 
			{
				cudaMemset(currentWeight, 0, width * height * sizeof(float));
				cudaMemset(tempWeights, 0, width * height * sizeof(float));

				// Position term
				CudaDistance<<<numOfBlocks, numOfThreadsPerBlock>>>(currentWeight, alpha, NULL, width, height, 2, 0, dx, dy, POSTERM, NULL);
				CUDA_ERROR_CHECK;

				// Color term
				CudaDistance<<<numOfBlocks, numOfThreadsPerBlock>>>(currentWeight, NULL, variances, width, height, 3, 1, dx, dy, COLTERM, 0.01); // lambda = 1 / (2 * beta^2), beta equal to 7 (Sec. 4.2 and Appendix)
				CUDA_ERROR_CHECK;		

				for (int fInd = 0; fInd < 5; fInd++)
				{
					int varOffset = (fInd + 1) * 3 * imgSize;
					int alphaOffset = (fInd + 1) * imgSize;
					// Feature term
					CudaDistance<<<numOfBlocks, numOfThreadsPerBlock>>>(currentWeight, &alpha[alphaOffset], &variances[varOffset], width, height, textureSizes[fInd+2], fInd+2, dx, dy, FEATTERM, NULL);
					CUDA_ERROR_CHECK;
				}

				CudaGetWeights<<<numOfBlocks, numOfThreadsPerBlock>>>(tempWeights, currentWeight, width, height);
				CUDA_ERROR_CHECK;
				CudaApplyWeights<<<numOfBlocks, numOfThreadsPerBlock>>>(filteredImg, tempWeights, totalWeights, width, height, 3, dx, dy, 1);
				CUDA_ERROR_CHECK;

			}
		}
	
		CudaNormalize<<<numOfBlocks, numOfThreadsPerBlock>>>(filteredImg, totalWeights, width, height, 3);
		CUDA_ERROR_CHECK;

	#endif	
	
	GpuErrorCheck(cudaFree(totalWeights));
	GpuErrorCheck(cudaFree(currentWeight));
	GpuErrorCheck(cudaFree(tempWeights));
	GpuErrorCheck(cudaUnbindTexture(textures));
}

void FilterFeatures(float* filteredImg, float* variances, int halfBlock, int halfPatch, int indTex, float* filteredVar, int width, int height) 
{
	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = (int) pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = (int) pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	// set texture parameters
    textures.addressMode[0] = cudaAddressModeClamp;
    textures.addressMode[1] = cudaAddressModeClamp;
    textures.filterMode = cudaFilterModePoint;
    textures.normalized = false;

	float* totalWeights;
	float* currentWeight;
	float* tempWeights;
	AllocateGpuMemory(totalWeights, width * height);
	AllocateGpuMemory(currentWeight, width * height);
	AllocateGpuMemory(tempWeights, width * height);

    // Bind the array to the texture
	GpuErrorCheck(cudaBindTextureToArray(textures, devSampleArray, channelDesc));

	int boxDelta = 2;

	for(int dy = -halfBlock; dy <= halfBlock; dy++) 
	{
		for(int dx = -halfBlock; dx <= halfBlock; dx++) 
		{
			cudaMemset(currentWeight, 0, width * height * sizeof(float));
			cudaMemset(tempWeights, 0, width * height * sizeof(float));

			CudaDistance<<<numOfBlocks, numOfThreadsPerBlock>>>(currentWeight, NULL, variances, width, height, textureSizes[indTex], indTex, dx, dy, COLTERM, 1.0f/textureSizes[indTex]);
			CUDA_ERROR_CHECK;
			CudaBoxFilter<<<numOfBlocks, numOfThreadsPerBlock>>>(tempWeights, currentWeight, halfPatch, width, height, 1, 0, 0, 0);
			CUDA_ERROR_CHECK;			
			CudaGetWeights<<<numOfBlocks, numOfThreadsPerBlock>>>(currentWeight, tempWeights, width, height);
			CUDA_ERROR_CHECK;
			CudaBoxFilter<<<numOfBlocks, numOfThreadsPerBlock>>>(tempWeights, currentWeight, boxDelta, width, height, 1, dx, dy, 0);
			CUDA_ERROR_CHECK;
			CudaApplyWeights<<<numOfBlocks, numOfThreadsPerBlock>>>(filteredImg, tempWeights, totalWeights, width, height, textureSizes[indTex], dx, dy, indTex);
			CUDA_ERROR_CHECK;
			CudaApplyWeightsToVar<<<numOfBlocks, numOfThreadsPerBlock>>>(filteredVar, variances, tempWeights, width, height, textureSizes[indTex], dx, dy);
			CUDA_ERROR_CHECK;

		}
	}
	
	CudaNormalize<<<numOfBlocks, numOfThreadsPerBlock>>>(filteredImg, totalWeights, width, height, textureSizes[indTex]);
	CUDA_ERROR_CHECK;
	CudaNormalize<<<numOfBlocks, numOfThreadsPerBlock>>>(filteredVar, totalWeights, width, height, textureSizes[indTex]);
	CUDA_ERROR_CHECK;
	
	GpuErrorCheck(cudaFree(totalWeights));
	GpuErrorCheck(cudaFree(currentWeight));
	GpuErrorCheck(cudaFree(tempWeights));
	GpuErrorCheck(cudaUnbindTexture(textures));
}

void CalcMeanDeviation(float* featureData, float* normMu, float* normStd, int width, int height, int featureLength) {

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	dim3 numOfBlocks(xSize, ySize); 

	GpuErrorCheck(cudaBindTextureToArray(textures, devSampleArray, channelDesc));

	float* deviceMedianDeviation;
	int blockSize = 3;
	int blockDelta = (int) floor(blockSize / 2.0);

	AllocateGpuMemory(deviceMedianDeviation, width * height);

	for(int i = 0; i < BLOCK_LENGTH; i++) {

		int nchannels = textureSizes[i+2];
		GpuErrorCheck(cudaMemset(deviceMedianDeviation, 0, width * height * sizeof(float)));
		int curOffset = MD_OFFSET + i;
		assert(curOffset < featureLength && curOffset >= 0);

		CudaCalcMeanDeviation<<<numOfBlocks, numOfThreadsPerBlock>>>(deviceMedianDeviation, width, height, nchannels, blockDelta, i + 2, normMu[curOffset], normStd[curOffset]);
		CUDA_ERROR_CHECK;

		GpuErrorCheck(cudaMemcpy(&featureData[MD_OFFSET*width*height + i*width*height], deviceMedianDeviation, width * height * sizeof(float), cudaMemcpyDeviceToHost));

	}
	
	GpuErrorCheck(cudaFree(deviceMedianDeviation));
	GpuErrorCheck(cudaUnbindTexture(textures));
	
}

void CalcFeatureStatistics(float* featureData, float* pixelFeatureSigma, float* normMu, float* normStd, int width, int height, int featureLength) {

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	dim3 numOfBlocks(xSize, ySize); 

	GpuErrorCheck(cudaBindTextureToArray(textures, devSampleArray, channelDesc));

	float* deviceMu;
	float* deviceSigma;
	
	for(int j = 0; j < NUM_OF_BLOCKS; j++) {

		int blockSize = statBlockSizes[j];
		int blockDelta = (int) floor(blockSize / 2.0);
		AllocateGpuMemory(deviceMu, width * height);
		AllocateGpuMemory(deviceSigma, width * height);

		int offset = 0;
		for(int i = 0; i < BLOCK_LENGTH; i++) {
			int nchannels = textureSizes[i+2];

			GpuErrorCheck(cudaMemset(deviceMu, 0, width * height * sizeof(float)));
			GpuErrorCheck(cudaMemset(deviceSigma, 0, width * height * sizeof(float)));
			int curMuOffset = STAT_OFFSET + i + j*2*BLOCK_LENGTH;
			int curStdOffset = STAT_OFFSET + i + j*2*BLOCK_LENGTH + BLOCK_LENGTH;
			assert(curMuOffset < featureLength && curMuOffset >= 0);
			assert(curStdOffset < featureLength && curStdOffset >= 0);
			CudaCalcFeatureStatistics<<<numOfBlocks, numOfThreadsPerBlock>>>(deviceMu, deviceSigma, pixelFeatureSigma, width, height, nchannels, blockDelta, i + 2, offset, NUM_OF_FEATURES, normMu[curMuOffset], normStd[curMuOffset], normMu[curStdOffset], normStd[curStdOffset]);
			CUDA_ERROR_CHECK;

			GpuErrorCheck(cudaMemcpy(&featureData[STAT_OFFSET*width*height + i*width*height + j*width*height*2*BLOCK_LENGTH], deviceMu, width * height * sizeof(float), cudaMemcpyDeviceToHost));
			GpuErrorCheck(cudaMemcpy(&featureData[STAT_OFFSET*width*height + (i + BLOCK_LENGTH)*width*height + j*width*height*2*BLOCK_LENGTH], deviceSigma, width * height * sizeof(float), cudaMemcpyDeviceToHost));
			
			offset += nchannels;
		}

		GpuErrorCheck(cudaFree(deviceMu));
		GpuErrorCheck(cudaFree(deviceSigma));

	}

	GpuErrorCheck(cudaUnbindTexture(textures));
	
}

void CalcGradients(float* featureData, float* normMu, float* normStd, int width, int height, int featureLength) 
{

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = (int) pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = (int) pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	GpuErrorCheck(cudaBindTextureToArray(textures, devSampleArray, channelDesc));
	
	float* deviceGrad;
	AllocateGpuMemory(deviceGrad, width * height);

	for(int i = 0; i < NUM_OF_SIGMAS - 1; i++) {

		int nchannels = textureSizes[i+2];
		int curOffset = GRAD_OFFSET + i;

		CudaCalcGradients<<<numOfBlocks, numOfThreadsPerBlock>>>(deviceGrad, width, height, nchannels, i+2, normMu[curOffset], normStd[curOffset]);
		CUDA_ERROR_CHECK;

		GpuErrorCheck(cudaMemcpy(&featureData[GRAD_OFFSET*width*height + i*width*height], deviceGrad, width * height * sizeof(float), cudaMemcpyDeviceToHost));
			
	}

	GpuErrorCheck(cudaFree(deviceGrad));
	GpuErrorCheck(cudaUnbindTexture(textures));

}

void CalcBlockMean(float* pixelColorMu, int width, int height, int blockDelta, float* colorMu) 
{

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	dim3 numOfBlocks(xSize, ySize); 

	CudaCalcBlockMean<<<numOfBlocks, numOfThreadsPerBlock>>>(pixelColorMu, width, height, blockDelta, colorMu);
	CUDA_ERROR_CHECK;

}

void CalcBlockStd(float* pixelColorMu, int width, int height, int blockDelta, float* colorMu, float* colorSigma) 
{

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	dim3 numOfBlocks(xSize, ySize); 

	CudaCalcBlockStd<<<numOfBlocks, numOfThreadsPerBlock>>>(pixelColorMu, width, height, blockDelta, colorMu, colorSigma);
	CUDA_ERROR_CHECK;

}

void ReplaceSpikeWithMedian(float* pixelColorMu, int width, int height, int blockDelta, float spikeFactor, float* colorMu, float* colorSigma) 
{

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	dim3 numOfBlocks(xSize, ySize);

	float* originalPixelColorMu;
	AllocateGpuMemory(originalPixelColorMu, width * height * NUM_OF_COLORS);
	GpuErrorCheck(cudaMemcpy(originalPixelColorMu, pixelColorMu, width * height * NUM_OF_COLORS * sizeof(float), cudaMemcpyDeviceToDevice));

	CudaReplaceSpikeWithMedian<<<numOfBlocks, numOfThreadsPerBlock>>>(originalPixelColorMu, pixelColorMu, width, height, blockDelta, spikeFactor, colorMu, colorSigma);
	CUDA_ERROR_CHECK;
	
	GpuErrorCheck(cudaFree(originalPixelColorMu));
}

void ComputeHaarWavelet(float* img, int width, int height, float* imgDWT, int blockSize) 
{

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	dim3 numOfBlocks(xSize, ySize);

	CudaComputeHaarWavelet<<<numOfBlocks, numOfThreadsPerBlock>>>(img, width, height, imgDWT, blockSize);
	CUDA_ERROR_CHECK;

}

void BoxFilter(float* dst, float* data, int halfBlock, int width, int height, int numOfChannels, int offset) 
{

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = (int) pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = (int) pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);
	
	CudaBoxFilter<<<numOfBlocks, numOfThreadsPerBlock>>>(dst, data, halfBlock, width, height, numOfChannels, 0, 0, offset);
	CUDA_ERROR_CHECK;
	
}

void ScaleVar(float* dst, float* smpVar, float* smpVarFlt, float* bufVarFlt, int width, int height, int numOfChannels, int offset) 
{

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = (int) pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = (int) pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);
	
	CudaScaleVariance<<<numOfBlocks, numOfThreadsPerBlock>>>(dst, smpVar, smpVarFlt, bufVarFlt, width, height, numOfChannels, offset);
	CUDA_ERROR_CHECK;
	
}

void CalcBufVar(float* dst, float* buf0, float* buf1, int width, int height, int numOfChannels) 
{

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(128, 1);

	int xSize = (width + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (height + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = (int) pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = (int) pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);
	
	CudaCalcBufVar<<<numOfBlocks, numOfThreadsPerBlock>>>(dst, buf0, buf1, width, height, numOfChannels);
	CUDA_ERROR_CHECK;
	
}


#endif