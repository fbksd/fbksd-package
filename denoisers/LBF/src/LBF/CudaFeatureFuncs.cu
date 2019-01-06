#include "CudaWrappers.cuh"

__global__ void CudaDistance(float* distance, float* alpha, float* variances, int width, int height, int numChannels, int indTex, int dx, int dy, TERMTYPE type, float inputAlpha)
{

	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height)
	{
		
		if(indx + dx < 0 || indx + dx >= width || indy + dy < 0 || indy + dy >= height)
			return;

		int imgSize = width * height;
		int curPixelInd = indx + (indy * width);
		
		float curTexture[3];
		float sumOfDiffs = 0;

		int patchX = indx + dx;
		int patchY = indy + dy;
		int centerPatchX = indx;
		int centerPatchY = indy;

		float4 neighborTex = tex2DLayered(textures, patchX, patchY, indTex);
		float4 centerTex = tex2DLayered(textures, centerPatchX, centerPatchY, indTex);
						
		curTexture[0] = (neighborTex.x - centerTex.x);
		curTexture[1] = (neighborTex.y - centerTex.y);
		curTexture[2] = (neighborTex.z - centerTex.z);
								
		int centerInd = centerPatchX + (centerPatchY * width);
		int neighborInd = patchX + (patchY * width);
		for(int q = 0; q < numChannels; q++) 
		{
			float delta;
			delta = curTexture[q] * curTexture[q];

			if(type == FEATTERM)
			{
				// Eq. 12
				float centerVar = variances[centerInd + q * imgSize];
				delta /= MAX(DISTANCE_EPSILON, centerVar);
			}
			else if (type == COLTERM)
			{
				// Eq. 11
				float centerVar = variances[centerInd + q * imgSize];
				float neighborVar = variances[neighborInd + q * imgSize];
				delta /= (COLOR_DISTANCE_EPSILON + neighborVar + centerVar);
			}

			sumOfDiffs += delta;
		}
		
		float curAlpha;
		if (type == COLTERM)
			curAlpha = inputAlpha;
		else
			curAlpha = alpha[curPixelInd];

		distance[curPixelInd] += sumOfDiffs * curAlpha;
	}
	
}


__global__ void CudaCalcGradients(float* dst, int width, int height, int nchannels, int offset, float avgNorm, float stdNorm)
{

	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height)
	{

		int curPixelInd = indx + (indy * width);
		int delta = 1;
		float xGrad[3] = {0, 0, 0};
		float yGrad[3] = {0, 0, 0};
		int count = 0;
		float weights[3] = {-1.0f, 0, 1.0f};

		for (int i = -delta; i <= delta; i++) {
			for (int j = -delta; j <= delta; j++) {

				int xIndex = MIN(indx + j, width - 1);
				xIndex = MAX(xIndex, 0);
				int yIndex = MIN(indy + i, height - 1);
				yIndex = MAX(yIndex, 0);

				float4 curData = tex2DLayered(textures, xIndex, yIndex, offset);

				float xWeight = weights[j+1];
				float yWeight = weights[i+1];
				for(int q = 0; q < nchannels; q++) {
					float curVal = 0;
					if(q == 0) {
						curVal = curData.x;
					} else if(q == 1) {
						curVal = curData.y;
					} else if(q == 2) {
						curVal = curData.z;
					} else {
						printf("Error in Grad: Too many channels being accessed!\n");
					}
					float xTmp = xWeight * curVal;
					float yTmp = yWeight * curVal;
					xGrad[q] += xTmp;
					yGrad[q] += yTmp;
				}
	
				count++;

			}
		}
		
		if(count != 0) 
		{
			float avgGrad = 0;
			for(int q = 0; q < nchannels; q++) {
				xGrad[q] /= float(count);
				yGrad[q] /= float(count);
				avgGrad += (xGrad[q]*xGrad[q] + yGrad[q]*yGrad[q]) / 2.0f;
			}
			avgGrad /= nchannels;
			dst[curPixelInd] = (avgGrad - avgNorm) / (stdNorm + NORM_EPS);
		} 
		else 
		{
			printf("ERROR: Grad count zero\n");
		}

	}
}

__global__ void CudaApplyWeights(float* dst, float* weights, int width, int height, int numOfChannels, int j, int i, int filterOffset, int offset)
{

	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height && (indx + j) >= 0 && (indx + j) < width && (indy + i) >= 0 && (indy + i) < height)
	{

		int imgSize = width * height;
		int curPixelInd = indx + (indy * width);

		float4 origSampleColor = tex2DLayered(textures, indx+j, indy+i, 1 + filterOffset);

		dst[curPixelInd + offset*imgSize] += weights[curPixelInd] * origSampleColor.x;
		if(numOfChannels > 1) {
			dst[curPixelInd + (offset + 1)*imgSize] += weights[curPixelInd] * origSampleColor.y;
		}
		if(numOfChannels > 2) {
			dst[curPixelInd + (offset + 2)*imgSize] += weights[curPixelInd] * origSampleColor.z;
		}

	}
}

__global__ void CudaApplyWeights(float* dst, float* weights, float* totalWeight, int width, int height, int numOfChannels, int j, int i, int filterOffset)
{

	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height && (indx + j) >= 0 && (indx + j) < width && (indy + i) >= 0 && (indy + i) < height)
	{

		int imgSize = width * height;
		int curPixelInd = indx + (indy * width);

		float4 origSampleColor = tex2DLayered(textures, indx+j, indy+i, filterOffset);

		totalWeight[curPixelInd] += weights[curPixelInd];

		dst[curPixelInd] += weights[curPixelInd] * origSampleColor.x;
		if(numOfChannels > 1) {
			dst[curPixelInd + 1 * imgSize] += weights[curPixelInd] * origSampleColor.y;
			dst[curPixelInd + 2 * imgSize] += weights[curPixelInd] * origSampleColor.z;
		}

	}
}

__global__ void CudaApplyWeightsToVar(float* dst, float* src, float* weights, int width, int height, int nchannels, int j, int i)
{

	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height && (indx + j) >= 0 && (indx + j) < width && (indy + i) >= 0 && (indy + i) < height)
	{

		int imgSize = width * height;
		int curPixelInd = indx + (indy * width);
		int neighborPixelInd = indx + j + ((indy + i) * width);

		for(int q = 0; q < nchannels; q++) {
			float curVal = weights[curPixelInd] * src[neighborPixelInd + q * imgSize];
			dst[curPixelInd + q * imgSize] += curVal; 
		}

	}
}


__global__ void CudaGetWeights(float* dst, float* src, int width, int height)
{

	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height)
	{

		int curPixelInd = indx + (indy * width);
		dst[curPixelInd] = exp(-MAX(0, src[curPixelInd]));

	}
}

__global__ void CudaNormalize(float* filteredData, float* weights, int width, int height, int numOfChannels)
{

	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height)
	{

		int imgSize = width * height;
		int curPixelInd = indx + (indy * width);
		
		if(weights[curPixelInd] != 0) {
			float normFactor = 1.0f / weights[curPixelInd];
			for(int q = 0; q < numOfChannels; q++) {
				filteredData[curPixelInd + q * imgSize] *= normFactor;
			}
		} else {
			printf("Error weights are zero\n");
		}

	}
}

__global__ void CudaCalcMeanDeviation(float* dst, int width, int height, int nchannels, int blockDelta, int offset, float cmAvgNorm, float cmStdNorm) {

	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int index = yIndex * width + xIndex;
	if(yIndex >= 0 && yIndex < height && xIndex >= 0 && xIndex < width) {

		int lowerX = MAX(xIndex - blockDelta, 0);
		int upperX = MIN(xIndex + blockDelta, width - 1);
		int lowerY = MAX(yIndex - blockDelta, 0);
		int upperY = MIN(yIndex + blockDelta, height - 1);
	
		int count = 0;

		float mu[3] = {0, 0, 0};

		for(int r = lowerY; r <= upperY; r++) {
			for(int s = lowerX; s <= upperX; s++) {

				float4 curData = tex2DLayered(textures, s, r, offset);
				for(int q = 0; q < nchannels; q++) {
					float curVal = 0;
					if(q == 0) {
						curVal = curData.x;
					} else if(q == 1) {
						curVal = curData.y;
					} else if(q == 2) {
						curVal = curData.z;
					} else {
						printf("Error in CM: Too many channels being accessed!\n");
					}
					mu[q] += curVal;
				}
			
				count++;
				
			}
		} 

		float meanScalar = 1.0f / count;
		for(int q = 0; q < nchannels; q++) {
			mu[q] *= meanScalar;
		}

		float cmTmp[3] = {0, 0, 0};
		for(int r = lowerY; r <= upperY; r++) {
			for(int s = lowerX; s <= upperX; s++) {
				float4 curData = tex2DLayered(textures, s, r, offset);
				for(int q = 0; q < nchannels; q++) {
					float curVal = 0;
					if(q == 0) {
						curVal = curData.x;
					} else if(q == 1) {
						curVal = curData.y;
					} else if(q == 2) {
						curVal = curData.z;
					} else {
						printf("Error in CM: Too many channels being accessed!\n");
					}
					cmTmp[q] += fabs(curVal - mu[q]);
				}
			}
		} 

		float cmAvg = 0;
		for(int q = 0; q < nchannels; q++) {
			cmAvg += meanScalar * cmTmp[q];
		}
		cmAvg /= nchannels;
		dst[index] = (cmAvg - cmAvgNorm) / (cmStdNorm + NORM_EPS);
	}
}

__global__ void CudaCalcFeatureStatistics(float* dstMu, float* dstSigma, float* srcSigma, int width, int height, int nchannels, int blockDelta, int offset, int sigmaOffset, int stride, float muAvgNorm, float muStdNorm, float sigmaAvgNorm, float sigmaStdNorm) {

	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int index = yIndex * width + xIndex;
	if(yIndex >= 0 && yIndex < height && xIndex >= 0 && xIndex < width) {

		if(blockDelta != 0) {

			int lowerX = MAX(xIndex - blockDelta, 0);
			int upperX = MIN(xIndex + blockDelta, width - 1);
			int lowerY = MAX(yIndex - blockDelta, 0);
			int upperY = MIN(yIndex + blockDelta, height - 1);
	
			int count = 0;
			float mu[3] = {0, 0, 0};

			for(int r = lowerY; r <= upperY; r++) {
				for(int s = lowerX; s <= upperX; s++) {

					float4 curData = tex2DLayered(textures, s, r, offset);
					for(int q = 0; q < nchannels; q++) {
						float curVal = 0;
						if(q == 0) {
							curVal = curData.x;
						} else if(q == 1) {
							curVal = curData.y;
						} else if(q == 2) {
							curVal = curData.z;
						} else {
							printf("Error in BlockMuStd: Too many channels being accessed!\n");
						}
						mu[q] += curVal;
					}
			
					count++;
				
				}
			} 

			float meanScalar = 1.0f / float(count);
			for(int q = 0; q < nchannels; q++) {
				mu[q] *= meanScalar;
			}

			float sigma[3] = {0, 0, 0};
			for(int r = lowerY; r <= upperY; r++) {
				for(int s = lowerX; s <= upperX; s++) {

					float4 curData = tex2DLayered(textures, s, r, offset);
					for(int q = 0; q < nchannels; q++) {
						float curVal = 0;
						if(q == 0) {
							curVal = curData.x;
						} else if(q == 1) {
							curVal = curData.y;
						} else if(q == 2) {
							curVal = curData.z;
						} else {
							printf("Error in BlockMuStd: Too many channels being accessed!\n");
						}
						float temp = curVal - mu[q];
						sigma[q] += temp * temp;
					}
				}
			} 

			float avgMu = 0;
			float avgSigma = 0;
			if(count < 2) {
				printf("Error: Less than 2 pixels in neighborhood\n");
			} 
			float sigmaScalar = 1.0f / float(count - 1);
			for(int q = 0; q < nchannels; q++) {
				avgMu += mu[q];
				sigma[q] *= sigmaScalar;
				sigma[q] = sqrtf(sigma[q]);
				avgSigma += sigma[q];
			}
			avgMu /= nchannels;
			avgSigma /= nchannels;
			dstMu[index] = (avgMu - muAvgNorm) / (muStdNorm + NORM_EPS);
			dstSigma[index] = (avgSigma - sigmaAvgNorm) / (sigmaStdNorm + NORM_EPS);

		} else {

			float avgMu = 0;
			float avgSigma = 0;
			float4 curData = tex2DLayered(textures, xIndex, yIndex, offset);
			for(int q = 0; q < nchannels; q++) {
				float curVal = 0;
				if(q == 0) {
					curVal = curData.x;
				} else if(q == 1) {
					curVal = curData.y;
				} else if(q == 2) {
					curVal = curData.z;
				} else {
					printf("Error: Too many channels being accessed!\n");
				}
				avgMu += curVal;
				avgSigma += sqrtf(srcSigma[index + (q + sigmaOffset) * width * height]);
			}

			avgMu /= nchannels;
			avgSigma /= nchannels;
			dstMu[index] = (avgMu - muAvgNorm) / (muStdNorm + NORM_EPS);
			dstSigma[index] = (avgSigma - sigmaAvgNorm) / (sigmaStdNorm + NORM_EPS);

		}
	}
}

__global__ void CudaComputeHaarWavelet(float* img, int width, int height, float* estNoise, int blockSize) 
{
	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	
	if(yIndex < height && yIndex >= 0 && xIndex < width && xIndex >= 0) 
	{
		int curPixelInd = yIndex * width + xIndex;
		int ymin = yIndex - blockSize/2;
		int ymax = yIndex + blockSize/2;
		int xmin = xIndex - blockSize/2;
		int xmax = xIndex + blockSize/2;

		float imgDWT[16];
		memset(imgDWT, 0, 16 * sizeof(float));

		int i = 0;
		int curX, curY;
		for (int indy = ymin; indy < ymax; indy+=2)
		{
			for (int indx = xmin; indx < xmax; indx+=2)
			{
				curX = abs(indx);
				curX = (curX >= width) ? 2 * (width-1) - curX : curX;
				curY = abs(indy);
				curY = (curY >= height) ? 2 * (height-1) - curY : curY;
				float a = img[curY * width + curX];
				
				curX = abs(indx+1);
				curX = (curX >= width) ? 2 * (width-1) - curX : curX;
				curY = abs(indy);
				curY = (curY >= height) ? 2 * (height-1) - curY : curY;
				float b = img[curY * width + curX];

				curX = abs(indx);
				curX = (curX >= width) ? 2 * (width-1) - curX : curX;
				curY = abs(indy+1);
				curY = (curY >= height) ? 2 * (height-1) - curY : curY;
				float c = img[curY * width + curX];

				curX = abs(indx+1);
				curX = (curX >= width) ? 2 * (width-1) - curX : curX;
				curY = abs(indy+1);
				curY = (curY >= height) ? 2 * (height-1) - curY : curY;
				float d = img[curY * width + curX];

				imgDWT[i] = abs((b+c-a-d) * 0.5f);
				i++;
			}
		}


		// MEDIAN of Details weighted by 1/0.6745

		float factor = 1 / 0.6745;
		int numPixels = 16;
		// bubble-sort
		for (int i = 0; i < numPixels; i++) 
		{
			for (int j = i + 1; j < numPixels; j++) 
			{
				if (imgDWT[i] > imgDWT[j]) /* swap? */
				{ 
					float tmp = imgDWT[i];
					imgDWT[i] = imgDWT[j];
					imgDWT[j] = tmp;
				}
			}
		}

		estNoise[curPixelInd] = (abs(imgDWT[7])+abs(imgDWT[8]))/2.0f * factor;


	}
}



__global__ void CudaScaleVariance(float* dst, float* smpVar, float* smpVarFlt, float* bufVarFlt, int width, int height, int numOfChannels, int offset)
{

	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height)
	{
		int curPixelInd = indx + indy*width;
		int imgSize = width * height;
		for(int q = 0; q < numOfChannels; q++) {
			float var = smpVar[curPixelInd + (offset + q)*imgSize];
			float ratio = (bufVarFlt[curPixelInd + q*imgSize] / (smpVarFlt[curPixelInd + q*imgSize] + 1.0e-10f));
			dst[curPixelInd + q*imgSize + offset*imgSize] = var * ratio;

		}
	}

}



__global__ void CudaCalcBufVar(float* dst, float* buf0, float* buf1, int width, int height, int numOfChannels) 
{
	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height)
	{
		int curPixelInd = indx + indy*width;
		int imgSize = width * height;
		for(int q = 0; q < numOfChannels; q++) {
			float buf0Val = buf0[curPixelInd + q*imgSize];
			float buf1Val = buf1[curPixelInd + q*imgSize];
			float delta = (buf1Val - buf0Val);
			float meanVal = buf0Val + (delta / 2.0f);
			dst[curPixelInd + q*imgSize] = (delta * (buf1Val - meanVal))/2.0f; // Division by 2 is normalization
		}
	}

}

__global__ void CudaAddBiasApplyActivation(ActivationFunc devFunc, float* a, float* b, int width, int height, bool isLastLayer) 
{
	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int index = yIndex * width + xIndex;
	if(yIndex < height && xIndex < width) {
		if(isLastLayer) {
			if(yIndex == 0) {
				a[index] = a[index] + b[yIndex];
			} else {
				a[index] = (*devFunc)(a[index] + b[yIndex]);
			}
		} else {
			a[index] = (*devFunc)(a[index] + b[yIndex]);
		}
	}
}

#ifdef FAST_FILTER
	__global__ void CudaFilterImg(float* filteredImg, float* alpha, float* variances, int halfBlock, int width, int height)
	{
		int indx = threadIdx.x + (blockIdx.x * blockDim.x);
		int indy = threadIdx.y + (blockIdx.y * blockDim.y);

		if (indx >= 0 && indx < width && indy >= 0 && indy < height)
		{

			float4 curTexture;
			int imgSize = width * height;
			int centerPixelInd = indx + (indy * width);

			/////////////////////// LOADING THE MEAN PRIMARY FEATURES AND VARIANCES ///////////////////////
			float centerSample[MAX_FILTER_SAMPLE_LENGTH];
			float centerVar[MAX_FILTER_SAMPLE_LENGTH - 2];

			/******************** LOADING MEAN PRIMARY FEATURES ****************/
			// Position
			curTexture = tex2DLayered(textures, indx, indy, 0);
			centerSample[X_COORD] = curTexture.x;
			centerSample[Y_COORD] = curTexture.y;
			// Color
			curTexture = tex2DLayered(textures, indx, indy, 1);
			centerSample[COLOR_1] = curTexture.x;
			centerSample[COLOR_2] = curTexture.y;
			centerSample[COLOR_3] = curTexture.z;
			// World
			curTexture = tex2DLayered(textures, indx, indy, 2);
			centerSample[WORLD_1_X] = curTexture.x;
			centerSample[WORLD_1_Y] = curTexture.y;
			centerSample[WORLD_1_Z] = curTexture.z;
			// Normal
			curTexture = tex2DLayered(textures, indx, indy, 3);
			centerSample[NORM_1_X] = curTexture.x;
			centerSample[NORM_1_Y] = curTexture.y;
			centerSample[NORM_1_Z] = curTexture.z;
			// Texture 1
			curTexture = tex2DLayered(textures, indx, indy, 4);
			centerSample[TEXTURE_1_X] = curTexture.x;
			centerSample[TEXTURE_1_Y] = curTexture.y;
			centerSample[TEXTURE_1_Z] = curTexture.z;
			// Texture 2
			curTexture = tex2DLayered(textures, indx, indy, 5);
			centerSample[TEXTURE_2_X] = curTexture.x;
			centerSample[TEXTURE_2_Y] = curTexture.y;
			centerSample[TEXTURE_2_Z] = curTexture.z;
			// Visibility
			curTexture = tex2DLayered(textures, indx, indy, 6);
			centerSample[VISIBILITY_1] = curTexture.x;

			/*****************************************************************/

			/******************** LOADING VARIANCES **************************/
			// Color
			centerVar[COLOR_1 - COLOR_1] = variances[centerPixelInd + (COLOR_1 - COLOR_1) * imgSize];
			centerVar[COLOR_2 - COLOR_1] = variances[centerPixelInd + (COLOR_2 - COLOR_1) * imgSize];
			centerVar[COLOR_3 - COLOR_1] = variances[centerPixelInd + (COLOR_3 - COLOR_1) * imgSize];
			// World
			centerVar[WORLD_1_X - COLOR_1] = variances[centerPixelInd + (WORLD_1_X - COLOR_1) * imgSize];
			centerVar[WORLD_1_Y - COLOR_1] = variances[centerPixelInd + (WORLD_1_Y - COLOR_1) * imgSize];
			centerVar[WORLD_1_Z - COLOR_1] = variances[centerPixelInd + (WORLD_1_Z - COLOR_1) * imgSize];
			// Normal
			centerVar[NORM_1_X - COLOR_1] = variances[centerPixelInd + (NORM_1_X - COLOR_1) * imgSize];
			centerVar[NORM_1_Y - COLOR_1] = variances[centerPixelInd + (NORM_1_Y - COLOR_1) * imgSize];
			centerVar[NORM_1_Z - COLOR_1] = variances[centerPixelInd + (NORM_1_Z - COLOR_1) * imgSize];
			// Texture 1
			centerVar[TEXTURE_1_X - COLOR_1] = variances[centerPixelInd + (TEXTURE_1_X - COLOR_1) * imgSize];
			centerVar[TEXTURE_1_Y - COLOR_1] = variances[centerPixelInd + (TEXTURE_1_Y - COLOR_1) * imgSize];
			centerVar[TEXTURE_1_Z - COLOR_1] = variances[centerPixelInd + (TEXTURE_1_Z - COLOR_1) * imgSize];
			// Texture 2
			centerVar[TEXTURE_2_X - COLOR_1] = variances[centerPixelInd + (TEXTURE_2_X - COLOR_1) * imgSize];
			centerVar[TEXTURE_2_Y - COLOR_1] = variances[centerPixelInd + (TEXTURE_2_Y - COLOR_1) * imgSize];
			centerVar[TEXTURE_2_Z - COLOR_1] = variances[centerPixelInd + (TEXTURE_2_Z - COLOR_1) * imgSize];				
			// Visibility
			centerVar[VISIBILITY_1 - COLOR_1] = variances[centerPixelInd + (VISIBILITY_1 - COLOR_1) * imgSize];

			/*****************************************************************/

			//////////////////////// END OF LOADING THE MEAN PRIMARY FEATURES AND VARIANCES //////////////////////


			/////////////////////// MAIN FILTERING LOOP ///////////////////////
			float sumOfDiffs;
			float curWeight;
			float totalWeight = 0;
			float accumColor[3] = {0.0f, 0.0f, 0.0f};
			float4 origSampleColor;
			int neighborFullPixelInd = 0;

			float neighborVarX = 0;
			float neighborVarY = 0;
			float neighborVarZ = 0;

			for (int i = -halfBlock; i <= halfBlock; i++)
			{
				for (int j = -halfBlock; j <= halfBlock; j++)
				{

					if(indx + j < 0 || indx + j >= width || indy + i < 0 || indy + i >= height)
						continue;

					sumOfDiffs = 0;
					neighborFullPixelInd = (indx + j) + ((indy + i) * width);
					origSampleColor = tex2DLayered(textures, indx+j, indy+i, 1);

					// Position
					curTexture = tex2DLayered(textures, indx+j, indy+i, 0);
					curTexture.x = (curTexture.x - centerSample[X_COORD]);
					curTexture.y = (curTexture.y - centerSample[Y_COORD]);
					sumOfDiffs  -= (curTexture.x * curTexture.x + curTexture.y * curTexture.y) * alpha[centerPixelInd];

					// Color
					curTexture = origSampleColor;
					curTexture.x = (curTexture.x - centerSample[COLOR_1]);
					curTexture.y = (curTexture.y - centerSample[COLOR_2]);
					curTexture.z = (curTexture.z - centerSample[COLOR_3]);
						
					curTexture.x *= curTexture.x;
					curTexture.y *= curTexture.y;
					curTexture.z *= curTexture.z;

					neighborVarX = variances[neighborFullPixelInd + (COLOR_1 - COLOR_1) * imgSize];
					neighborVarY = variances[neighborFullPixelInd + (COLOR_2 - COLOR_1) * imgSize];
					neighborVarZ = variances[neighborFullPixelInd + (COLOR_3 - COLOR_1) * imgSize];

					curTexture.x /= COLOR_DISTANCE_EPSILON + neighborVarX + centerVar[COLOR_1 - COLOR_1];
					curTexture.y /= COLOR_DISTANCE_EPSILON + neighborVarY + centerVar[COLOR_2 - COLOR_1];
					curTexture.z /= COLOR_DISTANCE_EPSILON + neighborVarZ + centerVar[COLOR_3 - COLOR_1];

					sumOfDiffs  -= (curTexture.x + curTexture.y + curTexture.z) * 0.01; // lambda = 1 / (2 * beta^2), beta equal to 7 (Sec. 4.2 and Appendix)
					
					// World
					curTexture = tex2DLayered(textures, indx+j, indy+i, 2);
					curTexture.x = (curTexture.x - centerSample[WORLD_1_X]);
					curTexture.y = (curTexture.y - centerSample[WORLD_1_Y]);
					curTexture.z = (curTexture.z - centerSample[WORLD_1_Z]);

					curTexture.x *= curTexture.x;
					curTexture.y *= curTexture.y;
					curTexture.z *= curTexture.z;

					curTexture.x /= MAX(DISTANCE_EPSILON, centerVar[WORLD_1_X - COLOR_1]);
					curTexture.y /= MAX(DISTANCE_EPSILON, centerVar[WORLD_1_Y - COLOR_1]);
					curTexture.z /= MAX(DISTANCE_EPSILON, centerVar[WORLD_1_Z - COLOR_1]);

					sumOfDiffs  -= (curTexture.x + curTexture.y + curTexture.z) * alpha[centerPixelInd + 1 * imgSize];

					// Normal
					curTexture = tex2DLayered(textures, indx+j, indy+i, 3);
					curTexture.x = (curTexture.x - centerSample[NORM_1_X]);
					curTexture.y = (curTexture.y - centerSample[NORM_1_Y]);
					curTexture.z = (curTexture.z - centerSample[NORM_1_Z]);
						
					curTexture.x *= curTexture.x;
					curTexture.y *= curTexture.y;
					curTexture.z *= curTexture.z;

					curTexture.x /= MAX(DISTANCE_EPSILON, centerVar[NORM_1_X - COLOR_1]);
					curTexture.y /= MAX(DISTANCE_EPSILON, centerVar[NORM_1_Y - COLOR_1]);
					curTexture.z /= MAX(DISTANCE_EPSILON, centerVar[NORM_1_Z - COLOR_1]);
	
					sumOfDiffs  -= (curTexture.x + curTexture.y + curTexture.z) * alpha[centerPixelInd + 2 * imgSize];

					// Texture 1
					curTexture = tex2DLayered(textures, indx+j, indy+i, 4);
					curTexture.x = (curTexture.x - centerSample[TEXTURE_1_X]);
					curTexture.y = (curTexture.y - centerSample[TEXTURE_1_Y]);
					curTexture.z = (curTexture.z - centerSample[TEXTURE_1_Z]);
						
					curTexture.x *= curTexture.x;
					curTexture.y *= curTexture.y;
					curTexture.z *= curTexture.z;

					curTexture.x /= (MAX(DISTANCE_EPSILON, centerVar[TEXTURE_1_X - COLOR_1]));
					curTexture.y /= (MAX(DISTANCE_EPSILON, centerVar[TEXTURE_1_Y - COLOR_1]));
					curTexture.z /= (MAX(DISTANCE_EPSILON, centerVar[TEXTURE_1_Z - COLOR_1]));

					sumOfDiffs  -= (curTexture.x + curTexture.y + curTexture.z) * alpha[centerPixelInd + 3 * imgSize];


					// Texture 2
					curTexture = tex2DLayered(textures, indx+j, indy+i, 5);
					curTexture.x = (curTexture.x - centerSample[TEXTURE_2_X]);
					curTexture.y = (curTexture.y - centerSample[TEXTURE_2_Y]);
					curTexture.z = (curTexture.z - centerSample[TEXTURE_2_Z]);
						
					curTexture.x *= curTexture.x;
					curTexture.y *= curTexture.y;
					curTexture.z *= curTexture.z;

					curTexture.x /= MAX(DISTANCE_EPSILON, centerVar[TEXTURE_2_X - COLOR_1]);
					curTexture.y /= MAX(DISTANCE_EPSILON, centerVar[TEXTURE_2_Y - COLOR_1]);
					curTexture.z /= MAX(DISTANCE_EPSILON, centerVar[TEXTURE_2_Z - COLOR_1]);

					sumOfDiffs  -= (curTexture.x + curTexture.y + curTexture.z) * alpha[centerPixelInd + 4 * imgSize];

					// Visibility
					curTexture = tex2DLayered(textures, indx+j, indy+i, 6);
					curTexture.x = (curTexture.x - centerSample[VISIBILITY_1]);
						
					curTexture.x *= curTexture.x;

					curTexture.x /= MAX(DISTANCE_EPSILON, centerVar[VISIBILITY_1 - COLOR_1]);
						
					sumOfDiffs  -= curTexture.x * alpha[centerPixelInd + 5 * imgSize];		


					curWeight = __expf(sumOfDiffs);
					totalWeight += curWeight;
					accumColor[0] += curWeight * origSampleColor.x;
					accumColor[1] += curWeight * origSampleColor.y;
					accumColor[2] += curWeight * origSampleColor.z;
				}
			}

			if(totalWeight != 0) 
			{
				float invTotalWeight = 1.0f / totalWeight;
				filteredImg[centerPixelInd] = accumColor[0] * invTotalWeight;
				filteredImg[centerPixelInd + imgSize] = accumColor[1] * invTotalWeight;
				filteredImg[centerPixelInd + 2 * imgSize] = accumColor[2] * invTotalWeight;
			} 
			else
				printf("ERROR: Filter normalization weight is zero!\n");

			////////////////////////////// END OF MAIN FILTERING LOOP ////////////////////////////////////

		}
	}
#endif

__global__ void CudaCalcBlockMean(float* pixelColorMu, int width, int height, int blockDelta, float* mu) 
{
	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	
	if(yIndex < height && yIndex >= 0 && xIndex < width && xIndex >= 0) {

		int lowerX = MAX(xIndex - blockDelta, 0);
		int upperX = MIN(xIndex + blockDelta, width - 1);
		int lowerY = MAX(yIndex - blockDelta, 0);
		int upperY = MIN(yIndex + blockDelta, height - 1);
	
		int count = 0;

		int currentIndex = yIndex * width + xIndex;
		int delta = width * height;
	
		for(int r = lowerY; r <= upperY; r++) {
			for(int s = lowerX; s <= upperX; s++) {

				if(r == yIndex && s == xIndex) {
					continue;
				}

				int index = r * width + s;
				for(int q = 0; q < NUM_OF_COLORS; q++) {
					float currentColorMu = pixelColorMu[index + q * delta];
					mu[currentIndex + q * delta] += currentColorMu;
				}
			
				count++;
				
			}
		} 

		float meanScalar = 1.0f / count;
		for(int q = 0; q < NUM_OF_COLORS; q++) {
			mu[currentIndex + q * delta] *= meanScalar;
		}
	}
}


__global__ void CudaCalcBlockStd(float* pixelColorMu, int width, int height, int blockDelta, float* mu, float* sigma) 
{
	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	
	if(yIndex < height && yIndex >= 0 && xIndex < width && xIndex >= 0) {

		int lowerX = MAX(xIndex - blockDelta, 0);
		int upperX = MIN(xIndex + blockDelta, width - 1);
		int lowerY = MAX(yIndex - blockDelta, 0);
		int upperY = MIN(yIndex + blockDelta, height - 1);
	
		int count = 0;

		int currentIndex = yIndex * width + xIndex;
		int delta = width * height;
		
		for(int r = lowerY; r <= upperY; r++) {
			for(int s = lowerX; s <= upperX; s++) {

				if(r == yIndex && s == xIndex) {
					continue;
				}

				int index = r * width + s;
				for(int q = 0; q < NUM_OF_COLORS; q++) {
					float currentColorMu = pixelColorMu[index + q * delta];
					float temp = currentColorMu - mu[currentIndex + q * delta];
					sigma[currentIndex + q * delta] += temp * temp;
				}

				count++;
			
			}
		} 

		if(count > 1) {

			float sigmaScalar = 1.0f / (count - 1);
			for(int q = 0; q < NUM_OF_COLORS; q++) {
				sigma[currentIndex + q * delta] *= sigmaScalar;
				sigma[currentIndex + q * delta] = sqrtf(sigma[currentIndex + q * delta]);
			}

		} else {

			printf("Error: Too few samples to calculate sigma!\n");
		
		}

	}
}

__global__ void CudaReplaceSpikeWithMedian(float* pixelColorMu, float* spikeRemoved, int width, int height, int blockDelta, float spikeFactor, float* mu, float* sigma) 
{
	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	
	if(yIndex < height && yIndex >= 0 && xIndex < width && xIndex >= 0) 
	{
		
		int currentIndex = yIndex * width + xIndex;
		int delta = width * height;

		if (abs(pixelColorMu[currentIndex + 0 * delta] - mu[currentIndex + 0 * delta]) > spikeFactor * sigma[currentIndex + 0 * delta] ||
			abs(pixelColorMu[currentIndex + 1 * delta] - mu[currentIndex + 1 * delta]) > spikeFactor * sigma[currentIndex + 1 * delta] ||
			abs(pixelColorMu[currentIndex + 2 * delta] - mu[currentIndex + 2 * delta]) > spikeFactor * sigma[currentIndex + 2 * delta])
		{
			float* v = new float[(2 * blockDelta + 1) * (2 * blockDelta + 1)];
			for(int q = 0; q < NUM_OF_COLORS; q++) 
			{
				
				memset(v, 0, (2 * blockDelta + 1) * (2 * blockDelta + 1) * sizeof(float));
				int ind = 0;
				for (int dx = - blockDelta; dx <= blockDelta; dx++) 
				{
					for (int dy = - blockDelta; dy <= blockDelta; dy++) 
					{
						int curX = xIndex + dx;
						int curY = yIndex + dy;

						if (curX >= 0 && curX < width && curY >= 0 && curY < height) // boundaries
						{
							int index = curY * width + curX;
							v[ind++] = pixelColorMu[index + q * delta];
						}
					}
				}

				int numPixels = (2 * blockDelta + 1) * (2 * blockDelta + 1);
				// bubble-sort
				for (int i = 0; i < numPixels; i++) 
				{
					for (int j = i + 1; j < numPixels; j++) 
					{
						if (v[i] > v[j]) /* swap? */
						{ 
							float tmp = v[i];
							v[i] = v[j];
							v[j] = tmp;
						}
					}
				}

				// pick the middle one
				spikeRemoved[currentIndex + q * delta] = v[int(floorf(numPixels / 2.0f))];
			}
			delete[] v;
		}
	}
}

__global__ void CudaBoxFilter(float* dst, float* src, int halfBlock, int width, int height, int numOfChannels, int xOffset, int yOffset, int srcOffset)
{

	int indx = threadIdx.x + (blockIdx.x * blockDim.x);
	int indy = threadIdx.y + (blockIdx.y * blockDim.y);

	if(indx >= 0 && indx < width && indy >= 0 && indy < height && indx + xOffset >= 0 && indx + xOffset < width && indy + yOffset >= 0 && indy + yOffset < height)
	{

		int imgSize = width * height;
		int curPixelInd = indx + (indy * width);

		int totalWeight = 0;
		float accumData[3] = {0.0f, 0.0f, 0.0f};
		
		for (int i = -halfBlock; i <= halfBlock; i++)
		{
			for (int j = -halfBlock; j <= halfBlock; j++)
			{

				if(indx + j < 0 || indx + j >= width || indy + i < 0 || indy + i >= height) {
					continue;
				}

				if(indx + j + xOffset < 0 || indx + j + xOffset >= width || indy + i + yOffset < 0 || indy + i + yOffset >= height) {
					continue;
				}
					
				int neighborIndex = (indx + j) + (indy + i)*width;
				for(int q = 0; q < numOfChannels; q++) {
					accumData[q] += src[neighborIndex + (srcOffset + q)*imgSize];
				}
				totalWeight++;
			}
		}
	
		if(totalWeight != 0) 
		{
			
			float invTotalWeight = 1.0f / float(totalWeight);
			for(int q = 0; q < numOfChannels; q++) {
				float curData = accumData[q] * invTotalWeight;
				dst[curPixelInd + q*imgSize] = curData;
			}
		} 
		else 
		{
			for(int q = 0; q < numOfChannels; q++) {
				dst[curPixelInd + q*imgSize] = totalWeight;
			}
		}

	}
}
