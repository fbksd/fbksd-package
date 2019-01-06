#include "FeatureExtractor.h"

void PrintError(cudaError_t err, char* file, int line);
void CalcGradients(float* featureData, float* normMu, float* normStd, int width, int height, int featureLength);
void ReplaceSpikeWithMedian(float* pixelColorMu, int width, int height, int blockDelta, float spikeFactor, float* colorMu, float* colorSigma);
void CalcBlockMean(float* pixelColorMu, int width, int height, int blockDelta, float* colorMu);
void CalcBlockStd(float* pixelColorMu, int width, int height, int blockDelta, float* colorMu, float* colorSigma);
void ComputeHaarWavelet(float* img, int width, int height, float* imgDWT, int blockSize);
void BoxFilter(float* dst, float* data, int halfBlock, int width, int height, int numOfChannels, int offset);
void ScaleVar(float* dst, float* smpVar, float* smpVarFlt, float* bufVarFlt, int width, int height, int numOfChannels, int offset);
void CalcBufVar(float* dst, float* buf0, float* buf1, int width, int height, int numOfChannels);
void CalcMeanDeviation(float* featureData, float* normMu, float* normStd, int width, int height, int featureLength);
void CalcFeatureStatistics(float* featureData, float* pixelFeatureSigma, float* normMu, float* normStd, int width, int height, int featureLength);
void FilterFeatures(float* filteredImg, float* variances, int halfBlock, int halfPatch, int indTex, float* filteredVar, int width, int height);
void AllocateGpuMemory(float *& dst, int size);

FeatureExtractor::FeatureExtractor(float* pbrtPixelData, float* pbrtVarData, float** pixelBuffer, int pbrtWidth, int pbrtHeight, int pbrtSpp, char* inputFolder) { 

	// Initialize data
	width = pbrtWidth;
	height = pbrtHeight;
	spp = pbrtSpp;
	sampleLength = SAMPLE_LENGTH;
	totalSize = width * height;
	numOfSamples = 0;
	featureLength = (int) NUM_OF_FEATURES_TO_SAVE;

	filteredData = new float[width * height * SAMPLE_LENGTH];
	pixelFeatureMu = new float[NUM_OF_FEATURES * totalSize];
	features = new float[featureLength * width * height];
	featMu = new float[featureLength];
	featStd = new float[featureLength];

	memset(pixelFeatureMu, 0, NUM_OF_FEATURES * totalSize * sizeof(float));
	memset(filteredData, 0, width * height * SAMPLE_LENGTH * sizeof(float));
	memset(features, 0, featureLength * width * height * sizeof(float));
	memset(featMu, 0, featureLength * sizeof(float));
	memset(featStd, 0, featureLength * sizeof(float));

	AllocateGpuMemory(var, (NUM_OF_COLORS + NUM_OF_FEATURES) * totalSize);
	AllocateGpuMemory(pixelFeatureSigma, NUM_OF_FEATURES * totalSize);
	AllocateGpuMemory(filterVar, NUM_OF_FEATURES * totalSize);
	AllocateGpuMemory(colorVar, NUM_OF_COLORS * totalSize);

	hostSamplesTexture = new float4[width * height * NUM_OF_CUDA_TEXTURES];

	// Read in normalization data
//	char filename[BUFFER_SIZE];
//	sprintf(filename, "%s/FeatureNorm.dat", inputFolder);
    std::string filename("FeatureNorm.dat");
    FILE* featurefp = fopen(filename.c_str(), "rb");
	if(!featurefp) {
        fprintf(stderr, "ERROR: Could not locate feature normalization info %s\n", filename.c_str());
		getchar();
		exit(-1);
	}

	float featSize;
	fread(&featSize, sizeof(float), 1, featurefp);
	assert((int) featSize == featureLength);
	fread(featMu, sizeof(float), featureLength, featurefp);
	fread(featStd, sizeof(float), featureLength, featurefp);
		
	fclose(featurefp);

	// Calculate two buffer variance
	ProcessBufferVar(pbrtVarData, pixelBuffer);
	NormalizeVar(pbrtVarData);

	// Finish loading data and send to gpu
	LoadData(pbrtPixelData, pbrtVarData, pixelBuffer);

}

FeatureExtractor::~FeatureExtractor() {

	delete[] pixelFeatureMu;
	delete[] features;
	delete[] featMu;
	delete[] featStd;
	delete[] filteredData;
	delete[] hostSamplesTexture;

	GpuErrorCheck(cudaFree(pixelFeatureSigma));

}

void FeatureExtractor::LoadData(float* pbrtPixelData, float* pbrtVarData, float** pixelBuffer) {

	size_t dataOffset = 0;
	size_t featureIndex = 0;
	int imgSize = width * height;

	for(size_t i = 0; i < width * height; i++) {

		size_t count = 0;
		size_t index = 0;
		
		for(int j = 0; j < SAMPLE_LENGTH; j++) {

			if(index >= FEATURE) {
				
				pixelFeatureMu[featureIndex] = pbrtPixelData[j + i*SAMPLE_LENGTH];				
				featureIndex++;
			
			}

			index++;
		}

		dataOffset += SAMPLE_LENGTH;
		assert(index == SAMPLE_LENGTH);

		// Update the number of samples we have
		numOfSamples++; 

	} 

	delete[] pbrtVarData;

	assert(featureIndex == NUM_OF_FEATURES * totalSize);

	SamplesToTexture(hostSamplesTexture, pbrtPixelData, width, height);
	LoadInputToCudaTexture(true);

}

void FeatureExtractor::SamplesToTexture(float4* samplesTexture, float* data, int width, int height) { 

	// This function writes the samples into a texture. The samples that are not
	// used in filtering are skipped. The arrangement is (width * numSamples, height, numTextures)
	// This is just the arrangement, but the actual data has been written in a 1D array.

	int totalSize = width * height * SAMPLE_LENGTH;
	int imgSize = width * height;
	for(int i = 0; i < height; i++) 
	{
		for(int j = 0; j < width; j++) 
		{
			// Get data at pixel
			int pixelIndex = (i* width) + j;
			

			// For all pixels
			int textureOffset = 0;
			int textureInd = 0;
			for(int q = 0; q < NUM_OF_CUDA_TEXTURES; q++) 
			{ 
				int currentSize = textureSizes[q];

				// Find the pixel in the source
				assert(pixelIndex <= totalSize);
				float* currentSampleData = data + pixelIndex * SAMPLE_LENGTH;
				float4 currentTextureData = {0.0f, 0.0f, 0.0f, 0.0f};

				currentTextureData.x = currentSampleData[textureOffset];

				if(currentSize > 1) {
					currentTextureData.y = currentSampleData[textureOffset + 1];
				} 
				if(currentSize > 2) { 
					currentTextureData.z = currentSampleData[textureOffset + 2];
				}
				if(currentSize > 3) { 
					currentTextureData.w = currentSampleData[textureOffset + 3];
				}

				samplesTexture[textureInd * width * height + pixelIndex] = currentTextureData;
				assert(textureInd * width * height + pixelIndex <= width * height * NUM_OF_CUDA_TEXTURES);
				textureInd++;

				textureOffset += currentSize;
			}
				
			assert(textureOffset == MAX_FILTER_SAMPLE_LENGTH);
			
		}
	}

}

void FeatureExtractor::LoadInputToCudaTexture(bool shouldAllocate) {

	// allocate array and copy image data
    channelDesc = cudaCreateChannelDesc<float4>();
	if(shouldAllocate) {
		GpuErrorCheck(cudaMalloc3DArray(&devSampleArray, &channelDesc, make_cudaExtent(width, height, NUM_OF_CUDA_TEXTURES), cudaArrayLayered));
	}
    cudaMemcpy3DParms textParams = {0};
    textParams.srcPos = make_cudaPos(0,0,0);
    textParams.dstPos = make_cudaPos(0,0,0);
	textParams.srcPtr = make_cudaPitchedPtr(hostSamplesTexture, width * sizeof(float4), width, height);
    textParams.dstArray = devSampleArray;
    textParams.extent = make_cudaExtent(width, height, NUM_OF_CUDA_TEXTURES);
    textParams.kind = cudaMemcpyHostToDevice;
	GpuErrorCheck(cudaMemcpy3D(&textParams));

}

float* FeatureExtractor::DeviceToHost(float* data, int size) 
{

	float* hostData = new float[size];
	GpuErrorCheck(cudaMemcpy(hostData, data, size * sizeof(float), cudaMemcpyDeviceToHost));
	
	GpuErrorCheck(cudaFree(data));
	
	return hostData;

}

float* FeatureExtractor::HostToDevice(float* data, int size) {

	
	float* devData;
	GpuErrorCheck(cudaMalloc(&(devData), size * sizeof(float))); 
	GpuErrorCheck(cudaMemcpy(devData, data, size * sizeof(float), cudaMemcpyHostToDevice));
	
	delete[] data;

	return devData;

}

float* FeatureExtractor::GetPixelFeatureMu() {
	return pixelFeatureMu;
}

float* FeatureExtractor::GetPixelFeatureSigma() {
	return pixelFeatureSigma;
}

float* FeatureExtractor::GetVar() {
	return var;
}

float* FeatureExtractor::GetFilteredData() {
	return filteredData;
}

int FeatureExtractor::GetFeatureLength() {
	return featureLength;
}

float FeatureExtractor::NormalizeVal(float val, int offset) {
	return (val - featMu[offset]) / (featStd[offset] + NORM_EPS);
}

float* FeatureExtractor::GetNormMu() {
	return featMu;
}

float* FeatureExtractor::GetNormStd() {
	return featStd;
}

float* FeatureExtractor::GetFeatures() {
	return features;
}

void FeatureExtractor::SaveVal(int x, int y, int offset, float val) {

	assert(featureLength > 0);
	assert(y >= 0 && x >= 0 && offset >= 0);
	assert(y < height && x < width && offset < featureLength);
		
	int index = (y * width + x) + offset * width * height;

	assert(index < featureLength * width * height);
	assert(features);
		
	assert(features[index] == 0);
	features[index] = NormalizeVal(val, offset);


}

void FeatureExtractor::ExtractMAD() {

	int blockSizeForMad = 8;
	float* MADFeature = new float[width * height * NUM_OF_FEATURES];
	memset(MADFeature, 0, width * height * NUM_OF_FEATURES);
	for(int q = 0; q < NUM_OF_FEATURES; q++) {
		CalcMAD(&MADFeature[q*width*height], blockSizeForMad, q + NUM_OF_COLORS);
	}

	int offset = 0;
	for(int q = 0; q < BLOCK_LENGTH; q++) {
		int nchannels = textureSizes[q + 2];
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {

				int index = i * width + j;
					
				float madAvg = 0;
				for(int k = 0; k < nchannels; k++) {
					float val = MADFeature[index + (offset + k)*width*height];
					madAvg += val;
				}
				madAvg /= nchannels;
				SaveVal(j, i, MAD_OFFSET + q, madAvg);

			}
		}
		offset += nchannels;
	}

	delete[] MADFeature;

}

// We prefilter the features as in Rousselle et al. 2013
void FeatureExtractor::PrefilterFeatures(float* pbrtPixelData, float** pixelBuffer) {

	// Init data
	float* totalWeights;
	float* currentWeightData;
	float* tempWeights;
	float* filtData;
	float* filteredVar;
	AllocateGpuMemory(totalWeights, totalSize);
	AllocateGpuMemory(currentWeightData, totalSize);
	AllocateGpuMemory(tempWeights, totalSize);
	AllocateGpuMemory(filtData, totalSize * NUM_OF_FEATURES);
	AllocateGpuMemory(filteredVar, totalSize * NUM_OF_FEATURES);

	// Filter settings
	int featureBlockSize = 11;
	int featurePatchSize = 7;
	int halfBlock = (int) floor(featureBlockSize / 2.0f);
	int halfPatch = (int) floor(featurePatchSize / 2.0f);

	int lengthProcessed = NUM_OF_COLORS + NUM_OF_POSITIONS;
	int varianceOffset = 0;

	for(int q = 0; q < NUM_OF_SIGMAS - 1; q++) {
		
		int curPos = q * 3 * width * height;
		FilterFeatures(&filtData[curPos], &filterVar[curPos], halfBlock, halfPatch, q+2, &filteredVar[curPos], width, height);

	}

	filtData = DeviceToHost(filtData, width*height*NUM_OF_FEATURES);

	lengthProcessed = 0;
	varianceOffset = 0;
	memset(filteredData, 0, totalSize * SAMPLE_LENGTH * sizeof(float));
	for(int q = 0; q < NUM_OF_SIGMAS + 1; q++) {
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				for(int k = 0; k < textureSizes[q]; k++) {
					if(q > 1) {
						filteredData[SAMPLE_LENGTH * (j + i*width) + lengthProcessed + k] = filtData[j + i*width + (k + varianceOffset)*width*height];
					} else {
						filteredData[SAMPLE_LENGTH * (j + i*width) + lengthProcessed + k] = pbrtPixelData[SAMPLE_LENGTH * (j + i*width) + lengthProcessed + k];
					}
				}
			}
		}
		lengthProcessed += textureSizes[q];
		if(q > 1) {
			varianceOffset += textureSizes[q];
		}
	}

	GpuErrorCheck(cudaMemcpy(pixelFeatureSigma, filteredVar, totalSize * NUM_OF_FEATURES * sizeof(float), cudaMemcpyDeviceToDevice));

	// Spike Removal
	float* colorMu;
	float* colorSigma;
	AllocateGpuMemory(colorMu, NUM_OF_COLORS * totalSize);
	AllocateGpuMemory(colorSigma, NUM_OF_COLORS * totalSize);
	CalcBlockMean(colorVar, width, height, SPIKE_WIN_SIZE_LARGE, colorMu);
	CalcBlockStd(colorVar, width, height, SPIKE_WIN_SIZE_LARGE, colorMu, colorSigma);
	ReplaceSpikeWithMedian(colorVar, width, height, SPIKE_WIN_SIZE_SMALL, SPIKE_FAC_FEAT, colorMu, colorSigma);
	GpuErrorCheck(cudaMemcpy(var, colorVar, NUM_OF_COLORS * totalSize * sizeof(float), cudaMemcpyDeviceToDevice));
	GpuErrorCheck(cudaMemcpy(&var[NUM_OF_COLORS * totalSize], filteredVar, NUM_OF_FEATURES * totalSize * sizeof(float), cudaMemcpyDeviceToDevice));

	// Cleanup
	for(int i = 0; i < 2; i++) { 
		delete[] pixelBuffer[i];
	}
	delete[] filtData;
	delete[] pixelBuffer;
	delete[] pbrtPixelData;
	GpuErrorCheck(cudaFree(filteredVar));
	GpuErrorCheck(cudaFree(currentWeightData));
	GpuErrorCheck(cudaFree(tempWeights));
	GpuErrorCheck(cudaFree(filterVar));
	GpuErrorCheck(cudaFree(colorVar));
	GpuErrorCheck(cudaFree(totalWeights));
	GpuErrorCheck(cudaFree(colorMu));
	GpuErrorCheck(cudaFree(colorSigma));

	// Load prefiltered data to CUDA textures
	SamplesToTexture(hostSamplesTexture, filteredData, width, height);
	LoadInputToCudaTexture(false);

}

void FeatureExtractor::ExtractGradients() {
	CalcGradients(features, featMu, featStd, width, height, featureLength);
}

void FeatureExtractor::ExtractFeatureStatistics() {
	CalcFeatureStatistics(features, pixelFeatureSigma, featMu, featStd, width, height, featureLength);
}

void FeatureExtractor::ExtractMeanDeviation() {
	CalcMeanDeviation(features, featMu, featStd, width, height, featureLength);
}

void FeatureExtractor::ExtractSpp() {

	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
			SaveVal(j, i, SPP_OFFSET, 1.0f / float(spp));
		}
	}

}

void FeatureExtractor::GetData(float* img, int offset) {

	// Loop through all the pixels in the image
	assert(offset >= NUM_OF_COLORS);
	
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
			int index = (width * i) + j;
			float val = pixelFeatureMu[NUM_OF_FEATURES * index + (offset-NUM_OF_COLORS)];
			img[i * width + j] = val;
		}
	} 

}

void FeatureExtractor::Dilate(float* input) {

	float* tmp = new float[width * height];
	memset(tmp, 0, width*height*sizeof(float));
	for (int yy = 0; yy < height; yy++)
		for(int xx = 0; xx < width; xx++)
		{
			// compare the N8 neighbors
			int pind = yy * width + xx;
			tmp[pind] = input[pind]; //important
			int ImaxInd = pind;
			float Imax = input[ImaxInd];
			int nind;	

			// left-up
			nind = MAX(0, yy - 1) * width + MAX(0, xx - 1);
			if(input[nind] > Imax)
				Imax = input[nind];
			// middle-up
			nind = MAX(0, yy - 1) * width + xx;
			if(input[nind] > Imax)
				Imax = input[nind];
			// right-up
			nind = MAX(0, yy - 1) * width + MIN(width - 1, xx + 1);
			if(input[nind] > Imax)
				Imax = input[nind];
			// right-middle
			nind = yy * width + MIN(width - 1, xx + 1);
			if(input[nind] > Imax)
				Imax = input[nind];
			// right-bottom
			nind = MIN(height - 1, yy + 1) * width + MIN(width - 1, xx + 1);
			if(input[nind] > Imax)
				Imax = input[nind];
			// middle-bottom
			nind = min(height - 1, yy + 1) * width + xx;
			if(input[nind] > Imax)
				Imax = input[nind];
			// left-bottom
			nind = MIN(height - 1, yy + 1) * width + MAX(0, xx - 1);
			if(input[nind] > Imax)
				Imax = input[nind];
			// left-middle
			nind = yy * width + max(0, xx - 1);
			if(input[nind] > Imax)
				Imax = input[nind];

			// replace with Imax
			if(Imax > input[pind])
				tmp[pind] = Imax;			  
		}

	memcpy(input, tmp, width * height * sizeof(float));
	delete[] tmp;
}

void FeatureExtractor::CalcMAD(float* MADFeature, int blockSize, int offset) {

	int numPixelsInDetails = blockSize * blockSize / 4;

	float* img = new float[width * height];
	memset(img, 0, width * height * sizeof(float));
	
	GetData(img, offset);
	
	img = HostToDevice(img, width * height);
	
	float* tmpMAD;
	AllocateGpuMemory(tmpMAD, totalSize);

	ComputeHaarWavelet(img, width, height, tmpMAD, blockSize);
	
	GpuErrorCheck(cudaMemcpy(MADFeature, tmpMAD, width * height * sizeof(float), cudaMemcpyDeviceToHost));

	/********** Necessary trick because of the wavelet **************/
	/////////////////////////////////////////////////////////////////////
	int halfWidth = width/2;
	int halfHeight = height/2;

	for(int by = 0; by < halfHeight; by++)	
		for(int bx = 0; bx < halfWidth; bx++)
		{
			float tmpMAD = 0;
			int indPixel;
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					indPixel = MIN(2 * by + i, height) * width + MIN(2 * bx + j, width);
					tmpMAD += MADFeature[indPixel];
				}
			}
			tmpMAD /= 4;
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					indPixel = MIN(2 * by + i, height) * width + MIN(2 * bx + j, width);
					MADFeature[indPixel] = tmpMAD;
				}
			}
		}
	////////////////////////////////////////////////////////////////////////
	
	
	Dilate(MADFeature);

	GpuErrorCheck(cudaFree(img));
	GpuErrorCheck(cudaFree(tmpMAD));

}


void FeatureExtractor::ProcessBufferVar(float* varData, float** pixelBuffer) {

	int imgSize = width * height;

	float varNorm = 1.0f / (spp * (spp - 1));	
	float* smpVar = new float[totalSize * NUM_OF_FEATURES];
	float* smpVarFlt;
	float* bufVarFlt;		
	float* buf0;
	float* buf1;
	float* bufVar;

	for(int k = 0; k < NUM_OF_FEATURES; k++) {
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				int index = j + i*width;
				smpVar[index + k*width*height] = varData[SAMPLE_LENGTH * index + FEATURE + k] * varNorm;
			}
		}
	}

	AllocateGpuMemory(smpVarFlt, totalSize);
	AllocateGpuMemory(bufVarFlt, totalSize);
	AllocateGpuMemory(buf0, totalSize);
	AllocateGpuMemory(buf1, totalSize);
	AllocateGpuMemory(bufVar, totalSize);

	smpVar = HostToDevice(smpVar, NUM_OF_FEATURES * totalSize);
	int boxDelta = 10;

	for(int k = 0; k < NUM_OF_FEATURES; k++) {
	
		// Calculate bufVar
		cudaMemcpy(buf0, &pixelBuffer[0][(FEATURE + k)*imgSize], totalSize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(buf1, &pixelBuffer[1][(FEATURE + k)*imgSize], totalSize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(bufVar, 0, totalSize * sizeof(float));

		CalcBufVar(bufVar, buf0, buf1, width, height, 1);

		// Filter bufVar
		BoxFilter(bufVarFlt, bufVar, boxDelta, width, height, 1, 0);

		// Filter smpVar
		BoxFilter(smpVarFlt, smpVar, boxDelta, width, height, 1, k);

		// Scale smpVar to account for bias
		ScaleVar(filterVar, smpVar, smpVarFlt, bufVarFlt, width, height, 1, k);

	}

	GpuErrorCheck(cudaFree(smpVar));

	assert(totalSize == width * height);

	smpVar = new float[totalSize * NUM_OF_COLORS];
	for(int k = 0; k < NUM_OF_COLORS; k++) {
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				int index = j + i*width;
				smpVar[index + k*width*height] = varData[SAMPLE_LENGTH * index + COLOR + k] * varNorm;
			}
		}
	}
	smpVar = HostToDevice(smpVar, NUM_OF_COLORS * totalSize);
	for(int k = 0; k < NUM_OF_COLORS; k++) {

		// Calculate bufVar
		cudaMemcpy(buf0, &pixelBuffer[0][(COLOR + k)*imgSize], totalSize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(buf1, &pixelBuffer[1][(COLOR + k)*imgSize], totalSize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(bufVar, 0, totalSize * sizeof(float));

		CalcBufVar(bufVar, buf0, buf1, width, height, 1);

		// Filter bufVar
		BoxFilter(bufVarFlt, bufVar, boxDelta, width, height, 1, 0);

		// Filter smpVar
		BoxFilter(smpVarFlt, smpVar, boxDelta, width, height, 1, k);

		// Scale smpVar to account for bias
		ScaleVar(colorVar, smpVar, smpVarFlt, bufVarFlt, width, height, 1, k);

	}

	GpuErrorCheck(cudaFree(smpVar));
	GpuErrorCheck(cudaFree(smpVarFlt));
	GpuErrorCheck(cudaFree(bufVarFlt));
	GpuErrorCheck(cudaFree(bufVar));
	GpuErrorCheck(cudaFree(buf0));
	GpuErrorCheck(cudaFree(buf1));

}

void FeatureExtractor::NormalizeVar(float* varData) {
	assert(spp > 1);
	float varNorm = 1.0f / (spp - 1);
	for(size_t i = 0; i < totalSize; i++) {
		for(int j = 0; j < SAMPLE_LENGTH; j++) {
			varData[SAMPLE_LENGTH * i + j] *= varNorm;
		}
	}
}
