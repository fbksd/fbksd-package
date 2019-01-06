#include "NeuralNetwork.h"
#include "Utilities.h"

void AddBiasApplyActivation(FUNC func, Matrix<float>& A, Matrix<float>& B, bool isLastLayer);
void FilterImage(float* filteredImg, float* alpha, float* variances, int halfBlock, int width, int height);
void CalcBlockMean(float* pixelColorMu, int width, int height, int blockDelta, float* colorMu);
void CalcBlockStd(float* pixelColorMu, int width, int height, int blockDelta, float* colorMu, float* colorSigma);
void ReplaceSpikeWithMedian(float* pixelColorMu, int width, int height, int blockDelta, float spikeFactor, float* colorMu, float* colorSigma); 

NeuralNetwork::NeuralNetwork(void)
{
}

NeuralNetwork::NeuralNetwork(char* inputFolder, int width, int height, int featureLength)
{

	// Input + output + number of hidden layers
	numOfLayers = 3;
	this->width = width;
	this->height = height;
	this->featureLength = featureLength;

	// Initialize layer sizes (Sec. 4.1)
	layerSizes = new int[numOfLayers];
	layerSizes[0] = 36;
	layerSizes[1] = 10;
	layerSizes[2] = 6;

	// Initialize activation functions in the hidden and output layers (Sec. 4.1)
	activationFuncsPerLayer = new FUNC[2*(numOfLayers-1)];
	activationFuncsPerLayer[0] = SIGMOID;
	activationFuncsPerLayer[1] = SIGMOID_DERIV;
	activationFuncsPerLayer[2] = REC_LINEAR;
	activationFuncsPerLayer[3] = REC_LINEAR_DERIV;


	// Size of blocks for filtering
	blockSize = 55;
	assert(blockSize % 2 == 1);
	halfBlock = floor(blockSize / 2.0f);

	devWeights = new Matrix<float>[2*(numOfLayers-1)];
	hostWeights = new Matrix<float>[2*(numOfLayers-1)];
	
	// initializing cublas handle for fast matrix operations
	CublasErrorCheck(cublasCreate(&handle));


	char buff[BUFFER_SIZE];
	sprintf(buff, "%s/Weights.dat", inputFolder);
	LoadNodeData(devWeights, hostWeights, buff);

}

NeuralNetwork::~NeuralNetwork(void)
{

	delete[] activationFuncsPerLayer;
	delete[] devWeights;
	delete[] hostWeights;
	delete[] layerSizes;

}

void NeuralNetwork::ApplyWeightsAndFilter(char* inputFolder, char* sceneName, float* featureData, float* varData, float* img) {

	// Initialize data
	Matrix<float>* devAct = new Matrix<float>[numOfLayers];
	Matrix<float> inputFeatures(width, height, featureLength);
	Matrix<float> blockInvStd(width, height, NUM_OF_COLORS + NUM_OF_FEATURES);
	inputFeatures.setElements(featureData);
	blockInvStd.setElements(varData);
    blockInvStd.setIsCudaMat(true);
	inputFeatures.Reshape(width * height, featureLength);

	// Send secondary features to NN to get filter weights (Eq. 7)
	FeedForward(inputFeatures, devAct);

	// Filter image (Eq. 1)
	Matrix<float> devFilteredImg(width * height, NUM_OF_COLORS);
	devFilteredImg.AllocateData(true);
	devFilteredImg.SetToZero();
	FilterImage(devFilteredImg.getElements(), devAct[numOfLayers-1].getElements(), blockInvStd.getElements(), halfBlock, width, height);

	// Remove spikes (Sec. 4.3)
	SpikeRemoval(devFilteredImg, height, width);

	// Save raw output
	char buff[BUFFER_SIZE];
	sprintf(buff, "%s/%s_LBF_flt.exr", inputFolder, sceneName);
	devFilteredImg.DeviceToHost();
	devFilteredImg.Reshape(width, height, NUM_OF_COLORS);
//	devFilteredImg.saveToEXR(buff);

    float* filteredImg = devFilteredImg.getElements();
    int numPixels = width * height;
    for(int i = 0; i < numPixels; ++i)
    {
       img[i*3] = filteredImg[i];
       img[i*3 + 1] = filteredImg[i + numPixels];
       img[i*3 + 2] = filteredImg[i + numPixels*2];
    }

	// Cleanup
	delete[] devAct;
    devFilteredImg.~Matrix();
    inputFeatures.~Matrix();
    blockInvStd.~Matrix();
	CublasErrorCheck(cublasDestroy(handle));
	GpuErrorCheck(cudaFreeArray(devSampleArray));

}

void NeuralNetwork::FeedForward(Matrix<float>& input, Matrix<float>* devAct)
{

	devAct[0] = Matrix<float>(input.getWidth(), input.getHeight());
	devAct[0].AllocateData(true);
	input.HostToDevice(devAct[0]);

	for (int i = 0; i < numOfLayers - 1; i++)
	{
		devAct[i+1] = Matrix<float> (devAct[i].getWidth(), devWeights[2*i].getHeight());
		devAct[i+1].AllocateData(true);

		// Implementing Eq. 7 as a matrix multiplication
		MatMultAB(handle, devWeights[2*i], devAct[i], devAct[i+1]);
		bool isLastLayer = false;
		if(i == numOfLayers - 2) {
			isLastLayer = true;
		}
		AddBiasApplyActivation(activationFuncsPerLayer[2*i], devAct[i+1], devWeights[2*i+1], isLastLayer);
		if (i < numOfLayers - 1)
			devAct[i].~Matrix();
	}

}

void NeuralNetwork::LoadNodeData(Matrix<float>* devData, Matrix<float>* hostData, char* fileName) 
{
	// Open file
    FILE* fp = OpenFile("Weights.dat", "rb");
	
	// Read in all of the weights
	for(int i = 0; i < 2 * (numOfLayers - 1); i++) {

		// Read width and height of this data
		int width, height;
		fread(&width, sizeof(int), 1, fp);
		fread(&height, sizeof(int), 1, fp);
		assert(width > 0 && height > 0);

		// Read weights from file
		hostData[i] = Matrix<float> (width, height);
		hostData[i].AllocateData(false);

		devData[i] = Matrix<float> (width, height);
		devData[i].AllocateData(true);

		float* currentElements = hostData[i].getElements();
		fread(currentElements, sizeof(float), width * height, fp);
	
		// Save these data to this network's data
		hostData[i].HostToDevice(devData[i]);

		// Make sure we are ready for the next data
		char temp[2];
		fread(&temp, sizeof(char), 2, fp);
		assert(!strcmp(temp,"\n"));

	}

	// Close file
	fclose(fp);

}

void NeuralNetwork::SpikeRemoval(Matrix<float>& devFilteredImg, int h, int w)
{

	Matrix<float> colorMu(w, h, NUM_OF_COLORS);
	Matrix<float> colorSigma(w, h, NUM_OF_COLORS);
	colorMu.AllocateData(true);
	colorSigma.AllocateData(true);
	colorMu.SetToZero();
	colorSigma.SetToZero();
	
	CalcBlockMean(devFilteredImg.getElements(), w, h, SPIKE_WIN_SIZE_SMALL, colorMu.getElements());
	CalcBlockStd(devFilteredImg.getElements(), w, h, SPIKE_WIN_SIZE_SMALL, colorMu.getElements(), colorSigma.getElements());	
	ReplaceSpikeWithMedian(devFilteredImg.getElements(), w, h, SPIKE_WIN_SIZE_SMALL, SPIKE_FAC_NN, colorMu.getElements(), colorSigma.getElements());
	
}

void NeuralNetwork::MatMultAB(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& B, Matrix<float>& C)
{
	assert(A.getIsCudaMat() && B.getIsCudaMat() && C.getIsCudaMat());
	assert(A.getWidth() == B.getHeight());
	assert(A.getHeight() == C.getHeight() && B.getWidth() == C.getWidth());
	assert(A.getDepth() == 1 &&  B.getDepth() == 1 && C.getDepth() == 1 );

	float alpha = 1;
	float beta = 0;

	CublasErrorCheck(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B.getWidth(), A.getHeight(), A.getWidth(), &alpha, B.getElements(), B.getWidth(), A.getElements(), A.getWidth(), &beta, C.getElements(), C.getWidth()));
}

