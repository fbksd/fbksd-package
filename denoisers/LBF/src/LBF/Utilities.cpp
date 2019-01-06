#include "Utilities.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "cuda_runtime_api.h"

using namespace std;
//using namespace Imf;
//using namespace Imath;


FILE* OpenFile(char* fileName, char* type)
{
	FILE* fp;
//	fopen_s(&fp, fileName, type);
    fp = fopen(fileName, type);

	if(!fp) 
	{
		fprintf(stderr, "ERROR: Could not open dat file %s\n", fileName);
		getchar();
		exit(-1);
	}

	return fp;
}

// Check for CUDA errors
void PrintError(cudaError_t err, char* file, int line) 
{
	
	if(err != cudaSuccess) 
	{
		fprintf(stderr, "CUDA ERROR: %s, file %s, line(%d)\n", cudaGetErrorString(err), file, line);
		getchar();
	}

}

// Output available GPU memory
void PrintAvailableMemory() 
{

	size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

}

// Check for Cublas errors
void CublasErrorCheck(cublasStatus_t err) 
{
	
	if(err != cudaSuccess) 
	{
		fprintf(stderr, "Cublas returned error code: %d, file %s, line(%d)\n", err, __FILE__, __LINE__);
		getchar();
	}

}

//static void WriteEXRImage(const std::string &name, float *pixels, float *alpha, int xRes, int yRes,
//												int totalXRes, int totalYRes, int xOffset, int yOffset)
//{
//    Rgba *hrgba = new Rgba[xRes * yRes];
//    for (int i = 0; i < xRes * yRes; ++i)
//        hrgba[i] = Rgba(pixels[3*i], pixels[3*i+1], pixels[3*i+2],
//                        alpha ? alpha[i]: 1.f);

//    Box2i displayWindow(V2i(0,0), V2i(totalXRes-1, totalYRes-1));
//    Box2i dataWindow(V2i(xOffset, yOffset), V2i(xOffset + xRes - 1, yOffset + yRes - 1));

//    try {
//        RgbaOutputFile file(name.c_str(), displayWindow, dataWindow, WRITE_RGBA);
//        file.setFrameBuffer(hrgba - xOffset - yOffset * xRes, 1, xRes);
//        file.writePixels(yRes);
//    }
//    catch (const std::exception &e) {
//        fprintf(stderr, "Unable to write image file \"%s\": %s", name.c_str(),
//            e.what());
//    }

//    delete[] hrgba;
//}

//void WriteEXRFile(char* fileName, int xRes, int yRes, float* input)
//{

//	int Length = xRes * yRes;
//	float* alpha = new float[Length];
//	float* rgb = new float[3*Length];
//	for(int i = 0; i < xRes*yRes ; i++)
//	{
//		rgb[3*i] = input[i];
//		rgb[3*i + 1] = input[i + Length];
//		rgb[3*i + 2] = input[i + 2 * Length];

//		alpha[i] = 1;

//	}
//	std::string strFileName = fileName;

//	WriteEXRImage(strFileName,rgb,alpha,xRes,yRes,xRes,yRes,0,0);
//	delete[] alpha;
//	delete[] rgb;

//}



