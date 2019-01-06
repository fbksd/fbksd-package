#include "ExrUtilities.h"

using namespace Imf;
using namespace Imath;

FILE* OpenFile(char* fileName, char* type) {

	FILE* fp;
    fopen_s(&fp, fileName, type);

	if(!fp) {
		fprintf(stderr, "ERROR: Could not open dat file %s\n", fileName);
		getchar();
		exit(-1);
	}

	return fp;
}

void readRgba1 (const char fileName[], Array2D<Rgba> &pixels, int &width, int &height) {

	RgbaInputFile file (fileName);
	Box2i dw = file.dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;

	pixels.resizeErase (height, width);
	file.setFrameBuffer (&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
	file.readPixels (dw.min.y, dw.max.y);

}

void WriteEXRImage(const std::string &name, float *pixels, float *alpha, int xRes, int yRes,
												int totalXRes, int totalYRes, int xOffset, int yOffset) {
    Rgba *hrgba = new Rgba[xRes * yRes];
    for (int i = 0; i < xRes * yRes; ++i)
        hrgba[i] = Rgba(pixels[3*i], pixels[3*i+1], pixels[3*i+2],
                        alpha ? alpha[i]: 1.f);

    Box2i displayWindow(V2i(0,0), V2i(totalXRes-1, totalYRes-1));
    Box2i dataWindow(V2i(xOffset, yOffset), V2i(xOffset + xRes - 1, yOffset + yRes - 1));

    try {
        RgbaOutputFile file(name.c_str(), displayWindow, dataWindow, WRITE_RGBA);
        file.setFrameBuffer(hrgba - xOffset - yOffset * xRes, 1, xRes);
        file.writePixels(yRes);
    }
    catch (const std::exception &e) {
        fprintf(stderr, "Unable to write image file \"%s\": %s", name.c_str(),
            e.what());
    }

    delete[] hrgba;
}

void WriteEXRFile(char* fileName, int xRes, int yRes, float* input) {

	int Length = xRes * yRes;
	float* alpha = new float[Length];
	float* rgb = new float[3*Length];
	for(int i = 0; i < xRes*yRes ; i++) {
		rgb[3*i] = input[i];
		rgb[3*i + 1] = input[i + Length];
		rgb[3*i + 2] = input[i + 2 * Length]; 

		alpha[i] = 1;

	}
	std::string strFileName = fileName;

	WriteEXRImage(strFileName,rgb,alpha,xRes,yRes,xRes,yRes,0,0);
	delete[] alpha;
	delete[] rgb;

} 

float tonemap(float val) {

	val = std::max(val, 0.0f);
	return std::min(1.0f, powf(val, 1.0f/MSE_COLOR_GAMMA));///0.9f);

}

void convertFromInf(float& val) {

	float zero = 0.0f;
	float inf = 1.0f / zero;

	if(val == inf || val == -inf) {
		val = HALF_MAX;
	}

}

float* ImageRead(char* filename, int& width, int& height) { 

	string name = filename;
	float* imgElements;

	if(name.substr(name.find_last_of(".") + 1) == "exr") 
	{

		assert(NUM_OF_COLORS == 3);
		
		////////////////// checking if GT file exist /////////////////////////
		FILE* fp = OpenFile(filename, "rb");
		fclose(fp);
		//////////////////////////////////////////////////////////////////////


		// Handle exr
		Array2D<Rgba> gtPixels;
		readRgba1(filename, gtPixels, width, height);
		
		
		imgElements = new float[width * height * NUM_OF_COLORS];

		for(int i = 0; i < height; i++) 
		{
			for(int j = 0; j < width; j++) 
			{

				int index = i * width + j;
				Rgba currentGT = gtPixels[i][j];

				float gtR = currentGT.r;
				imgElements[index] = gtR;

				float gtG = currentGT.g;
				imgElements[index + height * width] = gtG;

				float gtB = currentGT.b;
				imgElements[index + height * width * 2] = gtB;
				
			}
		}
	} 
	else 
	{

		printf("ERROR: unrecognized file type\n");
		getchar();
		exit(-1);
	
	}

	return imgElements;

}
