#ifndef	RPF_GLOBALS_H_INCLUDED
#define RPF_GLOBALS_H_INCLUDED

#include <vector>
#include <random>
#include "CImg.h"

void RPF(float* result, float* pbrtData, size_t pbrtWidth,
					  size_t pbrtHeight, size_t pbrtSpp, size_t pbrtSampleLength, int posCount, int colorCount, int featureCount, int randomCount, FILE* datafp);

using namespace std;
using namespace cimg_library;

typedef unsigned char uchar;

#define BUFFER_SIZE 1000

// Sample indexes
#define X_COORD		0
#define Y_COORD		1
#define COLOR_1		2
#define COLOR_2		3
#define COLOR_3		4
#define WORLD_1_X	5
#define WORLD_1_Y	6
#define WORLD_1_Z	7
#define NORM_1_X	8
#define NORM_1_Y	9
#define NORM_1_Z	10
#define TEXTURE_1_X	11
#define TEXTURE_1_Y	12
#define TEXTURE_1_Z	13 
#define WORLD_2_X	14
#define WORLD_2_Y	15
#define WORLD_2_Z	16
#define NORM_2_X	17
#define NORM_2_Y	18
#define NORM_2_Z	19
#define TEXTURE_2_X	20
#define TEXTURE_2_Y	21
#define	TEXTURE_2_Z	22
#define U_COORD		23
#define V_COORD		24
#define TIME		25
#define LIGHT_1		26
#define LIGHT_2		27

enum OFFSET {

	X_COORD_OFFSET		=	0,
	Y_COORD_OFFSET		=	1,
	COLOR_1_OFFSET		=	0,
	COLOR_2_OFFSET		=	1,
	COLOR_3_OFFSET		=	2,
	WORLD_1_X_OFFSET	=	0,
	WORLD_1_Y_OFFSET	=	1,
	WORLD_1_Z_OFFSET	=	2,
	NORM_1_X_OFFSET		=	3,
	NORM_1_Y_OFFSET		=	4,
	NORM_1_Z_OFFSET		=	5,
	TEXTURE_1_X_OFFSET	=	6,
	TEXTURE_1_Y_OFFSET	=	7,
	TEXTURE_1_Z_OFFSET	=	8,
	WORLD_2_X_OFFSET	=	9,
	WORLD_2_Y_OFFSET	=	10,
	WORLD_2_Z_OFFSET	=	11,
	NORM_2_X_OFFSET		=	12,
	NORM_2_Y_OFFSET		=	13,
	NORM_2_Z_OFFSET		=	14,
	TEXTURE_2_X_OFFSET	=	15,
	TEXTURE_2_Y_OFFSET	=	16,
	TEXTURE_2_Z_OFFSET	=	17,
	U_COORD_OFFSET		=	0,
	V_COORD_OFFSET		=	1,
	TIME_OFFSET			=	2,
	LIGHT_1_OFFSET		=	3,
	LIGHT_2_OFFSET		=	4

};

//***** MISCELLANEOUS PARAMETERS *****//

// Number of threads for parallel processing
#define NUM_OF_THREADS 8

// Length of labels (categories)
#define MAX_LABEL_LENGTH int(32)

// Refers to the x and y coordinates
#define NUM_OF_POSITIONS int(2)

// Refers to the number of color channels
#define NUM_OF_COLORS int(3)

// Refers to the number of features (see list below)
#define NUM_OF_FEATURES int(18) 

// Refers to the number of random parameters we use (see list below)
#define NUM_OF_RANDOM_PARAMS int(5) 

// Refers to the total size of a sample
#define SAMPLE_LENGTH int(NUM_OF_POSITIONS + NUM_OF_COLORS + NUM_OF_RANDOM_PARAMS + NUM_OF_FEATURES)

// Refers to the number of bins used in calculating the pdfs for mutual information
#define MAX_NUM_OF_BINS int(10) 

// A small number to avoid divide by zero errors
#define EPSILON 0.00001f

// This value multiplied by the block size is the number of normal random numbers to generate
// for a given block. Each block randomly chooses a position in the large array as the starting
// point for the next iteration
#define RANDOM_SET_SIZE 100

// Enable (1) or disable (0) SSE (Note: enabling SSE gives faster results so it is the default)
#define ENABLE_SSE 1

// The value for the gamma curve when outputting the final image
#define COLOR_GAMMA 0.5f

// Argument for aligned malloc used in SSE
#define ALIGNMENT 16

#define MAX_SAMP_VAL 10.0f

// Additional miscellaneous parameters
// The width and height of the image as well as the number of samples
// per pixel, and the length of the samples that were output from the renderer
// (NOTE: sampleLength should be equal to SAMPLE_LENGTH)
extern size_t width, height, samplesPerPixel, sampleLength;

extern int posIndex, colIndex, featIndex, randIndex, numOfFeat, numOfRand, version;

// The number of box filters to use, the starting size of the block,
// the ending size of the block, and the size that the block is reduced
// for every iteration. Note: if the reduceSize causes the block size to
// be lower than the end size, then the block size is clamped at end size
extern int numOfIterations, startSize, endSize, blockReduceSize; 

// This array holds the block sizes for each iteration starting with iteration 0
extern int* blockSizes;
extern bool hasExplicitBlocks;

// The value for the gamma curve when outputting the final image
#define MSE_COLOR_GAMMA 2.2

// Min and max functions
#undef min
#undef max

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Function for choosing random position in the array of normally distributed random numbers
//#define RAND rand()/((float)(RAND_MAX + 1))
#define RAND drand48()


//***** GAUSSIAN PARAMETERS *****//

// Number of gaussian values to precompute for the normal random number generator
#define GAUSSIAN_LENGTH int(4096)

// The max number of standard deviations for the normal random number generator
#define GAUSSIAN_CUT 6.0f

// Additional parameters for gaussian (defined in globals.cpp)
extern float gaussianDelta;  
extern float gaussianDeltaInv;
extern float gaussianTable[GAUSSIAN_LENGTH];

//***** OUTPUT QUALITY PARAMETERS *****//
// Note: These parameters are sensitive and affect the quality of the output greatly. Proceed with caution.

// Variance for the bilateral filter (used during the calculation of the weights)
extern float variance;

// This is the amount the variance for the bilateral filter changes by every iteration (the variance is reduced)
extern float stdIncrease;

// (1.0f / VAR_FACTOR) is the scalar that multiplies the variance for calculating the
// the normal random numbers
extern float randVarFactor;

// The number of standard deviations a feature can be away from the mean before being considered an outlier
// (Excluding the world position feature. See the next parameter)
extern int featureStdFactor;

// The number of standard deviations that the world position feature can be away from the mean before being
// considered an outlier
extern int worldStdFactor;

// The following two parameters are used to calculate the color weight, alpha. The first iteration this factor
// corresponds to the first parameter and it is increased by the value of the second parameter each iteration.
extern float startMutual;
extern float mutualIncrease;

// The following two parameters are used at the last stage of each iteration. The pixel statistics are 
// calculated and if the pixel colors are above a max threshold then the value is clamped. The first
// parameter refers to the starting number of standard deviations that are used to establish whether a
// color is an outlier. After each iteration this number is reduced by dividing by the second parameter.
extern float colorStdFactor;
extern float reduceFactor;

// ADDITIONAL PARAMETERS FOR CONFIG FILE
// These are parameters that I added to the config file to avoid internal changes in the code

// This multiplies the alpha term (the weight for the color channels)
extern float alphaFactor;

// This term is an additional parameter that multiplies the texture weight
extern float textureFactor;
extern float normFactor;

#endif
