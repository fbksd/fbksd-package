#ifndef	GLOBALS_FEATURE_H_INCLUDED
#define GLOBALS_FEATURE_H_INCLUDED

#include <vector>
#include <random>
#include "CImg.h"

using namespace std;
using namespace cimg_library;

typedef float (*ActivationFunc)(float);

void LBF(char* inputFolder, char* sceneName, float* pbrtPixelData, float* pbrtVarData, float** pixelBuffer, int pbrtWidth, 
                      int pbrtHeight, int pbrtSpp, float* img);

// For CUDA
#define GpuErrorCheck(ans) { PrintError((ans), __FILE__, __LINE__); }
#define CUDA_DEBUG 1								
#define FAST_FILTER 0
				
// Sample info
#define X_COORD			0
#define Y_COORD			1
#define COLOR_1			2
#define COLOR_2			3
#define COLOR_3			4
#define WORLD_1_X		5
#define WORLD_1_Y		6
#define WORLD_1_Z		7
#define NORM_1_X		8
#define NORM_1_Y		9
#define NORM_1_Z		10
#define TEXTURE_1_X		11
#define TEXTURE_1_Y		12
#define TEXTURE_1_Z		13 
#define TEXTURE_2_X		14
#define TEXTURE_2_Y		15
#define	TEXTURE_2_Z		16
#define VISIBILITY_1	17

#define COLOR COLOR_1
#define POSITION X_COORD
#define FEATURE WORLD_1_X 

#define NUM_OF_POSITIONS		2
#define NUM_OF_COLORS			3
#define NUM_OF_FEATURES			13
#define SAMPLE_LENGTH			(NUM_OF_POSITIONS + NUM_OF_COLORS + NUM_OF_FEATURES)

#define NUM_OF_WORLD_1			3
#define NUM_OF_NORM_1			3
#define NUM_OF_TEXTURE_1		3
#define NUM_OF_TEXTURE_2		3
#define NUM_OF_VISIBILITY_1		1

// Filter
#define NUM_OF_SIGMAS		6	// Size of filter (ie. 1 for gaussian, 2 for bilateral, etc.)
#define MAX_FILTER_SAMPLE_LENGTH 18
#define NUM_OF_CUDA_TEXTURES 7

static int textureSizes[NUM_OF_CUDA_TEXTURES] = {NUM_OF_POSITIONS, NUM_OF_COLORS, NUM_OF_WORLD_1, NUM_OF_NORM_1, NUM_OF_TEXTURE_1, NUM_OF_TEXTURE_2, NUM_OF_VISIBILITY_1};

#define MAX_EXP_VAL 10								
#define MIN_EXP_VAL -80								
#define COLOR_DISTANCE_EPSILON 1e-10f
#define DISTANCE_EPSILON 1e-4f

// Feature Extractor
#define NUM_OF_BLOCKS 2
#define BLOCK_LENGTH (NUM_OF_SIGMAS - 1)
#define NUM_OF_FEATURES_TO_SAVE (2 * BLOCK_LENGTH * NUM_OF_BLOCKS + 3 * BLOCK_LENGTH + 1)
#define MAD_OFFSET	0
#define STAT_OFFSET	BLOCK_LENGTH
#define MD_OFFSET (STAT_OFFSET + (2 * BLOCK_LENGTH * NUM_OF_BLOCKS))
#define SPP_OFFSET (MD_OFFSET + BLOCK_LENGTH)
#define GRAD_OFFSET (SPP_OFFSET + 1)

static int statBlockSizes[NUM_OF_BLOCKS] = {1,7};

// Misc
#define SPIKE_FAC_FEAT 10.0
#define SPIKE_WIN_SIZE_LARGE 27
#define SPIKE_FAC_NN 2.0
#define SPIKE_WIN_SIZE_SMALL 1

#define NORM_EPS 1.0e-6f

#define COLOR_GAMMA 2.2

#define BUFFER_SIZE 1000	

enum TERMTYPE {

	POSTERM = 0,
	COLTERM = 1,
	FEATTERM = 2

};

enum FUNC { 

	SIGMOID = 0,
	SIGMOID_DERIV = 1,
	REC_LINEAR = 2,
	REC_LINEAR_DERIV = 3,

};

#undef min
#undef max

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#if CUDA_DEBUG
	#define CUDA_ERROR_CHECK \
	GpuErrorCheck(cudaPeekAtLastError()); \
	GpuErrorCheck(cudaDeviceSynchronize())
#else
	#define CUDA_ERROR_CHECK
#endif

#endif
