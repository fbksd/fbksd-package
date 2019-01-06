#ifndef	GLOBALS_H_INCLUDED
#define GLOBALS_H_INCLUDED

#include <stdio.h>

// Set to 1 to enable saving of samples
#define SAVE_SAMPLES 1		// Save the samples from Pbrt. (Turn off to simply render the scene)
#define SAVE_IMAGES 0		// To output the pbrt samples as images switch to 1
#define WRITE_SAMPLES 0		// To write the files as a dat file switch to 1

// Type for the sample elements
typedef float SampleElem; 

//#define SAMPLER_API __declspec(dllexport)
#define SAMPLER_API

class SAMPLER_API SampleWriter;

#define BUFFER_SIZE 1000

// The value for the gamma curve when outputting the final image
#define COLOR_GAMMA 0.5f

#define SAMPLE_EPSILON 0.01

// Sample indexes
enum INDEX {

	X_COORD			=	0,
	Y_COORD			=	1,
	COLOR_1			=	2,
	COLOR_2			=	3,
	COLOR_3			=	4,
	WORLD_1_X		=	5,
	WORLD_1_Y		=	6,
	WORLD_1_Z		=	7,
	NORM_1_X		=	8,
	NORM_1_Y		=	9,
	NORM_1_Z		=	10,
	TEXTURE_1_X		=	11,
	TEXTURE_1_Y		=	12,
	TEXTURE_1_Z		=	13,
	WORLD_2_X		=	14,
	WORLD_2_Y		=	15,
	WORLD_2_Z		=	16,
	NORM_2_X		=	17,
	NORM_2_Y		=	18,
	NORM_2_Z		=	19,
	TEXTURE_2_X		=	20,
	TEXTURE_2_Y		=	21,
	TEXTURE_2_Z		=	22,
	U_COORD			=	23,
	V_COORD			=	24,
	TIME			=	25,
	LIGHT_1			=	26,
	LIGHT_2			=	27

};

// Sample categories
enum CATEGORY {

	POSITION	=	X_COORD,
	COLOR		=	COLOR_1,
	FEATURE		=	WORLD_1_X, 
	RANDOM		=	U_COORD

};

// Refers to the x and y coordinates
#define NUM_OF_POSITIONS		int(2)

// Refers to the number of color channels
#define NUM_OF_COLORS			int(3)

// Refers to the number of features (see list below)
#define NUM_OF_FEATURES			int(18) 

// Refers to the number of random parameters we use (see list below)
#define NUM_OF_RANDOM_PARAMS	int(5) 

// Refers to the total size of a sample
#define SAMPLE_LENGTH			int(NUM_OF_POSITIONS + NUM_OF_COLORS + NUM_OF_RANDOM_PARAMS + NUM_OF_FEATURES)

// Misc. functions
#undef min
#undef max

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Resolve compatibility issue with Windows only function fopen_s
#ifndef _WINDOWS
int fopen_s(FILE **f, const char *name, const char *mode);
#endif

#endif
