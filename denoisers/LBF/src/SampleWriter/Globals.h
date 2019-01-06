#ifndef	GLOBALS_H_INCLUDED
#define GLOBALS_H_INCLUDED

#include <stdio.h>

typedef float SampleElem; 

// Set to 1 to enable saving data from pbrt (i.e., 0 is default pbrt)
#define SAVE_SAMPLES 1

// Use as a static class in Pbrt
//#define SAMPLER_API __declspec(dllexport)
#define SAMPLER_API
class SAMPLER_API SampleWriter;

// To avoid samples on pixel boundaries
#define SAMPLE_EPSILON 0.01

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
	TEXTURE_2_X_OFFSET	=	9,
	TEXTURE_2_Y_OFFSET	=	10,
	TEXTURE_2_Z_OFFSET	=	11,
	VISIBILITY_1_OFFSET	=	12,

}; 

#endif
