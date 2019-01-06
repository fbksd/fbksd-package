#ifndef PBRT_FILM_IMAGE_HISTO_H
#define PBRT_FILM_IMAGE_HISTO_H

//Maximum number of bins
#define MAX_NUMBINS 30

#include "pbrt.h"
#include "mitchell.h"
#include <string>
#include <vector>
#include <memory>
using std::string;

// ImageFilm Declarations
class ImageFilmHisto
{
public:
    // ImageFilm Histo 09.08.12
    ImageFilmHisto(int xres, int yres, Filter *filt, int nbins, float gamma, float fMval, float fsval);
    
    ~ImageFilmHisto() {
        delete[] filterTable;
    }
    void AddSample(float* sample);
    void GetSampleExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    void GetPixelExtent(int *xstart, int *xend, int *ystart, int *yend) const;
    // returns the image and the histogram
    std::pair<std::unique_ptr<float[]>, std::unique_ptr<float[]>> WriteImage();
private:
    // ImageFilm Private Data
    int xResolution, yResolution;
    Filter *filter;
    float cropWindow[4];
    string filename;
    int xPixelStart, yPixelStart, xPixelCount, yPixelCount;
    struct Pixel {
        Pixel() {
            for (int i = 0; i < 3; ++i) Lxyz[i] = splatXYZ[i] = 0.f;
            weightSum = 0.f;
            
            /*mdelbra 09.08.12*/
            for (int i = 0; i < 3; ++i) 
                for(int j =0; j < MAX_NUMBINS; j++)
                    histLxyz[i][j] = 0.f;
            
            nsamples = 0.f;            
            
        }
        float Lxyz[3];
        float weightSum;
        float splatXYZ[3];
        float pad;
        
        //mdelbra 09.08.12 (pointer to histogram, size unknown apriori..(nbins)
        float histLxyz[3][MAX_NUMBINS];
        float nsamples;
        
    };
    std::vector<Pixel> pixels;
    float *filterTable;
    
    //mdelbra 09.08.12
    //Histogram Parameters
    int nBins;
    float gamma;
    float Mval;
    float sval;
    string histFilename;
};


#endif // PBRT_FILM_IMAGE_HISTO_H
