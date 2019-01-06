#include "imageHisto.h"


/*BEGIN ImageFilm Method Definitions by mdelbra 09.08.12 (histo)*/
ImageFilmHisto::ImageFilmHisto(int xres, int yres, Filter *filt, int nB, float gam, float fMval, float fsval)
{
    xResolution = xres;
    yResolution = yres;
    filter = filt;
    cropWindow[0] = 0.f;
    cropWindow[1] = 1.f;
    cropWindow[2] = 0.f;
    cropWindow[3] = 1.f;
    
    //Histogram parameters mdelbra 09.08.12
    nBins = nB;
    gamma = gam;
    Mval = fMval;
    sval = fsval;
    
    // Compute film image extent
    xPixelStart = Ceil2Int(xResolution * cropWindow[0]);
    xPixelCount = max(1, Ceil2Int(xResolution * cropWindow[1]) - xPixelStart);
    yPixelStart = Ceil2Int(yResolution * cropWindow[2]);
    yPixelCount = max(1, Ceil2Int(yResolution * cropWindow[3]) - yPixelStart);
    
    // Allocate film image storage
    pixels = std::vector<Pixel>(xPixelCount * yPixelCount);
    
    // Precompute filter weight table
#define FILTER_TABLE_SIZE 16
    filterTable = new float[FILTER_TABLE_SIZE * FILTER_TABLE_SIZE];
    float *ftp = filterTable;
    for (int y = 0; y < FILTER_TABLE_SIZE; ++y) {
        float fy = ((float)y + .5f) *
        filter->yWidth / FILTER_TABLE_SIZE;
        for (int x = 0; x < FILTER_TABLE_SIZE; ++x) {
            float fx = ((float)x + .5f) *
            filter->xWidth / FILTER_TABLE_SIZE;
            *ftp++ = filter->Evaluate(fx, fy);
        }
    }
}
/*END ImageFilm Method Definitions by mdelbra 09.08.12 (histo)*/


/*END AddSampleHisto mdelbra 04.03.2013*/
void ImageFilmHisto::AddSample(float* sample)
{
    // Compute sample's raster extent
    float dimageX = sample[0] - 0.5f;
    float dimageY = sample[1] - 0.5f;
    int x0 = Ceil2Int (dimageX - filter->xWidth);
    int x1 = Floor2Int(dimageX + filter->xWidth);
    int y0 = Ceil2Int (dimageY - filter->yWidth);
    int y1 = Floor2Int(dimageY + filter->yWidth);
    x0 = max(x0, xPixelStart);
    x1 = min(x1, xPixelStart + xPixelCount - 1);
    y0 = max(y0, yPixelStart);
    y1 = min(y1, yPixelStart + yPixelCount - 1);
    if ((x1-x0) < 0 || (y1-y0) < 0)
    {
        //        PBRT_SAMPLE_OUTSIDE_IMAGE_EXTENT(const_cast<CameraSample *>(&sample));
        return;
    }

    // Loop over filter support and add sample to pixel arrays
    // NOTE: color values in the sample array are in RGB.
    float xyz[3];
    RGBToXYZ(&sample[2], xyz);
    float rgb[3] = {sample[2], sample[3], sample[4]};

    // Precompute $x$ and $y$ filter table offsets
    int *ifx = ALLOCA(int, x1 - x0 + 1);
    for (int x = x0; x <= x1; ++x) {
        float fx = fabsf((x - dimageX) *
                         filter->invXWidth * FILTER_TABLE_SIZE);
        ifx[x-x0] = min(Floor2Int(fx), FILTER_TABLE_SIZE-1);
    }
    int *ify = ALLOCA(int, y1 - y0 + 1);
    for (int y = y0; y <= y1; ++y) {
        float fy = fabsf((y - dimageY) *
                         filter->invYWidth * FILTER_TABLE_SIZE);
        ify[y-y0] = min(Floor2Int(fy), FILTER_TABLE_SIZE-1);
    }
    bool syncNeeded = (filter->xWidth > 0.5f || filter->yWidth > 0.5f);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            // Evaluate filter value at $(x,y)$ pixel
            int offset = ify[y-y0]*FILTER_TABLE_SIZE + ifx[x-x0];
            float filterWt = filterTable[offset];


            // Update pixel values with filtered sample contribution
            Pixel & pixel = pixels[x - xPixelStart + (y-yPixelStart)*xPixelCount];
            pixel.Lxyz[0] += filterWt * xyz[0];
            pixel.Lxyz[1] += filterWt * xyz[1];
            pixel.Lxyz[2] += filterWt * xyz[2];
            pixel.weightSum += filterWt;

            //Add to histogram contribution mdelbra 09.08.12
            pixel.nsamples += filterWt;

            int ibL;
            int ibH;
            float wbL;
            float wbH;


            for(int j=0;j<3;j++)//Loop XYZ or RGB
            {

                float v = filterWt*rgb[j];
                v = v>0?v:0;

                /*Compress dynamical range*/
                if(gamma>1) v = pow(v,1/gamma);

                /*normalize to max_Vale*/
                if(Mval>0) v  = v/Mval;

                /*Truncate to SATURE_LEVEL*/
                v = v> sval ? sval : v;

                float fbin = v * (nBins-2);

                int ibinL = (int) (fbin); //Low bin

                /*Check out of bounds when ibinL > nbins-1;
                     only equality is possible?
                     */

                if(ibinL < nBins-2) //inbounds
                {
                    //High bin weight, Low bin weight 1-wH

                    float wH = fbin - ibinL;
                    ibL = ibinL;
                    wbL = 1.0f - wH;
                    ibH = ibinL+1;
                    wbH = wH;
                }
                else { //out of bounds... v >= 1

                    float wH = (v - 1.0f)/(sval - 1);

                    ibL = nBins-2;
                    wbL = 1.0f - wH;
                    ibH = nBins-1;
                    wbH = wH;
                }

                pixel.histLxyz[j][ibL] += wbL;
                pixel.histLxyz[j][ibH] += wbH;
            }
        }
    }
}
/*END AddSampleHisto mdelbra 04.03.2013*/


void ImageFilmHisto::GetSampleExtent(int *xstart, int *xend,
                                int *ystart, int *yend) const {
    *xstart = Floor2Int(xPixelStart + 0.5f - filter->xWidth);
    *xend   = Ceil2Int(xPixelStart + 0.5f + xPixelCount +
                       filter->xWidth);

    *ystart = Floor2Int(yPixelStart + 0.5f - filter->yWidth);
    *yend   = Ceil2Int(yPixelStart + 0.5f + yPixelCount +
                       filter->yWidth);
}


void ImageFilmHisto::GetPixelExtent(int *xstart, int *xend,
                               int *ystart, int *yend) const {
    *xstart = xPixelStart;
    *xend   = xPixelStart + xPixelCount;
    *ystart = yPixelStart;
    *yend   = yPixelStart + yPixelCount;
}



/*BEGIN WriteImageHisto mdelbra 05.03.2013*/
std::pair<std::unique_ptr<float[]>, std::unique_ptr<float[]>> ImageFilmHisto::WriteImage() {
    // Convert image to RGB and compute final pixel values
    int nPix = xPixelCount * yPixelCount;
    std::unique_ptr<float[]> rgb(new float[3*nPix]);
    
    //3*nbins + nsamples per pixel
    std::unique_ptr<float[]> hist(new float[(3*nBins+1)*nPix]);
    
    int offset = 0;
    for (int y = 0; y < yPixelCount; ++y) {
        for (int x = 0; x < xPixelCount; ++x) {
            Pixel& pixel = pixels[x + y*xPixelCount];
            // Convert pixel XYZ color to RGB
            XYZToRGB(pixel.Lxyz, &rgb[3*offset]);
            
            // Normalize pixel with weight sum
            float weightSum = pixel.weightSum;
            if (weightSum != 0.f) {
                /*if (weightSum > 0.001) {*/
                float invWt = 1.f / weightSum;
                //if(rgb[3*offset  ]<0 || rgb[3*offset+1]<0 || rgb[3*offset+2 ]<0)
                //    printf("%d %d\n", x,y);
                rgb[3*offset  ] = max(0.f, rgb[3*offset  ] * invWt);
                rgb[3*offset+1] = max(0.f, rgb[3*offset+1] * invWt);
                rgb[3*offset+2] = max(0.f, rgb[3*offset+2] * invWt);
            }
            
            //Save histogram mdelbra 09.08.12
            for(int i=0;i<nBins;i++)
                for(int j=0;j<3;j++)
                    hist[j*nPix*nBins + i*nPix + offset] = pixel.histLxyz[j][i];
            hist[3*nPix*nBins + offset] = pixel.nsamples;

            ++offset;
        }
    }

    return std::make_pair(std::move(rgb), std::move(hist));
}
/*END WriteImageHisto mdelbra 05.03.2013*/
