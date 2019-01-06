/* 
 * File:   multifilm.cpp
 * Author: Fabrice Rousselle
 * 
 * Created on March 31, 2013
 */

#include "multifilm.h"

#include <iostream>
#include <fstream>
#include "stdafx.h"
#include "parallel.h"
#include "imageio.h"
#include <limits>
#include <vector>
#include <sys/param.h>
#include <string.h>
#include "box.h"
#include "multifilm.h"

#include <algorithm>
#include <numeric>      // std::accumulate

#include <fstream>

//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_linalg.h>
//#include <gsl/gsl_cblas.h>
//#include <gsl/gsl_blas.h>

const int dump_x = -1;//691;//581;//437;//410;//406;
const int dump_y = -1;//1024-255-1;//1024-455-1;//1024-185-1;//1024-256-1;//1024-261-1;


// MultiFilm Method Definitions
MultiFilm::MultiFilm(int xres, int yres, Filter *filt, int wnd_rad)
    : Film(xres, yres) {
//    memcpy(cropWindow, crop, 4 * sizeof(float));
    filename = "result";
    cropWindow[0] = 0.f;
    cropWindow[1] = 1.f;
    cropWindow[2] = 0.f;
    cropWindow[3] = 1.f;
    // Compute film image extent
    xPixelStart = Ceil2Int(xResolution * cropWindow[0]);
    xPixelCount = max(1, Ceil2Int(xResolution * cropWindow[1]) - xPixelStart);
    yPixelStart = Ceil2Int(yResolution * cropWindow[2]);
    yPixelCount = max(1, Ceil2Int(yResolution * cropWindow[3]) - yPixelStart);
    filter = new FeatureFilter(xPixelCount, yPixelCount);

    // Make sure we're using a box filter
    BoxFilter *boxFilter = dynamic_cast<BoxFilter *> (filt);
    if (boxFilter == NULL) {
        Error("MultiFilm: only BOX pixel filter is supported!");
    }
    else if (boxFilter->xWidth != 0.5 || boxFilter->yWidth != 0.5) {
        Error("MultiFilm: only BOX pixel filter of width 0.5 supported!");
    }
    
    // Possibly open window for image display
//    if (openWindow || PbrtOptions.openWindow) {
//        Warning("Support for opening image display window not available in this build.");
//    }

    // Allocate film image storage
    numbuffers = 2;
    npixels = xPixelCount * yPixelCount;
    buffers_nsamples.resize(numbuffers);
    buffers_rgb.resize(numbuffers);
    buffers_normal.resize(numbuffers);
    buffers_albedo.resize(numbuffers);
    buffers_position.resize(numbuffers);
    buffers_visibility.resize(numbuffers);
    buffers_caustic.resize(numbuffers);
    for (int i = 0; i < numbuffers; i++) {
        buffers_nsamples[i].resize(npixels);
        buffers_rgb[i].resize(3*npixels);
        buffers_normal[i].resize(3*npixels);
        buffers_albedo[i].resize(3*npixels);
        buffers_position[i].resize(3*npixels);
        buffers_visibility[i].resize(npixels);
        buffers_caustic[i].resize(3*npixels);
    }
    statistics.Init(npixels);
    reference.resize(3*xPixelCount*yPixelCount);
    sel_map.resize(3*npixels);
    
    // Store user parameters
    this->wnd_rad = wnd_rad;
    
    itr_count = 0;
    
}


void MultiFilm::AddSample(const float* sample, int target) {
    int x = sample[IMAGE_X];
    int y = sample[IMAGE_Y];

//    if(sample[COLOR_B] > 1.f)
//        return;

    // Loop over filter support and add sample to pixel arrays
    float rgb[3];
    memcpy(rgb, &sample[COLOR_R], 3*sizeof(float));
    float albedo[3];
    memcpy(albedo, &sample[TEXTURE_R], 3*sizeof(float));
    float albedoMax = std::max(1.f, std::max(albedo[0], std::max(albedo[1], albedo[2])));
    albedo[0] /= albedoMax;
    albedo[1] /= albedoMax;
    albedo[2] /= albedoMax;
    float caustic[3] = {1.f, 1.f, 1.f};
    float xyz[3];
    memcpy(xyz, &sample[WORLD_X], 3*sizeof(float));
    float position[3];
    position[0] =  3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2];
    position[1] = -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2];
    position[2] =  0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2];
    float normal[3];
    memcpy(normal, &sample[NORMAL_X], 3*sizeof(float));
//    float vis = (sample[DIRECT_R] + sample[DIRECT_G] + sample[DIRECT_B])/3.f > 0.f ? 1.f : 0.f;
    float vis = sample[DIRECT_R] > 0.f ? 1.f : 0.f;

    // Always use AtomicAdd since adaptive sampling might be using large kernels
    // Update pixel values with sample contribution
    float delta;
    int pix = xPixelCount * (y - yPixelStart) + x - xPixelStart;
    buffers_nsamples[target][pix] += 1;
    for (int c = 0; c < 3; c++) {
        // rgb
        delta = rgb[c] - buffers_rgb[target][3*pix+c];
        buffers_rgb[target][3*pix+c] += delta / buffers_nsamples[target][pix];
        // normal
        if (!std::isnan(normal[c])) {
            delta = (normal[c] + 1.f) / 2.f - buffers_normal[target][3*pix+c];
            buffers_normal[target][3*pix+c] += delta / buffers_nsamples[target][pix];
        }
        // albedo
        delta = albedo[c] - buffers_albedo[target][3*pix+c];
        buffers_albedo[target][3*pix+c] += delta / buffers_nsamples[target][pix];
        // position
        delta = position[c] - buffers_position[target][3*pix+c];
        buffers_position[target][3*pix+c] += delta / buffers_nsamples[target][pix];
        // caustic
        delta = caustic[c] - buffers_caustic[target][3*pix+c];
        buffers_caustic[target][3*pix+c] += delta / buffers_nsamples[target][pix];
    }
    // visibility
    delta = vis - buffers_visibility[target][pix];
    buffers_visibility[target][pix] += delta / buffers_nsamples[target][pix];
    // Store variance information
    statistics.nsamples[pix] += 1;
    for (int c = 0; c < 3; c++) {
        // rgb
        delta = rgb[c] - statistics.rgb_mean[3*pix+c];
        statistics.rgb_mean[3*pix+c] += delta / statistics.nsamples[pix];
        statistics.rgb_m2[3*pix+c] += delta * (rgb[c] - statistics.rgb_mean[3*pix+c]);
        // normal
        if (!std::isnan(normal[c])) {
            delta = (normal[c] + 1.f) / 2.f - statistics.normal_mean[3*pix+c];
            statistics.normal_mean[3*pix+c] += delta / statistics.nsamples[pix];
            statistics.normal_m2[3*pix+c] += delta * ((normal[c] + 1.f) / 2.f - statistics.normal_mean[3*pix+c]);
        }
        // albedo
        delta = albedo[c] - statistics.albedo_mean[3*pix+c];
        statistics.albedo_mean[3*pix+c] += delta / statistics.nsamples[pix];
        statistics.albedo_m2[3*pix+c] += delta * (albedo[c] - statistics.albedo_mean[3*pix+c]);
        // position
        delta = position[c] - statistics.position_mean[3*pix+c];
        statistics.position_mean[3*pix+c] += delta / statistics.nsamples[pix];
        statistics.position_m2[3*pix+c] += delta * (position[c] - statistics.position_mean[3*pix+c]);
        // caustic
        delta = caustic[c] - statistics.caustic_mean[3*pix+c];
        statistics.caustic_mean[3*pix+c] += delta / statistics.nsamples[pix];
        statistics.caustic_m2[3*pix+c] += delta * (caustic[c] - statistics.caustic_mean[3*pix+c]);
    }
    // visibility
    delta = vis - statistics.visibility_mean[pix];
    statistics.visibility_mean[pix] += delta / statistics.nsamples[pix];
    statistics.visibility_m2[pix] += delta * (vis - statistics.visibility_mean[pix]);
}


void MultiFilm::GetSampleExtent(int *xstart, int *xend,
                                int *ystart, int *yend) const {
    *xstart = xPixelStart;
    *xend   = xPixelStart + xPixelCount + 1.f;

    *ystart = yPixelStart;
    *yend   = yPixelStart + yPixelCount + 1.f;
}


void MultiFilm::GetPixelExtent(int *xstart, int *xend,
                               int *ystart, int *yend) const {
    *xstart = xPixelStart;
    *xend   = xPixelStart + xPixelCount;
    *ystart = yPixelStart;
    *yend   = yPixelStart + yPixelCount;
}


void MultiFilm::GetBufferMeanVariance(Buffer &mean_variance, const BufferSet &buffers, int nchannels) {
    int nbuffers = int(buffers.size());
    mean_variance.resize(nchannels * npixels);
    Buffer mean(mean_variance.size(), 0.f);
    std::fill(mean_variance.begin(), mean_variance.end(), 0.f);
    for (int b = 0; b < nbuffers; b++) {
        for (int pix = 0; pix < npixels; pix++) {
            for (int c = 0; c < nchannels; c++) {
                int idx = nchannels*pix+c;
                float delta = buffers[b][idx] - mean[idx];
                mean[idx] += delta / (b+1);
                mean_variance[idx] += delta * (buffers[b][idx] - mean[idx]);
            }
        }
    }
    for (int pix = 0; pix < npixels; pix++) {
        for (int c = 0; c < nchannels; c++) {
            mean_variance[nchannels*pix+c] /= nbuffers * (nbuffers - 1);
        }
    }
}

void MultiFilm::UpdateVariances() {
    // we directly use the sample mean variance when using random samples, but
    // use the buffer variance when using low-discrepancy samples
    // rgb
    statistics.rgb_mean_var = statistics.rgb_m2;
    for (int pix = 0; pix < npixels; pix++) {
        for (int c = 0; c < 3; c++) {
            statistics.rgb_mean_var[3*pix+c] /= statistics.nsamples[pix] * (statistics.nsamples[pix] - 1);
        }
    }
    // normal
    statistics.normal_mean_var = statistics.normal_m2;
    for (int pix = 0; pix < npixels; pix++) {
        for (int c = 0; c < 3; c++) {
            statistics.normal_mean_var[3*pix+c] /= statistics.nsamples[pix] * (statistics.nsamples[pix] - 1);
        }
    }
    // albedo
    statistics.albedo_mean_var = statistics.albedo_m2;
    for (int pix = 0; pix < npixels; pix++) {
        for (int c = 0; c < 3; c++) {
            statistics.albedo_mean_var[3*pix+c] /= statistics.nsamples[pix] * (statistics.nsamples[pix] - 1);
        }
    }
    // caustic
    statistics.caustic_mean_var = statistics.caustic_m2;
    for (int pix = 0; pix < npixels; pix++) {
        for (int c = 0; c < 3; c++) {
            statistics.caustic_mean_var[3*pix+c] /= statistics.nsamples[pix] * (statistics.nsamples[pix] - 1);
        }
    }
    // position
    statistics.position_mean_var = statistics.position_m2;
    for (int pix = 0; pix < npixels; pix++) {
        for (int c = 0; c < 3; c++) {
            statistics.position_mean_var[3*pix+c] /= statistics.nsamples[pix] * (statistics.nsamples[pix] - 1);
        }
    }
    // visibility
    statistics.visibility_mean_var = statistics.visibility_m2;
    for (int pix = 0; pix < npixels; pix++) {
        statistics.visibility_mean_var[pix] /= statistics.nsamples[pix] * (statistics.nsamples[pix] - 1);
    }
}


//void MultiFilm::ReadImageExr(Buffer &img, const char *name) {
//    int xsize, ysize;
//    RGBSpectrum *data = ::ReadImage(name, &xsize, &ysize);
//    img.resize(3 * xsize * ysize);
//    for (int pix = 0; pix < xsize * ysize; pix++) {
//        data[pix].ToRGB(&img[3*pix]);
//    }
//    delete [] data;
//}


//void MultiFilm::ReadFeatureRef(BufferSet &buffers, Buffer &var, const char *feat) {
//    size_t idx = filename.find_last_of("/");
//    size_t offset = (idx == string::npos) ? 0 : idx+1;
//    string tmp(filename.begin() + offset, filename.end());
//    string base(tmp.begin(), tmp.begin() + tmp.find_first_of("_"));
    
//    char filename[256];
//    buffers.resize(buffers_nsamples.size());
//    for (size_t i = 0; i < buffers.size(); i++) {
//        sprintf(filename, "references/%s_box_ref_%s_b0_r%04d.exr", base.c_str(), feat, xResolution);
//        ReadImageExr(buffers[i], filename);
//    }
//    sprintf(filename, "references/%s_box_ref_%s_var_r%04d.exr", base.c_str(), feat, xResolution);
//    ReadImageExr(var, filename);
//}


void MultiFilm::PushAllFeatures(float *maxPos) {
    filter->SetFilterMode(FILTER_WITHOUT_FEATURES);
    filter->SetWndRad(5);
    filter->SetPtcRad(3);
    filter->SetVarNumScale(1.f);
    filter->SetVarDenScale(sqr(1.f));
    
    bool filter_features = true;
    
    Buffer tmp(3*npixels);
    
    // normal
    if (use_ld_samples) {
        filter->PushGuide(buffers_normal, statistics.normal_mean_var, 3);
    }
    else {
        filter->PushGuide(statistics.normal_mean, statistics.normal_mean_var, 3);
    }
    filter->PushFeature(buffers_normal, 3, 1e-3f, filter_features);
    // albedo
    if (use_ld_samples) {
        filter->PushGuide(buffers_albedo, statistics.albedo_mean_var, 3);
    }
    else {
        filter->PushGuide(statistics.albedo_mean, statistics.albedo_mean_var, 3);
    }
    filter->PushFeature(buffers_albedo, 3, 1e-3f, filter_features);
    // position
    *maxPos = 0.f;
    for (size_t i = 0; i < statistics.position_mean.size(); i++) {
        if (!std::isinf(statistics.position_mean[i])) {
            *maxPos = max(*maxPos, statistics.position_mean[i]);
        }
    }
    if (use_ld_samples) {
        filter->PushGuide(buffers_position, statistics.position_mean_var, 3);
    }
    else {
        filter->PushGuide(statistics.position_mean, statistics.position_mean_var, 3);
    }
    filter->PushFeature(buffers_position, 3, sqr(*maxPos) * 1e-3f, filter_features);
    // visibility
    if (use_ld_samples) {
        filter->PushGuide(buffers_visibility, statistics.visibility_mean_var, 1);
    }
    else {
        filter->PushGuide(statistics.visibility_mean, statistics.visibility_mean_var, 1);
    }
    filter->PushFeature(buffers_visibility, 1, 1e-3f, filter_features);
	// caustic
	if (use_ld_samples) {
	    filter->PushGuide(buffers_caustic, statistics.caustic_mean_var, 3);
	}
	else {
	    filter->PushGuide(statistics.caustic_mean, statistics.caustic_mean_var, 3);
	}
	filter->PushFeature(buffers_caustic, 3, 1e-3f, filter_features);
	
    filter->SetFilterMode(FILTER_WITH_FEATURES);
}


void MultiFilm::GetFilteredDataFast(Buffer &filtered_data, Buffer &derivative, Buffer &filtered_data_spp) {
    filter->Reset();
    
    UpdateVariances();
    
    // push features to filter and enable their use
    float maxPos;
    PushAllFeatures(&maxPos);
    
    // push the guide image
    if (use_ld_samples) {
        filter->PushGuide(buffers_rgb, statistics.rgb_mean_var, 3);
    }
    else {
        filter->PushGuide(statistics.rgb_mean, statistics.rgb_mean_var, 3);
    }
    filter->GetGuideVariance(statistics.rgb_mean_var_flt);
    
    // filter the data
    filter->SetPtcRad(1);
    filter->SetWndRad(wnd_rad);
    filter->SetVarNumScale(1.f);
    filter->SetVarDenScale(sqr(0.45f));
    filter->SetVarDenScaleFeat(sqr(0.60f));
    filter->PushDataDelta(buffers_rgb, statistics.nsamples, 3);
    filter->GetFilteredDataAndDerivative(filtered_data, derivative, filtered_data_spp);
}


void MultiFilm::GetFilteredData(Buffer &filtered_data1, Buffer &filtered_data2) {
    filter->Reset();
    
    UpdateVariances();
    
    // push features to filter and enable their use
    float maxPos;
    PushAllFeatures(&maxPos);
    
    BufferSet scales_mse, weights, confidence, derivatives;
    vector<BufferSet> filtered_scales_buffers;
    ComputeFilteredScales(scales_mse, weights, filtered_scales_buffers, confidence, maxPos, derivatives);
    
    size_t nscales = scales_mse.size();
    BufferSet sel_weights(nscales), filtered_buffers(nscales);
    ComputeSelectionMap(sel_weights, scales_mse, maxPos, derivatives);
    ApplySelectionMap(filtered_data1, filtered_buffers, filtered_scales, filtered_scales_buffers, sel_weights);
    
    // second filter pass
    filter->SetWndRad(wnd_rad / 2);
    filter->SetPtcRad(3);
    filter->SetVarNumScale(1.f);
    filter->SetVarDenScale(sqr(0.45f));
    filter->SetVarDenScaleFeat(sqr(0.60f));
    filter->SetUseDiffVar(false);
    filter->SetFeatureThreshold(0, 1e-4f); // normal
    filter->SetFeatureThreshold(1, 1e-4f); // albedo
    filter->SetFeatureThreshold(2, sqr(maxPos) * 1e-4f); // position
    filter->SetFeatureThreshold(3, 1e-4f); // visibility
    filter->SetFeatureThreshold(4, 1e-4f); // caustic
    filter->PushGuide(filtered_buffers, 3);
    filter->PushData(filtered_buffers, 3);
    filter->GetFilteredData(filtered_data2);
}


void MultiFilm::ComputeSURE(Buffer &mse, const Buffer &img, const Buffer &ref, const Buffer &ref_var, const Buffer &derivative, float var_scale) {
    mse.resize(img.size());
    
    // compute per channel MSE, which will be averaged later
    for (size_t i = 0; i < img.size(); i++) {
        mse[i] = (sqr(img[i]-ref[i]) + 2.f * derivative[i] * var_scale * ref_var[i] - var_scale * ref_var[i]) / var_scale;
    }
}


void MultiFilm::DumpMap(Buffer &data, const string &tag, DumpType dumpType, int buf, int width, int height) {
    width = (width == -1) ? xPixelCount : width;
    height = (height == -1) ? yPixelCount : height;
    Buffer tmp(3*npixels);
    for (int i = 0; i < npixels; i++) {
        tmp[3*i+0] = tmp[3*i+1] = tmp[3*i+2] = data[i];
    }
    DumpImageRGB(tmp, tag, dumpType, buf, width, height);
}


void MultiFilm::DumpImageRGB(vector<float> &data, const string &tag, DumpType dumpType, int buf, int width, int height) {
    // Retrieve "base" name
//    string base(filename.begin(), filename.begin() + filename.find_last_of("."));
    string base = "result";

    // Generate output filename
    char name[256];
    if (dumpType == DUMP_FINAL) {
        if (buf >= 0) {
            sprintf(name, "%s_%s_b%d.exr", base.c_str(), tag.c_str(), buf);
        }
        else {
            sprintf(name, "%s_%s.exr", base.c_str(), tag.c_str());
        }
    }
    else { // (dumpType == DUMP_ITERATION)
        if (buf >= 0) {
            sprintf(name, "%s_%s_b%d_itr%03d.exr", base.c_str(), tag.c_str(), buf, itr_count);
        }
        else {
            sprintf(name, "%s_%s_itr%03d.exr", base.c_str(), tag.c_str(), itr_count);
        }
    }

    // Write to disk
    width = (width == -1) ? xPixelCount : width;
    height = (height == -1) ? yPixelCount : height;

    ::WriteImage(name, &data[0], NULL, width, height, width, height, 0, 0);
}


void MultiFilm::WriteImage(float* img) {
//    ::WriteImage(filename, &statistics.rgb_mean[0], NULL, xPixelCount, yPixelCount, xPixelCount, yPixelCount, 0, 0);
    
    // Filter out noise from data and store the result
    Buffer flt_pass1(3*npixels), flt_pass2(3*npixels);
    
    GetFilteredData(flt_pass1, flt_pass2);
    
//    DumpImageRGB(flt_pass1, "int", DUMP_FINAL);
//    DumpImageRGB(flt_pass2, "flt", DUMP_FINAL);

    memcpy(img, flt_pass2.data(), flt_pass2.size()*sizeof(float));
    
    DumpImageRGB(statistics.rgb_mean, "rgb_mean", DUMP_FINAL);
    DumpImageRGB(sel_map, "sel", DUMP_FINAL);
    DumpMap(statistics.nsamples, "spp", DUMP_FINAL);
    DumpMap(statistics.visibility_mean, "visibility_mean", DUMP_FINAL);
    DumpImageRGB(statistics.normal_mean, "normal_mean", DUMP_FINAL);
    DumpImageRGB(statistics.albedo_mean, "albedo_mean", DUMP_FINAL);
    DumpImageRGB(statistics.position_mean, "position_mean", DUMP_FINAL);
    DumpImageRGB(statistics.caustic_mean, "caustic_mean", DUMP_FINAL);

    printf("Mean sample count per buffer: [");
    for (size_t buf = 0; buf < buffers_nsamples.size(); buf++) {
        float avg = std::accumulate(buffers_nsamples[buf].begin(), buffers_nsamples[buf].end(), 0.f) / npixels;
        printf("%s%.2f", buf == 0 ? "" : ", ", avg);
    }
    printf("]\n");
    
}


void MultiFilm::UpdateDisplay(int x0, int y0, int x1, int y1, float splatScale) {
}



void MultiFilm::GetSamplingMap(int spp, int nsamples, Buffer &map) {
    // filter the data
    Buffer filtered_data(3*npixels), derivative(3*npixels), filtered_data_spp(npixels);
    GetFilteredDataFast(filtered_data, derivative, filtered_data_spp);
    
    // estimate the error
    Buffer mse(3*npixels);
    ComputeSURE(mse, filtered_data, statistics.rgb_mean, statistics.rgb_mean_var_flt, derivative);
    
    // we use the relative MSE as a basis
    map.resize(npixels);
    for (int pix = 0; pix < npixels; pix++) {
        map[pix] = max(0.f, rgb2avg(&mse[3*pix]));
        map[pix] /= (1e-2f + sqr(rgb2avg(&filtered_data[3*pix]))) * filtered_data_spp[pix];
    }
    
    // filter the map
    filter->SetPtcRad(3);
    filter->SetWndRad(10);
    filter->SetVarDenScale(sqr(1.f));
    filter->SetVarDenScaleFeat(sqr(0.60f));
    filter->PushData(map, 1);
    filter->GetFilteredData(map);
    
    // normalize the map so that it sums up to nsamples / nbuffers
    int nbuffers = int(buffers_rgb.size());
    float sum = std::accumulate(map.begin(), map.end(), 0.f);
    float nsamples_one = float(nsamples) / nbuffers;
    for (int pix = 0; pix < npixels; pix++) {
        map[pix] *= nsamples_one / sum;
    }
    
    // Clamp the map to "lim" samples per pixel max. "lim" is set to spp-1, so
    // that, even with error propagation, no more than spp can be picked
    int nPixOver1, nPixOver2;
    int lim = spp;
    do {
        nPixOver1 = 0;
        for (int pix = 0; pix < npixels; pix++) {
            if (map[pix] > lim) {
                map[pix] = lim;
                nPixOver1 += 1;
            }
        }
        // Redistribute the remaining budget over the map
        nPixOver2 = 0;
        float distributed = accumulate(map.begin(), map.end(), 0.f);
        float scale = (nsamples_one-nPixOver1*lim)/(distributed-nPixOver1*lim);
        if (scale < 0) {
            Severe("Negative scale in sample redistribution!");
        }
        for (int pix = 0; pix < npixels; pix++) {
            if (map[pix] < lim)
                map[pix] *= scale;
            
            if (map[pix] > lim)
                nPixOver2 += 1;
        }
    } while(nPixOver2 > 0);
    
    itr_count++;
}


void MultiFilm::ApplySelectionMap(Buffer &filtered_data,
        BufferSet &filtered_buffers,
        const BufferSet &filtered_scales,
        const vector<BufferSet> &filtered_scales_buffers,
        const BufferSet &sel_weights) {
    size_t nscales = filtered_scales_buffers.size();
    size_t nbuffers = filtered_scales_buffers[0].size();
    
    filtered_buffers.clear();
    filtered_buffers.resize(nbuffers);
    for (size_t b = 0; b < nbuffers; b++) {
        filtered_buffers[b].resize(3*npixels, 0.f);
    }
    
    // Compute final output using the scale selection
    std::fill(sel_map.begin(), sel_map.end(), 0.f);
    std::fill(filtered_data.begin(), filtered_data.end(), 0.f);
    for (int pix = 0; pix < npixels; pix++) {
        float acc = 0.f;
        for (size_t s = 0; s < nscales; s++) {
            acc += sel_weights[s][pix];
            if (nscales < 4) {
                sel_map[3*pix+s] += sel_weights[s][pix];
            }
            else {
                for (int c = 0; c < 3; c++) {
                    sel_map[3*pix+c] += s * sel_weights[s][pix] / (nscales-1);
                }
            }
            for (int c = 0; c < 3; c++) {
                filtered_data[3*pix+c] += sel_weights[s][pix] * filtered_scales[s][3*pix+c];
            }
            for (size_t b = 0; b < nbuffers; b++) {
                for (int c = 0; c < 3; c++) {
                    filtered_buffers[b][3*pix+c] += sel_weights[s][pix] * filtered_scales_buffers[s][b][3*pix+c];
                }
            }
        }
        for (int c = 0; c < 3; c++) {
            filtered_data[3*pix+c] /= acc;
        }
        for (size_t b = 0; b < nbuffers; b++) {
            for (int c = 0; c < 3; c++) {
                filtered_buffers[b][3*pix+c] /= acc;
            }
        }
    }
}


void MultiFilm::ComputeSelectionMap(BufferSet &sel_weights, const BufferSet &scales_mse, float maxPos, const BufferSet &derivatives) {
    size_t nscales = sel_weights.size();
    
    // filter the scales errors
    BufferSet scales_mse_flt(nscales);
    for (size_t s = 0; s < scales_mse_flt.size(); s++) {
        scales_mse_flt[s].resize(npixels);
        for (int pix = 0; pix < npixels; pix++) {
            scales_mse_flt[s][pix] = rgb2avg(&(scales_mse[s][3*pix]));
        }
    }
    float v = 1.f;
    filter->SetVarDenScale(sqr(v));
    filter->SetFilterMode(FILTER_WITHOUT_FEATURES);
    filter->SetWndRad(1);
    filter->SetPtcRad(3);
    filter->PushData(scales_mse_flt, 1);
    filter->GetFilteredDataBuffers(scales_mse_flt);
    
    // compute selection map and then filter it
    sel_map.resize(3*npixels);
    Buffer mse_min(npixels, numeric_limits<float>::infinity());
    for (int i = nscales-1; i >= 0; i--) {
        for (int pix = 0; pix < npixels; pix++) {
            if (scales_mse_flt[i][pix] < mse_min[pix]) {
                if (nscales == 1 || i > 0 || rgb2avg(&(derivatives[0][3*pix])) <= rgb2avg(&(derivatives[1][pix]))) {
                    mse_min[pix] = scales_mse_flt[i][pix];
                    sel_map[pix] = i;
                }
            }
        }
    }
    
    // compute binary maps
    BufferSet bin_maps(nscales);
    for (size_t s = 0; s < nscales; s++) {
        bin_maps[s].resize(npixels, 0.f);
    }
    for (int pix = 0; pix < npixels; pix++) {
        int sel = sel_map[pix];
        bin_maps[sel][pix] = 1;
    }
    
    filter->SetWndRad(5);
    filter->SetVarDenScale(sqr(v));
    filter->PushData(bin_maps, 1);
    filter->GetFilteredDataBuffers(sel_weights);
    
    filter->SetFeatureThreshold(0, 1e-3f, true); // normal
    filter->SetFeatureThreshold(1, 1e-3f, true); // albedo
    filter->SetFeatureThreshold(2, sqr(maxPos) * 1e-3f, true); // position
    filter->SetFeatureThreshold(3, 1e-3f, true); // visibility
    filter->SetFeatureThreshold(4, 1e-3f, true); // caustics
    filter->SetFilterMode(FILTER_WITH_FEATURES);
}


void MultiFilm::ComputeFilteredScales(BufferSet &scales_mse,
        BufferSet &weights, vector<BufferSet> &filtered_scales_buffers,
        BufferSet &confidence, float maxPos, BufferSet &derivatives) {
    // define scales parameters
    vector<int> ptc_rad_c, wnd_rad_c;
    vector<float> k_c, k_f, f_th;
    
    // Set up the parameters of the three candidate filters
    k_c.push_back(0.45f); k_c.push_back(0.45f); k_c.push_back(1e10f);
    k_f.push_back(0.60f); k_f.push_back(0.60f); k_f.push_back(0.60f);
    ptc_rad_c.push_back(1); ptc_rad_c.push_back(3); ptc_rad_c.push_back(3);
    wnd_rad_c.push_back(wnd_rad); wnd_rad_c.push_back(wnd_rad); wnd_rad_c.push_back(wnd_rad);
    f_th.push_back(1e-3f); f_th.push_back(1e-3f); f_th.push_back(1e-4f);
    
    // set filter parameters
    filter->SetFilterMode(FILTER_WITH_FEATURES);
    filter->SetWndRad(wnd_rad);
    filter->SetPtcRad(3);
    filter->SetVarNumScale(1.f);
    
    size_t nscales = k_c.size();
    scales_mse.resize(nscales);
    weights.resize(nscales);
    filtered_scales.resize(nscales);
    filtered_scales_buffers.resize(nscales);
    confidence.resize(nscales);
    derivatives.resize(nscales);
    
    // push the guide image
    if (use_ld_samples) {
        filter->PushGuide(buffers_rgb, statistics.rgb_mean_var, 3);
    }
    else {
        filter->PushGuide(statistics.rgb_mean, statistics.rgb_mean_var, 3);
    }
    filter->GetGuideVariance(statistics.rgb_mean_var_flt);
    
    for (size_t i = 0; i < k_c.size(); i++) {
        filtered_scales[i].resize(3*npixels);
        derivatives[i].resize(3*npixels);
        
        filter->SetFeatureThreshold(0, f_th[i], true); // normal
        filter->SetFeatureThreshold(1, f_th[i], true); // albedo
        filter->SetFeatureThreshold(2, sqr(maxPos) * f_th[i], true); // position
        filter->SetFeatureThreshold(3, f_th[i], true); // visibility
        filter->SetFeatureThreshold(4, f_th[i], true); // caustic
        
        // set filter parameters for this scale
        filter->SetPtcRad(ptc_rad_c[i]);
        filter->SetWndRad(wnd_rad_c[i]);
        filter->SetVarDenScale(sqr(k_c[i]));
        filter->SetVarDenScaleFeat(sqr(k_f[i]));
        
        filter->PushDataDelta(buffers_rgb, 3);
        filter->GetFilteredDataBuffers(filtered_scales_buffers[i]);
        filter->GetFilteredDataAndDerivative(filtered_scales[i], derivatives[i]);
        
        // estimate error
        ComputeSURE(scales_mse[i], filtered_scales[i], statistics.rgb_mean, statistics.rgb_mean_var_flt, derivatives[i]);
    }
    
    // Restore default thresholds
    filter->SetWndRad(wnd_rad);
    filter->SetFeatureThreshold(0, 1e-3f); // normal
    filter->SetFeatureThreshold(1, 1e-3f); // albedo
    filter->SetFeatureThreshold(2, sqr(maxPos) * 1e-3f); // position
    filter->SetFeatureThreshold(3, 1e-3f); // visibility
    filter->SetFeatureThreshold(4, 1e-3f); // caustics
}


//MultiFilm *CreateMultiFilm(const ParamSet &params, Filter *filter) {
//    string filename = params.FindOneString("filename", PbrtOptions.imageFile);
//    if (filename == "")
//#ifdef PBRT_HAS_OPENEXR
//        filename = "pbrt.exr";
//#else
//        filename = "pbrt.tga";
//#endif

//    int xres = params.FindOneInt("xresolution", 640);
//    int yres = params.FindOneInt("yresolution", 480);
//    if (PbrtOptions.quickRender) xres = max(1, xres / 4);
//    if (PbrtOptions.quickRender) yres = max(1, yres / 4);
//    bool openwin = params.FindOneBool("display", false);
//    float crop[4] = { 0, 1, 0, 1 };
//    int cwi;
//    const float *cr = params.FindFloat("cropwindow", &cwi);
//    if (cr && cwi == 4) {
//        crop[0] = Clamp(min(cr[0], cr[1]), 0., 1.);
//        crop[1] = Clamp(max(cr[0], cr[1]), 0., 1.);
//        crop[2] = Clamp(min(cr[2], cr[3]), 0., 1.);
//        crop[3] = Clamp(max(cr[2], cr[3]), 0., 1.);
//    }

//    ///////////////////////
//    // Parse denoiser flags

//    // Filter window radius
//    int wnd_rad = params.FindOneInt("wnd_rad", 10);
    
//    return new MultiFilm(xres, yres, filter, crop, filename, openwin,
//        wnd_rad);
//}
