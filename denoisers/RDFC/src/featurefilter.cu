/* 
 * File:   featurefilter.cpp
 * Author: Fabrice Rousselle
 *
 * Created on 31. mars 2013, 18:37
 */

#include "featurefilter.h"

#include <cuda.h>
#include "pbrt.h"

#define DELTA_SCALE      1.01f
#define DELTA_OFFSET_NUM 0//1e-10f
#define DELTA_OFFSET_DEN 1e-10f
#define DELTA_THRESHOLD  0//1e-6f
#define WEIGHT_EPSILON   1e-3f
#define DISTANCE_EPSILON 1e-10f
#define MAX_D2           25//16 // 4*4
#define PTC_RAD_D        1

#define BLOCK_X 1
#define BLOCK_Y 128

//#define FLAT_KERNEL
#define FLAT_KERNEL_THRESHOLD 5.f

int count = 0;

#define CUDA_CHECK_RETURN(value) {                                \
cudaError_t _m_cudaStat = value;                                  \
if (_m_cudaStat != cudaSuccess) {                                 \
    fprintf(stderr, "Error %s at line %d in file %s\n",           \
            cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1);                                                  \
    } }


// Macros used by CUDA functions to compute center and neighbor pixel indices
#define CLAMP_MIRROR(pos, pos_max) \
    ((pos) < 0) ? -(pos) : ((pos) >= (pos_max)) ? 2*(pos_max) - (pos) - 2 : (pos)
#define CALCULATE_INDEX(width, height) \
    int x = blockIdx.x * blockDim.x + threadIdx.x; \
    int y = blockIdx.y * blockDim.y + threadIdx.y; \
    if (x >= (width) || y >= (height)) return; \
    int index = (y) * (width) + x
#define CALCULATE_INDICES(width, height, dx, dy) \
    int x = blockIdx.x * blockDim.x + threadIdx.x; \
    int y = blockIdx.y * blockDim.y + threadIdx.y; \
    if (x >= (width) || y >= (height)) return; \
    int indexC = y * (width) + x; \
    int x1 = CLAMP_MIRROR(x+(dx), (width)); \
    int y1 = CLAMP_MIRROR(y+(dy), (height)); \
    int indexN = x1 + y1 * (width)
#define CALCULATE_INDICES_SYM(width, height, dx, dy) \
    int x = blockIdx.x * blockDim.x + threadIdx.x; \
    int y = blockIdx.y * blockDim.y + threadIdx.y; \
    if (x >= (width) || y >= (height)) return; \
    int indexC = y * (width) + x; \
    int x1 = CLAMP_MIRROR(x+(dx), (width)); \
    int y1 = CLAMP_MIRROR(y+(dy), (height)); \
    int indexN1 = x1 + y1 * (width); \
    int x2 = CLAMP_MIRROR(x-(dx), (width)); \
    int y2 = CLAMP_MIRROR(y-(dy), (height)); \
    int indexN2 = x2 + y2 * (width)


__global__
void apply_weights(int width, int height, float *data_filtered,
        float *wgt_max, float *wgt_sum, const float *wgt_n1, const float *wgt_n2,
        const float *data, int dx, int dy, int nbuffers, int nchannels) {
    CALCULATE_INDICES_SYM(width, height, dx, dy);
    // update max weight
    wgt_max[indexC] = max(wgt_max[indexC], wgt_n1[indexC]);
    wgt_max[indexC] = max(wgt_max[indexC], wgt_n2[indexC]);
    // update weights sum
    wgt_sum[indexC] += wgt_n1[indexC] + wgt_n2[indexC];
    // apply the weights
    int offset = 0.f;
    for (int b = 0; b < nbuffers; b++) {
        for (int c = 0; c < nchannels; c++) {
            int idxc  = offset + nchannels * indexC  + c;
            int idxn1 = offset + nchannels * indexN1 + c;
            int idxn2 = offset + nchannels * indexN2 + c;
            data_filtered[idxc] += data[idxn1] * wgt_n1[indexC] + data[idxn2] * wgt_n2[indexC];
        }
        offset += width * height * nchannels;
    }
}


__global__
void accumulate_scaled(int width, int height, float *dst, const float *src, int nchannels, float scale) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        dst[idx] += scale * src[idx];
    }
}


__global__
void accumulate_squared(int width, int height, float *dst, const float *src, int nchannels) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        dst[idx] += src[idx] * src[idx];
    }
}


__global__
void scale_data_delta(int width, int height, float *dst, const float *src, int nchannels) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        dst[idx] = DELTA_SCALE * (DELTA_OFFSET_NUM + src[idx]);
    }
}


__global__
void scale_data(int width, int height, float *data, const float scale, int nchannels) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        data[idx] *= scale;
    }
}


__global__
void derivative(int width, int height, float *dst, const float *out1,
    const float *out2, const float *in1, const float *in2, int nchannels) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        float delta_in = DELTA_OFFSET_DEN + in1[idx] - in2[idx];
        dst[idx] = out1[idx] / delta_in - out2[idx] / delta_in;
    }
}


__global__
void ratio(int width, int height, float *dst, const float *num,
    const float *den, int nchannels) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        dst[idx] = num[idx] / (1e-10f + den[idx]);
    }
}


__global__
void scale_sample_variance(int width, int height, float *dst, const float *buf_var_flt, const float *smp_var_flt, const float *smp_var, int nchannels) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        dst[idx] = smp_var[idx] * buf_var_flt[idx] / (1e-10f + smp_var_flt[idx]);
    }
}


__global__
void difference(int width, int height, float *dst, const float *out1,
    const float *out2, const float *in1, const float *in2, int nchannels) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        if (abs(out1[idx]-out2[idx]) < DELTA_THRESHOLD) {
            dst[idx] = 0;
        }
        else {
            float t1 = out1[idx];// / (DELTA_OFFSET_DEN + in1[idx] - in2[idx]);
            float t2 = out2[idx];// / (DELTA_OFFSET_DEN + in1[idx] - in2[idx]);
            dst[idx] = (t2-t1);
        }
    }
}


__global__
void update_online_variance(int width, int height, float *mean, float *M2, const float *data, int nchannels, int n) {
    CALCULATE_INDEX(width, height);
    // compute variance: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        float delta = data[idx] - mean[idx];
        mean[idx] += delta / n;
        M2[idx] += delta * (data[idx] - mean[idx]);
    }
}


__global__
void update_mean_variance(int width, int height, float *dst, const float *dat, const float *dat_sqr, int nchannels, int nbuffers) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        // compute variance: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Na.C3.AFve_algorithm
        dst[idx] = (dat_sqr[idx] - ((dat[idx]*dat[idx])/nbuffers))/(nbuffers - 1);
        // divide by number of samples to get the sample mean variance
        dst[idx] /= nbuffers;
        if (dst[idx] < 1e-6f) {
            dst[idx] = 0.f;
        }
    }
}


__global__
void sqr_gradient(int width, int height, float *dst, const float *src, int nchannels) {
    CALCULATE_INDEX(width, height);
    
    int idxC, idxN1, idxN2;
    for (int c = 0; c < nchannels; c++) {
        idxC = nchannels*index+c;
        // Horizontal
        float gN1, gN2, gHsqr = 0.f;
        if (x > 0 && x < width-1) {
            idxN1 = nchannels*(index-1)+c;
            idxN2 = nchannels*(index+1)+c;
            gN1 = (src[idxN1] - src[idxC]) / 2.f;
            gN2 = (src[idxN2] - src[idxC]) / 2.f; // div by 2 to get gradient over half-pixel
            gHsqr = min(gN1*gN1, gN2*gN2);
        }
        
        // Vertical
        float gVsqr = 0.f;
        if (y > 0 && y < height-1) {
            idxN1 = nchannels*(index-width)+c;
            idxN2 = nchannels*(index+width)+c;
            gN1 = (src[idxN1] - src[idxC]) / 2.f;
            gN2 = (src[idxN2] - src[idxC]) / 2.f; // div by 2 to get gradient over half-pixel
            gVsqr = min(gN1*gN1, gN2*gN2);
        }
        
        // Square gradient
        dst[idxC] = gHsqr + gVsqr;//0.f;//
    }
}


__global__
void get_var_th(int width, int height, float *var_th, const float *var, const float *sqr_gradient, float threshold, int nchannels, float scale) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        var_th[idx] = max(var[idx], scale * max(sqr_gradient[idx], threshold));
    }
}


__global__
void conv_box(int width, int height, float *dst, const float *src, int r, int stride, int nchannels)
{
    CALCULATE_INDEX(width, height);
    
    int r1, r2;
    // if stride > 1, we assume we're doing vertical filtering
    if (stride > 1) {
        r1 = min(r, y);
        r2 = min(r, height-1-y);
    }
    else {
        r1 = min(r, x);
        r2 = min(r, width-1-x);
    }
    int kernel_length = (r1+r2) + 1;
    
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels * index + c;
        float acc = 0;
        for (int i = -r1; i <= r2; ++ i) {
            acc += src[idx + i * nchannels * stride];
        }
        dst[idx] = acc / kernel_length;
    }
}


__global__
void conv_box_all(int width, int height,
        float *dst1, const float *src1,
        float *dst2, const float *src2,
        float *dstS, const float *srcS,
        int r, int stride, int nchannels)
{
    CALCULATE_INDEX(width, height);
    
    int r1, r2;
    // if stride > 1, we assume we're doing vertical filtering
    if (stride > 1) {
        r1 = min(r, y);
        r2 = min(r, height-1-y);
    }
    else {
        r1 = min(r, x);
        r2 = min(r, width-1-x);
    }
    int kernel_length = (r1+r2) + 1;
    
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels * index + c;
        float acc1 = 0, acc2 = 0, accS = 0;
        for (int i = -r1; i <= r2; ++ i) {
            acc1 += src1[idx + i * nchannels * stride];
            acc2 += src2[idx + i * nchannels * stride];
            accS += srcS[idx + i * nchannels * stride];
        }
        dst1[idx] = acc1 / kernel_length;
        dst2[idx] = acc2 / kernel_length;
        dstS[idx] = accS / kernel_length;
    }
}


__global__
void conv_pyr(int width, int height, float *dst, const float *src, int stride, int nchannels, float v)
{
    CALCULATE_INDEX(width, height);
    
    // restrict kernel if it overlaps the image boundary
    float k[3] = {v, 1.0f, v};
    int i1, i2;
    // for single-channel images, stride=1 for horizontal filtering and for
    // three channels images, stride=3
    if (stride > 1) {
        i1 = -min(y-1, 0);
        i2 = 3-max((y+1)-(height-1), 0);
    }
    else {
        i1 = -min(x-1, 0);
        i2 = 3-max((x+1)-(width-1), 0);
    }

    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels*index+c;
        float acc = 0.f, acc_wgt = 0.f;
        for (int i = i1; i < i2; i++) {
            acc += k[i] * src[idx + (i-1) * nchannels * stride];
            acc_wgt += k[i];
        }
        dst[idx] = acc / acc_wgt;
    }
}


__global__
void distance(int width, int height,
    float *d2_n1, float *d2_n2, float *d2_ns,
    const float *data, const float *data_var_num, const float *data_var_den,
    float var_num_scale, float var_den_scale, int dx, int dy, int nchannels, bool use_diff_var) {
    CALCULATE_INDICES_SYM(width, height, dx, dy);
    
    float d, d2_n1_val, d2_n2_val, d2_ns_val, d2_n1_acc, d2_n2_acc, d2_ns_acc;
    d2_n1_acc = 0;
    d2_n2_acc = 0;
    d2_ns_acc = 0;
    for (int c = 0; c < nchannels; c++) {
        // The center and neighbor channel indices
        int idxC  = nchannels * indexC  + c;
        int idxN1 = nchannels * indexN1 + c;
        int idxN2 = nchannels * indexN2 + c;
        
        float varC  = data_var_num[idxC];
        float varN1 = min(varC, data_var_num[idxN1]);
        float varN2 = min(varC, data_var_num[idxN2]);
        float var_denC  = data_var_den[idxC];
        float var_denN1 = data_var_den[idxN1];
        float var_denN2 = data_var_den[idxN2];
        // compute distances
        d = data[idxC] - data[idxN1];
        d2_n1_val = d * d - var_num_scale * (varC + varN1);
        d = data[idxC] - data[idxN2];
        d2_n2_val = d * d - var_num_scale * (varC + varN2);
        d = data[idxC] - (data[idxN1]+data[idxN2])/2.f;
        d2_ns_val = d * d - var_num_scale * (varC + (varN1+varN2)/4.f);
        // accumulate normalized distances
        if (use_diff_var) {
            d2_n1_acc += d2_n1_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + var_denN1));
            d2_n2_acc += d2_n2_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + var_denN2));
            d2_ns_acc += d2_ns_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + (var_denN1 + var_denN2) / 4.f));
        }
        else {
            d2_n1_acc += d2_n1_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2));
            d2_n2_acc += d2_n2_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2));
            d2_ns_acc += d2_ns_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2));
        }
    }
    d2_n1[indexC] = d2_n1_acc / nchannels;
    d2_n2[indexC] = d2_n2_acc / nchannels;
    d2_ns[indexC] = d2_ns_acc / nchannels;
}


__global__
void distance_delta(int width, int height,
    float *d2_n1_b, float *d2_n2_b, float *d2_ns_b,
    float *d2_n1_s, float *d2_n2_s, float *d2_ns_s,
    const float *data, const float *data_var_num, const float *data_var_den,
    float var_num_scale, float var_den_scale, int dx, int dy, int nchannels, bool use_diff_var) {
    CALCULATE_INDICES_SYM(width, height, dx, dy);
    
    float d, d2_n1_val, d2_n2_val, d2_ns_val;
    float d2_n1_acc_b = 0.f, d2_n2_acc_b = 0.f, d2_ns_acc_b = 0.f;
    float d2_n1_acc_s = 0.f, d2_n2_acc_s = 0.f, d2_ns_acc_s = 0.f;
    for (int c = 0; c < nchannels; c++) {
        // The center and neighbor channel indices
        int idxC  = nchannels * indexC  + c;
        int idxN1 = nchannels * indexN1 + c;
        int idxN2 = nchannels * indexN2 + c;
        
        float varC  = data_var_num[idxC];
        float varN1 = min(varC, data_var_num[idxN1]);
        float varN2 = min(varC, data_var_num[idxN2]);
        float var_denC  = data_var_den[idxC];
        float var_denN1 = data_var_den[idxN1];
        float var_denN2 = data_var_den[idxN2];
        // compute distances
        d = data[idxC] - data[idxN1];
        d2_n1_val = d * d - var_num_scale * (varC + varN1);
        d = data[idxC] - data[idxN2];
        d2_n2_val = d * d - var_num_scale * (varC + varN2);
        d = data[idxC] - (data[idxN1]+data[idxN2])/2.f;
        d2_ns_val = d * d - var_num_scale * (varC + (varN1+varN2)/4.f);
        // accumulate normalized distances
        if (use_diff_var) {
            d2_n1_acc_b += d2_n1_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + var_denN1));
            d2_n2_acc_b += d2_n2_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + var_denN2));
            d2_ns_acc_b += d2_ns_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + (var_denN1 + var_denN2) / 4.f));
        }
        else {
            d2_n1_acc_b += d2_n1_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2.f));
            d2_n2_acc_b += d2_n2_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2.f));
            d2_ns_acc_b += d2_ns_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2.f));
        }
        // compute distances
        float scaledValC = DELTA_SCALE * (DELTA_OFFSET_NUM + data[idxC]);
        d = scaledValC - data[idxN1];
        d2_n1_val = d * d - var_num_scale * (varC + varN1);
        d = scaledValC - data[idxN2];
        d2_n2_val = d * d - var_num_scale * (varC + varN2);
        d = scaledValC - (data[idxN1]+data[idxN2])/2.f;
        d2_ns_val = d * d - var_num_scale * (varC + (varN1+varN2)/4.f);
        // accumulate normalized distances
        if (use_diff_var) {
            d2_n1_acc_s += d2_n1_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + var_denN1));
            d2_n2_acc_s += d2_n2_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + var_denN2));
            d2_ns_acc_s += d2_ns_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + (var_denN1 + var_denN2) / 4.f));
        }
        else {
            d2_n1_acc_s += d2_n1_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2.f));
            d2_n2_acc_s += d2_n2_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2.f));
            d2_ns_acc_s += d2_ns_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2.f));
        }
    }
    d2_n1_b[indexC] = d2_n1_acc_b / nchannels;
    d2_n2_b[indexC] = d2_n2_acc_b / nchannels;
    d2_ns_b[indexC] = d2_ns_acc_b / nchannels;
    d2_n1_s[indexC] = d2_n1_acc_s / nchannels;
    d2_n2_s[indexC] = d2_n2_acc_s / nchannels;
    d2_ns_s[indexC] = d2_ns_acc_s / nchannels;
}


__global__
void get_weights(int width, int height,
    float *wgt_n1, float *wgt_n2, float *wgt_ns,
    const float *d2_n1, const float *d2_n2, const float *d2_ns) {
    CALCULATE_INDEX(width, height);
    wgt_n1[index] = exp(-max(0.f, d2_n1[index]));
    wgt_n2[index] = exp(-max(0.f, d2_n2[index]));
    wgt_ns[index] = exp(-max(0.f, d2_ns[index]));
}


__global__
void keep_valid_sym_weights(int width, int height,
    float *wgt_n1, float *wgt_n2, const float *wgt_ns,
    const float *d2_n1, const float *d2_n2, float d2_max) {
    CALCULATE_INDEX(width, height);
    
    float a = 0.f;
    if (d2_n1[index] < d2_max && d2_n2[index] < d2_max) {
        a = min(1.f, max(0.f, wgt_ns[index] / (wgt_n1[index] + wgt_n2[index]) - 1));
    }
    wgt_n1[index] = a * wgt_ns[index] + (1-a) * wgt_n1[index];
    wgt_n2[index] = a * wgt_ns[index] + (1-a) * wgt_n2[index];
}


__global__
void keep_min_weights(int width, int height, float *wgt, const float *wgt_new) {
    CALCULATE_INDEX(width, height);
    wgt[index] = min(wgt[index], wgt_new[index]);
}


__global__
void keep_max_weights(int width, int height, float *wgt, const float *wgt_new) {
    CALCULATE_INDEX(width, height);
    wgt[index] = max(wgt[index], wgt_new[index]);
}


__global__
void relax_data(int width, int height, float *dst, const float *src, const float *wgt, int dx, int dy, int nbuffers, int nchannels) {
    CALCULATE_INDICES(width, height, dx, dy);
    int offset = 0.f;
    for (int b = 0; b < nbuffers; b++) {
        for (int c = 0; c < nchannels; c++) {
            int idxc = offset + nchannels * indexC + c;
            int idxn = offset + nchannels * indexN + c;
            dst[idxc] += src[idxn] * wgt[indexC];
        }
        offset += width * height * nchannels;
    }
}


__global__
void normalize(int width, int height, float *dst, const float *src, const float *acc, int nbuffers, int nchannels) {
    CALCULATE_INDEX(width, height);
    int offset = 0.f;
    for (int b = 0; b < nbuffers; b++) {
        for (int c = 0; c < nchannels; c++) {
            int idx = offset + nchannels * index + c;
            dst[idx] = src[idx] / acc[index];
        }
        offset += width * height * nchannels;
    }
}


__global__
void set_val(int width, int height, float *dst, float val, int nchannels) {
    CALCULATE_INDEX(width, height);
    for (int c = 0; c < nchannels; c++) {
        int idx = nchannels * index + c;
        dst[idx] = val;
    }
}


__global__
void update_delta_dist(int width, int height,
    float *d2_n1_s, float *d2_n2_s, float *d2_ns_s,
    const float *d2_n1_b, const float *d2_n2_b, const float *d2_ns_b,
    const float *d2_n1, const float *d2_n2, const float *d2_ns,
    float wgt_center, bool add_sym) {
    CALCULATE_INDEX(width, height);
    if (add_sym) {
        d2_n1_s[index] = d2_n1[index] + wgt_center * (d2_n1_s[index] - d2_n1_b[index] + d2_n2_s[index] - d2_n2_b[index]);
        d2_n2_s[index] = d2_n2[index] + wgt_center * (d2_n2_s[index] - d2_n2_b[index] + d2_n1_s[index] - d2_n1_b[index]);
        d2_ns_s[index] = d2_ns[index] + wgt_center * (d2_ns_s[index] - d2_ns_b[index]);
    }
    else {
        d2_n1_s[index] = d2_n1[index] + wgt_center * (d2_n1_s[index] - d2_n1_b[index]);
        d2_n2_s[index] = d2_n2[index] + wgt_center * (d2_n2_s[index] - d2_n2_b[index]);
        d2_ns_s[index] = d2_ns[index] + wgt_center * (d2_ns_s[index] - d2_ns_b[index]);
    }
}


__global__
void dist_wgt_feat(int width, int height, float *wgt_n1, float *wgt_n2, float *wgt_n1_s, float *wgt_n2_s,
        const float *data, const float *data_var_num, const float *data_var_den,
        float var_num_scale, float var_den_scale, int dx, int dy, int nchannels, bool use_diff_var, float d2_max) {
    CALCULATE_INDICES_SYM(width, height, dx, dy);
    // compute the squared distance
    float d2_n1 = 0.f, d2_n2 = 0.f, d2_ns = 0.f;
    float d, d2_n1_val, d2_n2_val, d2_ns_val;
    for (int c = 0; c < nchannels; c++) {
        // The center and neighbor channel indices
        int idxC  = nchannels * indexC  + c;
        int idxN1 = nchannels * indexN1 + c;
        int idxN2 = nchannels * indexN2 + c;
        
        float varC  = data_var_num[idxC];
        float varN1 = min(varC, data_var_num[idxN1]);
        float varN2 = min(varC, data_var_num[idxN2]);
        float var_denC  = data_var_den[idxC];
        float var_denN1 = data_var_den[idxN1];
        float var_denN2 = data_var_den[idxN2];
        // compute distances
        d = data[idxC] - data[idxN1];
        d2_n1_val = d * d - var_num_scale * (varC + varN1);
        d = data[idxC] - data[idxN2];
        d2_n2_val = d * d - var_num_scale * (varC + varN2);
        d = data[idxC] - (data[idxN1]+data[idxN2])/2.f;
        d2_ns_val = d * d - var_num_scale * (varC + (varN1+varN2)/4.f);
        // accumulate normalized distances
        if (use_diff_var) {
            d2_n1 += d2_n1_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + var_denN1));
            d2_n2 += d2_n2_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + var_denN2));
            d2_ns += d2_ns_val / (DISTANCE_EPSILON + var_den_scale * (var_denC + (var_denN1 + var_denN2) / 4.f));
        }
        else {
            d2_n1 += d2_n1_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2));
            d2_n2 += d2_n2_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2));
            d2_ns += d2_ns_val / (DISTANCE_EPSILON + var_den_scale * (var_denC * 2));
        }
    }
    d2_n1 /= nchannels;
    d2_n2 /= nchannels;
    d2_ns /= nchannels;
    
    // compute the weights
#ifdef FLAT_KERNEL
    float wgt_n1_cur = d2_n1 < FLAT_KERNEL_THRESHOLD ? 1.f : 0.f;
    float wgt_n2_cur = d2_n2 < FLAT_KERNEL_THRESHOLD ? 1.f : 0.f;
    float wgt_ns_cur = d2_ns < FLAT_KERNEL_THRESHOLD ? 1.f : 0.f;
#else
    float wgt_n1_cur = exp(-max(0.f, d2_n1));
    float wgt_n2_cur = exp(-max(0.f, d2_n2));
    float wgt_ns_cur = exp(-max(0.f, d2_ns));
#endif
    // use the symmetric if we can
    float a = 0.f;
    if (d2_n1 < d2_max && d2_n2 < d2_max) {
        a = min(1.f, max(0.f, wgt_ns_cur / (wgt_n1_cur + wgt_n2_cur) - 1));
    }
    wgt_n1_cur = a * wgt_ns_cur + (1-a) * wgt_n1_cur;
    wgt_n2_cur = a * wgt_ns_cur + (1-a) * wgt_n2_cur;
    // update the output weights if the current ones are more constraining
    wgt_n1[indexC] = min(wgt_n1[indexC], wgt_n1_cur);
    wgt_n2[indexC] = min(wgt_n2[indexC], wgt_n2_cur);
    // update weights using scaled data
    wgt_n1_s[indexC] = min(wgt_n1_s[indexC], wgt_n1_cur);
    wgt_n2_s[indexC] = min(wgt_n2_s[indexC], wgt_n2_cur);
}


__global__
void update_weights_delta(int width, int height,
    float *wgt_n1, float *wgt_n2, float *wgt_n1_s, float *wgt_n2_s,
    const float *in_d2_n1, const float *in_d2_n2, const float *in_d2_ns,
    const float *in_d2_n1_b, const float *in_d2_n2_b, const float *in_d2_ns_b,
    const float *in_d2_n1_s, const float *in_d2_n2_s, const float *in_d2_ns_s,
    int dx, int dy, float wgt_center, bool add_sym, float d2_max) {
    CALCULATE_INDEX(width, height);
    
    // retrieve the relevant squared distances
    float d2_n1   = in_d2_n1[index],   d2_n2   = in_d2_n2[index],   d2_ns   = in_d2_ns[index];
    float d2_n1_b = in_d2_n1_b[index], d2_n2_b = in_d2_n2_b[index], d2_ns_b = in_d2_ns_b[index];
    float d2_n1_s = in_d2_n1_s[index], d2_n2_s = in_d2_n2_s[index], d2_ns_s = in_d2_ns_s[index];
    if (add_sym) {
        d2_n1_s = d2_n1 + wgt_center * (d2_n1_s - d2_n1_b + d2_n2_s - d2_n2_b);
        d2_n2_s = d2_n2 + wgt_center * (d2_n2_s - d2_n2_b + d2_n1_s - d2_n1_b);
        d2_ns_s = d2_ns + wgt_center * (d2_ns_s - d2_ns_b);
    }
    else {
        d2_n1_s = d2_n1 + wgt_center * (d2_n1_s - d2_n1_b);
        d2_n2_s = d2_n2 + wgt_center * (d2_n2_s - d2_n2_b);
        d2_ns_s = d2_ns + wgt_center * (d2_ns_s - d2_ns_b);
    }
    
    // compute the weights
#ifdef FLAT_KERNEL
    float wgt_n1_cur = d2_n1 < FLAT_KERNEL_THRESHOLD ? 1.f : 0.f;
    float wgt_n2_cur = d2_n2 < FLAT_KERNEL_THRESHOLD ? 1.f : 0.f;
    float wgt_ns_cur = d2_ns < FLAT_KERNEL_THRESHOLD ? 1.f : 0.f;
    float wgt_n1_s_cur = d2_n1_s < FLAT_KERNEL_THRESHOLD ? 1.f : 0.f;
    float wgt_n2_s_cur = d2_n2_s < FLAT_KERNEL_THRESHOLD ? 1.f : 0.f;
    float wgt_ns_s_cur = d2_ns_s < FLAT_KERNEL_THRESHOLD ? 1.f : 0.f;
#else
    float wgt_n1_cur = exp(-max(0.f, d2_n1));
    float wgt_n2_cur = exp(-max(0.f, d2_n2));
    float wgt_ns_cur = exp(-max(0.f, d2_ns));
    float wgt_n1_s_cur = exp(-max(0.f, d2_n1_s));
    float wgt_n2_s_cur = exp(-max(0.f, d2_n2_s));
    float wgt_ns_s_cur = exp(-max(0.f, d2_ns_s));
#endif
    
    // use symmetric weights if possible
    float a = 0.f;
    if (d2_n1 < d2_max && d2_n2 < d2_max) {
        a = min(1.f, max(0.f, wgt_ns_cur / (wgt_n1_cur + wgt_n2_cur) - 1));
    }
    wgt_n1_cur = a * wgt_ns_cur + (1-a) * wgt_n1_cur;
    wgt_n2_cur = a * wgt_ns_cur + (1-a) * wgt_n2_cur;
    //
    if (d2_n1_s < d2_max && d2_n2_s < d2_max) {
        a = min(1.f, max(0.f, wgt_ns_s_cur / (wgt_n1_s_cur + wgt_n2_s_cur) - 1));
    }
    wgt_n1_s_cur = a * wgt_ns_s_cur + (1-a) * wgt_n1_s_cur;
    wgt_n2_s_cur = a * wgt_ns_s_cur + (1-a) * wgt_n2_s_cur;
    
    // update weights if the current ones are more constraining
    wgt_n1[index] = min(wgt_n1[index], wgt_n1_cur);
    wgt_n2[index] = min(wgt_n2[index], wgt_n2_cur);
    wgt_n1_s[index] = min(wgt_n1_s[index], wgt_n1_s_cur);
    wgt_n2_s[index] = min(wgt_n2_s[index], wgt_n2_s_cur);
}


FeatureFilter::FeatureFilter(int width, int height) {
    this->width = width;
    this->height = height;
    feat_idx = 0;
    nbuffers = nbuffers_alloc = 0;
    kdata = 0.45;
    kfeat = numeric_limits<float>::infinity();
    data = data_mean = data_mean_var_num = data_mean_var_den = NULL;
    data_filtered = data_filtered_mean = NULL;
    size_img_bytes_1c = width * height * sizeof(float);
    size_img_bytes_3c = width * height * sizeof(float) * 3;
    // Allocate guide buffers
    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_mean, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_mean_var_num, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_mean_var_den, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_filtered_mean, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_s_mean, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_s_mean_filtered, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &spp_filtered, size_img_bytes_1c));
    // Allocate temporary work buffers
    CUDA_CHECK_RETURN(cudaMalloc((void **) &tmp, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &tmp2, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &tmp3, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &tmp4, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &tmp_n1, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &tmp_n2, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &tmp_ns, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d2_n1,      size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d2_n2,      size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d2_ns,      size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d2_n1_b,    size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d2_n2_b,    size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d2_ns_b,    size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d2_n1_s,    size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d2_n2_s,    size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d2_ns_s,    size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_n1,     size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_n2,     size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_ns,     size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_n1_tmp, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_n2_tmp, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_ns_tmp, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_n1_s, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_n2_s, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_ns_s, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_n1_tmp_s, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_n2_tmp_s, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_ns_tmp_s, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_sum, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_max, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_sum_s, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &wgt_max_s, size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &ones, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &conf, size_img_bytes_1c));
    
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    set_val<<<grid, block>>>(width, height, ones, 1.f, 3);
    set_val<<<grid, block>>>(width, height, conf, 1.f, 1);
    
    img.resize(3*width*height);
    
    // set default parameters
    use_diff_var = true;
    wnd_rad = 10;
    ptc_rad = 3;
    var_num_scale = 1.f;
    var_den_scale = 1.f;
    var_num_scale_feat = 1.f;
    var_den_scale_feat = 1.f;
    mode = FILTER_WITH_FEATURES;
}


FeatureFilter::~FeatureFilter() {
    CUDA_CHECK_RETURN(cudaFree(data));
    CUDA_CHECK_RETURN(cudaFree(data_mean));
    CUDA_CHECK_RETURN(cudaFree(data_mean_var_num));
    CUDA_CHECK_RETURN(cudaFree(data_mean_var_den));
    CUDA_CHECK_RETURN(cudaFree(data_filtered));
    CUDA_CHECK_RETURN(cudaFree(data_filtered_mean));
    CUDA_CHECK_RETURN(cudaFree(data_s_mean));
    CUDA_CHECK_RETURN(cudaFree(data_s_mean_filtered));
    CUDA_CHECK_RETURN(cudaFree(spp_filtered));
    for (size_t i = 0; i < features.size(); i++) {
        CUDA_CHECK_RETURN(cudaFree(features[i]));
        CUDA_CHECK_RETURN(cudaFree(features_var_num[i]));
        CUDA_CHECK_RETURN(cudaFree(features_var_den[i]));
    }
    feat_idx = 0;
    features.clear();
    features_var_num.clear();
    features_var_den.clear();
    features_nchannels.clear();
    // free temporary work buffers
    CUDA_CHECK_RETURN(cudaFree(tmp));
    CUDA_CHECK_RETURN(cudaFree(tmp2));
    CUDA_CHECK_RETURN(cudaFree(tmp3));
    CUDA_CHECK_RETURN(cudaFree(tmp4));
    CUDA_CHECK_RETURN(cudaFree(tmp_n1));
    CUDA_CHECK_RETURN(cudaFree(tmp_n2));
    CUDA_CHECK_RETURN(cudaFree(tmp_ns));
    CUDA_CHECK_RETURN(cudaFree(d2_n1));
    CUDA_CHECK_RETURN(cudaFree(d2_n2));
    CUDA_CHECK_RETURN(cudaFree(d2_ns));
    CUDA_CHECK_RETURN(cudaFree(d2_n1_b));
    CUDA_CHECK_RETURN(cudaFree(d2_n2_b));
    CUDA_CHECK_RETURN(cudaFree(d2_ns_b));
    CUDA_CHECK_RETURN(cudaFree(d2_n1_s));
    CUDA_CHECK_RETURN(cudaFree(d2_n2_s));
    CUDA_CHECK_RETURN(cudaFree(d2_ns_s));
    CUDA_CHECK_RETURN(cudaFree(wgt_n1));
    CUDA_CHECK_RETURN(cudaFree(wgt_n2));
    CUDA_CHECK_RETURN(cudaFree(wgt_ns));
    CUDA_CHECK_RETURN(cudaFree(wgt_n1_tmp));
    CUDA_CHECK_RETURN(cudaFree(wgt_n2_tmp));
    CUDA_CHECK_RETURN(cudaFree(wgt_ns_tmp));
    CUDA_CHECK_RETURN(cudaFree(wgt_n1_s));
    CUDA_CHECK_RETURN(cudaFree(wgt_n2_s));
    CUDA_CHECK_RETURN(cudaFree(wgt_ns_s));
    CUDA_CHECK_RETURN(cudaFree(wgt_n1_tmp_s));
    CUDA_CHECK_RETURN(cudaFree(wgt_n2_tmp_s));
    CUDA_CHECK_RETURN(cudaFree(wgt_ns_tmp_s));
    CUDA_CHECK_RETURN(cudaFree(wgt_sum));
    CUDA_CHECK_RETURN(cudaFree(wgt_max));
    CUDA_CHECK_RETURN(cudaFree(wgt_sum_s));
    CUDA_CHECK_RETURN(cudaFree(wgt_max_s));
    CUDA_CHECK_RETURN(cudaFree(ones));
    CUDA_CHECK_RETURN(cudaFree(conf));
}


void FeatureFilter::Reset() {
    feat_idx = 0;
}


void FeatureFilter::MallocData(size_t nbuffers) {
    if (data == NULL) {
        nbuffers_alloc = int(nbuffers);
        CUDA_CHECK_RETURN(cudaMalloc((void **) &data, nbuffers * size_img_bytes_3c));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &data_filtered, nbuffers * size_img_bytes_3c));
    }
    else if(this->nbuffers_alloc < int(nbuffers)) {
        nbuffers_alloc = int(nbuffers);
        // the buffer count changed, reallocate data buffer
        CUDA_CHECK_RETURN(cudaFree(data));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &data, nbuffers * size_img_bytes_3c));
        CUDA_CHECK_RETURN(cudaFree(data_filtered));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &data_filtered, nbuffers * size_img_bytes_3c));
    }
    this->nbuffers = nbuffers;
}


void FeatureFilter::PushFeature(const BufferSet &buffers, int nchannels,
        float threshold, bool filter) {
    // allocate memory
    if (feat_idx == features.size()) {
        features.resize(feat_idx+1);
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(features[feat_idx]), size_img_bytes_3c));
        features_var_num.resize(feat_idx+1);
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(features_var_num[feat_idx]), size_img_bytes_3c));
        features_var_den.resize(feat_idx+1);
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(features_var_den[feat_idx]), size_img_bytes_3c));
        features_nchannels.resize(feat_idx+1);
    }
    
    if (filter) {
        PushData(buffers, nchannels);
        GetDataMean(features[feat_idx], data_filtered);
        GetDataMeanVariance(features_var_num[feat_idx], data_filtered);
    }
    else {
        CUDA_CHECK_RETURN(cudaMemcpy(features[feat_idx], data_mean, nchannels * size_img_bytes_1c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(features_var_num[feat_idx], data_mean_var_num, nchannels * size_img_bytes_1c, cudaMemcpyDeviceToDevice));
    }
    features_nchannels[feat_idx] = nchannels;
    SetFeatureThreshold(feat_idx, threshold);
    feat_idx++;
}


void FeatureFilter::SetFeatureThreshold(size_t feat_idx, float threshold, bool use_grad, float scale) {
    if (feat_idx >= features.size()) return;
    
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    // get the squared gradient into the tmp buffer
    if (use_grad) {
        sqr_gradient<<<grid, block>>>(width, height, tmp, features[feat_idx], features_nchannels[feat_idx]);
    }
    else {
        CUDA_CHECK_RETURN(cudaMemset(tmp, 0, features_nchannels[feat_idx] * size_img_bytes_1c));
    }
    // store the max of (var, gradient, threshold) into var_th
    get_var_th<<<grid, block>>>(width, height, features_var_den[feat_idx], features_var_num[feat_idx], tmp, threshold, features_nchannels[feat_idx], scale);
}


void FeatureFilter::PushGuide(const BufferSet &buffers, int nchannels) {
    nchannels_data = nchannels_guide = nchannels;
    if (data == NULL || int(buffers.size()) != nbuffers) {
        MallocData(buffers.size());
    }
    
    size_t size_img_bytes = (nchannels_guide == 1) ? size_img_bytes_1c : size_img_bytes_3c;
    int offset = 0;
    for (size_t i = 0; i < buffers.size(); i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(data + offset, &(buffers[i][0]), size_img_bytes, cudaMemcpyHostToDevice));
        offset += nchannels_guide * width * height;
    }
    
    GetDataMean(data_mean, data);
    GetDataMeanVariance(data_mean_var_num, data);
    CUDA_CHECK_RETURN(cudaMemcpy(data_mean_var_den, data_mean_var_num, size_img_bytes, cudaMemcpyDeviceToDevice));
}


void FeatureFilter::PushGuide(const Buffer &mean, const Buffer &mean_var,
        int nchannels) {
    if (data == NULL) {
        MallocData(1);
    }
    
    nchannels_guide = nchannels;
    size_t size_img_bytes = (nchannels_guide == 1) ? size_img_bytes_1c : size_img_bytes_3c;
    CUDA_CHECK_RETURN(cudaMemcpy(data_mean, &(mean[0]), size_img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(data_mean_var_num, &(mean_var[0]), size_img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(data_mean_var_den, &(mean_var[0]), size_img_bytes, cudaMemcpyHostToDevice));
}


void FeatureFilter::PushGuide(const BufferSet &buffers, const Buffer &mean_var,
        int nchannels) {
    nchannels_data = nchannels_guide = nchannels;
    
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    if (data == NULL || int(buffers.size()) != nbuffers) {
        MallocData(buffers.size());
    }
    
    // push the data to the card
    size_t size_img_bytes = (nchannels_guide == 1) ? size_img_bytes_1c : size_img_bytes_3c;
    int offset = 0;
    for (size_t i = 0; i < buffers.size(); i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(data + offset, &(buffers[i][0]), size_img_bytes, cudaMemcpyHostToDevice));
        offset += nchannels_guide * width * height;
    }
    
    // compute the buffer variance
    float *buf_var = tmp2, *smp_var = tmp3;
    GetDataMean(data_mean, data);
    CUDA_CHECK_RETURN(cudaMemcpy(smp_var, &(mean_var[0]), size_img_bytes, cudaMemcpyHostToDevice));
    GetScaledSampleVariance(buf_var, data, smp_var, nchannels);
    
    CUDA_CHECK_RETURN(cudaMemcpy(data_mean_var_num, buf_var, size_img_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(data_mean_var_den, buf_var, size_img_bytes, cudaMemcpyDeviceToDevice));
}


void FeatureFilter::PushGuideNlm(const BufferSet &buffers, const Buffer &mean_var,
        int nchannels) {
    nchannels_data = nchannels_guide = nchannels;
    
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    size_t size_img_bytes = (nchannels_guide == 1) ? size_img_bytes_1c : size_img_bytes_3c;
    
    // push the guide and use the basic variance filtering
    PushGuide(buffers, nchannels);
    
    // retrieve the buffer variance and copy the sample variance
    float *buf_var = tmp2, *smp_var = tmp3;
    GetDataMeanVariance(buf_var, data, 0.f);
    CUDA_CHECK_RETURN(cudaMemcpy(smp_var, &(mean_var[0]), size_img_bytes, cudaMemcpyHostToDevice));
    
    // filter the variances
    MallocData(2);
    int offset = 0;
    CUDA_CHECK_RETURN(cudaMemcpy(data + offset, buf_var, size_img_bytes, cudaMemcpyDeviceToDevice));
    offset += nchannels * width * height;
    CUDA_CHECK_RETURN(cudaMemcpy(data + offset, smp_var, size_img_bytes, cudaMemcpyDeviceToDevice));
    feat_only = true;
    FilterData(10, 3, 2, 2.f, .65f*.65f, nchannels, nchannels, FILTER_WITHOUT_COLOR, 1.f, 0.60f * 0.60f, true);
    
    // scale the original sample variance
    float *buf_var_flt = data_filtered, *smp_var_flt = data_filtered + offset;
    scale_sample_variance<<<grid, block>>>(width, height, data_mean_var_num, buf_var_flt, smp_var_flt, smp_var, nchannels);
    CUDA_CHECK_RETURN(cudaMemcpy(data_mean_var_den, data_mean_var_num, size_img_bytes, cudaMemcpyDeviceToDevice));
}


void FeatureFilter::GetGuideVariance(Buffer &var) {
    var.resize(nchannels_guide * width * height);
    CUDA_CHECK_RETURN(cudaMemcpy(&var[0], data_mean_var_num, nchannels_guide * size_img_bytes_1c, cudaMemcpyDeviceToHost));
}


void FeatureFilter::PushData(const Buffer &buffer, int nchannels) {
    nchannels_data = nchannels;
    if (data == NULL || nbuffers != 1) {
        MallocData(1);
    }
    
    size_t size_img_bytes = (nchannels_data == 1) ? size_img_bytes_1c : size_img_bytes_3c;
    CUDA_CHECK_RETURN(cudaMemcpy(data, &(buffer[0]), size_img_bytes, cudaMemcpyHostToDevice));
    
    feat_only = var_den_scale > 1e3f;
    FilterData(wnd_rad, ptc_rad, max(ptc_rad-PTC_RAD_D, 0), var_num_scale, var_den_scale, nchannels_data, nchannels_guide, mode, var_num_scale_feat, var_den_scale_feat, use_diff_var);
}


void FeatureFilter::PushData(const BufferSet &buffers, int nchannels) {
    nchannels_data = nchannels;
    if (data == NULL || int(buffers.size()) != nbuffers) {
        MallocData(buffers.size());
    }
    
    size_t size_img_bytes = (nchannels_data == 1) ? size_img_bytes_1c : size_img_bytes_3c;
    int offset = 0;
    for (size_t i = 0; i < buffers.size(); i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(data + offset, &(buffers[i][0]), size_img_bytes, cudaMemcpyHostToDevice));
        offset += nchannels_data * width * height;
    }
    
    feat_only = var_den_scale > 1e3f;
    FilterData(wnd_rad, ptc_rad, max(ptc_rad-PTC_RAD_D, 0), var_num_scale, var_den_scale, nchannels_data, nchannels_guide, mode, var_num_scale_feat, var_den_scale_feat, use_diff_var);
}


void FeatureFilter::PushDataDelta(const BufferSet &buffers, int nchannels) {
    nchannels_data = nchannels;
    if (data == NULL || int(buffers.size()) != nbuffers) {
        MallocData(buffers.size());
    }
    
    size_t size_img_bytes = (nchannels_data == 1) ? size_img_bytes_1c : size_img_bytes_3c;
    int offset = 0;
    for (size_t i = 0; i < buffers.size(); i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(data + offset, &(buffers[i][0]), size_img_bytes, cudaMemcpyHostToDevice));
        offset += nchannels_data * width * height;
    }
    
    feat_only = var_den_scale > 1e3f;
    if (feat_only) {
        FilterData(wnd_rad, ptc_rad, 0, var_num_scale, var_den_scale, nchannels_data, nchannels_guide, mode, var_num_scale_feat, var_den_scale_feat, true);
    }
    else {
        FilterDataDelta(wnd_rad, ptc_rad, var_num_scale, var_den_scale, nchannels_data, nchannels_guide, mode, var_num_scale_feat, var_den_scale_feat);
    }
}


void FeatureFilter::PushDataDelta(const BufferSet &buffers, const Buffer &spp, int nchannels) {
    nchannels_data = nchannels;
    if (data == NULL || int(buffers.size()) != nbuffers) {
        MallocData(buffers.size());
    }
    
    size_t size_img_bytes = (nchannels_data == 1) ? size_img_bytes_1c : size_img_bytes_3c;
    int offset = 0;
    for (size_t i = 0; i < buffers.size(); i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(data + offset, &(buffers[i][0]), size_img_bytes, cudaMemcpyHostToDevice));
        offset += nchannels_data * width * height;
    }
    CUDA_CHECK_RETURN(cudaMemcpy(conf, &(spp[0]), size_img_bytes_1c, cudaMemcpyHostToDevice));
    
    feat_only = var_den_scale > 1e3f;
    FilterDataDeltaSpp(wnd_rad, ptc_rad, var_num_scale, var_den_scale, nchannels_data, nchannels_guide, mode, var_num_scale_feat, var_den_scale_feat);
}


void FeatureFilter::PushDataDelta(const Buffer &buffer, const Buffer &spp, int nchannels) {
    nchannels_data = nchannels;
    if (data == NULL) {
        MallocData(1);
    }
    
    size_t size_img_bytes = (nchannels_data == 1) ? size_img_bytes_1c : size_img_bytes_3c;
    CUDA_CHECK_RETURN(cudaMemcpy(data, &(buffer[0]), size_img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(conf, &(spp[0]), size_img_bytes_1c, cudaMemcpyHostToDevice));
    
    feat_only = var_den_scale > 1e3f;
    FilterDataDeltaSpp(wnd_rad, ptc_rad, var_num_scale, var_den_scale, nchannels_data, nchannels_guide, mode, var_num_scale_feat, var_den_scale_feat);
}


void FeatureFilter::PushDataMeanVariance(const Buffer &variance, int nchannels) {
    CUDA_CHECK_RETURN(cudaMemcpy(data_mean_var_den, &(variance[0]), nchannels * size_img_bytes_1c, cudaMemcpyHostToDevice));
}


void FeatureFilter::WeightsData(float *weights, int x, int y, int wnd_rad,
        int ptc_rad, int ptc_rad_wgt, float var_num_scale, float var_den_scale,
        int nchannels_data, int nchannels_guide, FeatureFilterMode mode,
        float var_num_scale_feat, float var_den_scale_feat) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    // index of the pixel for which we want the filter
    int idx = x + y * width;
    
    // Initialize accumulators
    set_val<<<grid, block>>>(width, height, wgt_max, WEIGHT_EPSILON, 1);
    
    // Filter again, using patch-based filtering
    for (int dy = -wnd_rad; dy <= 0; ++ dy) {
        int dx_max = (dy == 0) ? -1 : +wnd_rad;
        for (int dx = -wnd_rad; dx <= dx_max; ++ dx) {
            // initialize all weights to 1, and then update with actual weights
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n1, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n2, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            UpdateWeights(data_mean, data_mean_var_num, data_mean_var_den, var_num_scale, var_den_scale, ptc_rad, ptc_rad_wgt, dx, dy, nchannels_guide, true);
            if (mode == FILTER_WITH_FEATURES) {
                for (size_t i = 0; i < features.size(); i++) {
                    UpdateWeightsFeat(features[i], features_var_num[i], features_var_den[i],
                         var_num_scale_feat, var_den_scale_feat, 0, 0, dx, dy, features_nchannels[i], false);
                }
            }
            keep_max_weights<<<grid, block>>>(width, height, wgt_max, wgt_n1);
            keep_max_weights<<<grid, block>>>(width, height, wgt_max, wgt_n2);
            // write the weights to the buffer
            int flt1 = (wnd_rad+dx) + (wnd_rad+dy) * (2*wnd_rad+1);
            int flt2 = (wnd_rad-dx) + (wnd_rad-dy) * (2*wnd_rad+1);
            CUDA_CHECK_RETURN(cudaMemcpy(weights + flt1, wgt_n1 + idx, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK_RETURN(cudaMemcpy(weights + flt2, wgt_n2 + idx, sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    accumulate_scaled<<<grid, block>>>(width, height, wgt_sum, wgt_max, 1, 1.f);
    // write the center weight to the buffer
    int flt = wnd_rad + wnd_rad * (2*wnd_rad+1);
    CUDA_CHECK_RETURN(cudaMemcpy(weights + flt, wgt_max + idx, sizeof(float), cudaMemcpyDeviceToHost));
}


void FeatureFilter::FilterData(int wnd_rad, int ptc_rad, int ptc_rad_wgt,
        float var_num_scale, float var_den_scale, int nchannels_data,
        int nchannels_guide, FeatureFilterMode mode, float var_num_scale_feat,
        float var_den_scale_feat, bool use_diff_var) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    // Initialize accumulators
    CUDA_CHECK_RETURN(cudaMemset(data_filtered, 0, nbuffers * size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMemset(wgt_sum, 0, size_img_bytes_1c));
    set_val<<<grid, block>>>(width, height, wgt_max, WEIGHT_EPSILON, 1);
    
    // Filter again, using patch-based filtering
    for (int dy = -wnd_rad; dy <= 0; ++ dy) {
        int dx_max = (dy == 0) ? -1 : +wnd_rad;
        for (int dx = -wnd_rad; dx <= dx_max; ++ dx) {
            // initialize all weights to 1, and then update with actual weights
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n1, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n2, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            if (!feat_only && mode != FILTER_WITHOUT_COLOR) {
                UpdateWeights(data_mean, data_mean_var_num, data_mean_var_den, var_num_scale, var_den_scale, ptc_rad, ptc_rad_wgt, dx, dy, nchannels_guide, use_diff_var);
            }
            if (mode != FILTER_WITHOUT_FEATURES) {
                for (size_t i = 0; i < features.size(); i++) {
//                    UpdateWeights(features[i], features_var_num[i], features_var_den[i],
//                        var_num_scale_feat, var_den_scale_feat, 0, 0, dx, dy, features_nchannels[i], true && use_diff_var);
                    UpdateWeightsFeat(features[i], features_var_num[i], features_var_den[i],
                        var_num_scale_feat, var_den_scale_feat, 0, 0, dx, dy, features_nchannels[i], true && use_diff_var);
                }
            }
            // apply the weights to the data
            apply_weights<<<grid, block>>>(width, height, data_filtered,
                    wgt_max, wgt_sum, wgt_n1, wgt_n2, data, dx, dy, nbuffers, nchannels_data);
        }
    }
    
    // Add contribution of center pixel
    accumulate_scaled<<<grid, block>>>(width, height, wgt_sum, wgt_max, 1, 1.f);
    relax_data<<<grid, block>>>(width, height, data_filtered, data, wgt_max, 0, 0, nbuffers, nchannels_data);
    normalize<<<grid, block>>>(width, height, data_filtered, data_filtered, wgt_sum, nbuffers, nchannels_data);
}


void FeatureFilter::FilterDataDelta(int wnd_rad, int ptc_rad,
        float var_num_scale, float var_den_scale, int nchannels_data,
        int nchannels_guide, FeatureFilterMode mode, float var_num_scale_feat,
        float var_den_scale_feat) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    // Initialize accumulators
    CUDA_CHECK_RETURN(cudaMemset(data_filtered, 0, nbuffers * size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMemset(wgt_sum, 0, size_img_bytes_1c));
    set_val<<<grid, block>>>(width, height, wgt_max, WEIGHT_EPSILON, 1);
    CUDA_CHECK_RETURN(cudaMemset(data_s_mean_filtered, 0, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMemset(wgt_sum_s, 0, size_img_bytes_1c));
    set_val<<<grid, block>>>(width, height, wgt_max_s, WEIGHT_EPSILON, 1);
    
    // get the mean of the scaled data
    CUDA_CHECK_RETURN(cudaMemset(data_s_mean, 0, size_img_bytes_3c));
    scale_data_delta<<<grid, block>>>(width, height, data_s_mean, data_mean, 3);
    
    // Filter again, using patch-based filtering
    for (int dy = -wnd_rad; dy <= 0; ++ dy) {
        int dx_max = (dy == 0) ? -1 : +wnd_rad;
        for (int dx = -wnd_rad; dx <= dx_max; ++ dx) {
            // initialize all weights to 1, and then update with actual weights
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n1, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n2, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n1_s, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n2_s, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            UpdateWeightsDelta(data_mean, data_mean_var_num, data_mean_var_den, var_num_scale, var_den_scale, ptc_rad, dx, dy, nchannels_guide);
            if (mode == FILTER_WITH_FEATURES) {
                for (size_t i = 0; i < features.size(); i++) {
//                    UpdateWeights(features[i], features_var_num[i], features_var_den[i],
//                         var_num_scale_feat, var_den_scale_feat, 0, 0, dx, dy, features_nchannels[i], true);
//                    keep_min_weights<<<grid, block>>>(width, height, wgt_n1_s, wgt_n1_tmp);
//                    keep_min_weights<<<grid, block>>>(width, height, wgt_n2_s, wgt_n2_tmp);
                    UpdateWeightsFeat(features[i], features_var_num[i], features_var_den[i],
                         var_num_scale_feat, var_den_scale_feat, 0, 0, dx, dy, features_nchannels[i], true);
                }
            }
            // apply the weights to the data
            apply_weights<<<grid, block>>>(width, height, data_filtered,
                    wgt_max, wgt_sum, wgt_n1, wgt_n2, data, dx, dy, nbuffers, nchannels_data);
            // apply the weights to the scaled data
            apply_weights<<<grid, block>>>(width, height, data_s_mean_filtered,
                    wgt_max_s, wgt_sum_s, wgt_n1_s, wgt_n2_s, data_mean, dx, dy, 1, nchannels_data);
        }
    }
    
    // add contribution of center pixel
    accumulate_scaled<<<grid, block>>>(width, height, wgt_sum, wgt_max, 1, 1.f);
    relax_data<<<grid, block>>>(width, height, data_filtered, data, wgt_max, 0, 0, nbuffers, nchannels_data);
    normalize<<<grid, block>>>(width, height, data_filtered, data_filtered, wgt_sum, nbuffers, nchannels_data);
    // add contribution of scaled data center pixel
    accumulate_scaled<<<grid, block>>>(width, height, wgt_sum_s, wgt_max_s, 1, 1.f);
    relax_data<<<grid, block>>>(width, height, data_s_mean_filtered, data_s_mean, wgt_max_s, 0, 0, 1, nchannels_data);
    normalize<<<grid, block>>>(width, height, data_s_mean_filtered, data_s_mean_filtered, wgt_sum_s, 1, nchannels_data);
}


void FeatureFilter::FilterDataDeltaSpp(int wnd_rad, int ptc_rad,
        float var_num_scale, float var_den_scale, int nchannels_data,
        int nchannels_guide, FeatureFilterMode mode, float var_num_scale_feat,
        float var_den_scale_feat) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    // Initialize accumulators
    CUDA_CHECK_RETURN(cudaMemset(data_filtered, 0, nbuffers * size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMemset(wgt_sum, 0, size_img_bytes_1c));
    set_val<<<grid, block>>>(width, height, wgt_max, WEIGHT_EPSILON, 1);
    CUDA_CHECK_RETURN(cudaMemset(data_s_mean_filtered, 0, size_img_bytes_3c));
    CUDA_CHECK_RETURN(cudaMemset(wgt_sum_s, 0, size_img_bytes_1c));
    set_val<<<grid, block>>>(width, height, wgt_max_s, WEIGHT_EPSILON, 1);
    CUDA_CHECK_RETURN(cudaMemset(spp_filtered, 0, size_img_bytes_1c));
    
    // get the mean of the scaled data
    CUDA_CHECK_RETURN(cudaMemset(data_s_mean, 0, size_img_bytes_3c));
    scale_data_delta<<<grid, block>>>(width, height, data_s_mean, data_mean, 3);
    
    // Filter again, using patch-based filtering
    for (int dy = -wnd_rad; dy <= 0; ++ dy) {
        int dx_max = (dy == 0) ? -1 : +wnd_rad;
        for (int dx = -wnd_rad; dx <= dx_max; ++ dx) {
            // initialize all weights to 1, and then update with actual weights
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n1, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n2, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n1_s, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(wgt_n2_s, ones, size_img_bytes_1c, cudaMemcpyDeviceToDevice));
            UpdateWeightsDelta(data_mean, data_mean_var_num, data_mean_var_den, var_num_scale, var_den_scale, ptc_rad, dx, dy, nchannels_guide);
            if (mode == FILTER_WITH_FEATURES) {
                for (size_t i = 0; i < features.size(); i++) {
                    UpdateWeightsFeat(features[i], features_var_num[i], features_var_den[i],
                         var_num_scale_feat, var_den_scale_feat, 0, 0, dx, dy, features_nchannels[i], true);
                }
            }
            // apply the weights to the data
            accumulate_scaled<<<grid, block>>>(width, height, wgt_sum, wgt_n1, 1, 1.f);
            accumulate_scaled<<<grid, block>>>(width, height, wgt_sum, wgt_n2, 1, 1.f);
            relax_data<<<grid, block>>>(width, height, data_filtered, data, wgt_n1, +dx, +dy, nbuffers, nchannels_data);
            relax_data<<<grid, block>>>(width, height, data_filtered, data, wgt_n2, -dx, -dy, nbuffers, nchannels_data);
            keep_max_weights<<<grid, block>>>(width, height, wgt_max, wgt_n1);
            keep_max_weights<<<grid, block>>>(width, height, wgt_max, wgt_n2);
            // apply the weights to the scaled data
            accumulate_scaled<<<grid, block>>>(width, height, wgt_sum_s, wgt_n1_s, 1, 1.f);
            accumulate_scaled<<<grid, block>>>(width, height, wgt_sum_s, wgt_n2_s, 1, 1.f);
            relax_data<<<grid, block>>>(width, height, data_s_mean_filtered, data_mean, wgt_n1_s, +dx, +dy, 1, nchannels_data);
            relax_data<<<grid, block>>>(width, height, data_s_mean_filtered, data_mean, wgt_n2_s, -dx, -dy, 1, nchannels_data);
            keep_max_weights<<<grid, block>>>(width, height, wgt_max_s, wgt_n1_s);
            keep_max_weights<<<grid, block>>>(width, height, wgt_max_s, wgt_n2_s);
            // apply the weights to the spp
            relax_data<<<grid, block>>>(width, height, spp_filtered, conf, wgt_n1, +dx, +dy, 1, 1);
            relax_data<<<grid, block>>>(width, height, spp_filtered, conf, wgt_n2, -dx, -dy, 1, 1);
        }
    }
    
    // add contribution of center pixel
    accumulate_scaled<<<grid, block>>>(width, height, wgt_sum, wgt_max, 1, 1.f);
    relax_data<<<grid, block>>>(width, height, data_filtered, data, wgt_max, 0, 0, nbuffers, nchannels_data);
    normalize<<<grid, block>>>(width, height, data_filtered, data_filtered, wgt_sum, nbuffers, nchannels_data);
    // add contribution of scaled data center pixel
    accumulate_scaled<<<grid, block>>>(width, height, wgt_sum_s, wgt_max_s, 1, 1.f);
    relax_data<<<grid, block>>>(width, height, data_s_mean_filtered, data_s_mean, wgt_max_s, 0, 0, 1, nchannels_data);
    normalize<<<grid, block>>>(width, height, data_s_mean_filtered, data_s_mean_filtered, wgt_sum_s, 1, nchannels_data);
    // add contribution of center pixel to the filtered spp
    relax_data<<<grid, block>>>(width, height, spp_filtered, conf, wgt_max, 0, 0, 1, 1);
}


void FeatureFilter::GetPixelWeights(int x, int y, Buffer &weights) {
    int nweights = (2*wnd_rad+1) * (2*wnd_rad+1);
    weights.resize(nweights, 0.f);
    WeightsData(&weights[0], x, y, wnd_rad, ptc_rad, max(0, ptc_rad-1), var_num_scale,
        var_den_scale, nchannels_data, nchannels_guide, mode,
        var_num_scale_feat, var_den_scale_feat);
}


void FeatureFilter::GetFilteredData(Buffer &output) {
    GetDataMean(data_filtered_mean, data_filtered);
    CUDA_CHECK_RETURN(cudaMemcpy(&(output[0]), data_filtered_mean, nchannels_data * size_img_bytes_1c, cudaMemcpyDeviceToHost));
}


void FeatureFilter::GetFilteredDataBuffers(BufferSet &output) {
    output.resize(nbuffers);
    int offset = 0;
    for (int i = 0; i < nbuffers; i++) {
        output[i].resize(width*height*nchannels_data);
        CUDA_CHECK_RETURN(cudaMemcpy(&(output[i][0]), data_filtered + offset, nchannels_data * size_img_bytes_1c, cudaMemcpyDeviceToHost));
        offset += width * height * nchannels_data;
    }
}


void FeatureFilter::GetFilteredDataAndDerivative(Buffer &output, Buffer &deriv) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    ratio<<<grid, block>>>(width, height, tmp, wgt_max, wgt_sum, 1);
    CUDA_CHECK_RETURN(cudaMemcpy(&(img[0]), tmp, size_img_bytes_1c, cudaMemcpyDeviceToHost));
    
    GetFilteredData(output);
    if (feat_only) {
        for (int i = 0; i < width * height; i++) {
            deriv[3*i+0] = deriv[3*i+1] = deriv[3*i+2] = img[i];
        }
    }
    else {
        derivative<<<grid, block>>>(width, height, tmp, data_filtered_mean, data_s_mean_filtered, data_mean, data_s_mean, nchannels_data);
        size_t size_img_bytes = (nchannels_data == 1) ? size_img_bytes_1c : size_img_bytes_3c;
        CUDA_CHECK_RETURN(cudaMemcpy(&(deriv[0]), tmp, size_img_bytes, cudaMemcpyDeviceToHost));
    }
    
    if (nchannels_data == 1) {
        for (int i = 0; i < width * height; i++) {
            if (deriv[i] < 0.f || deriv[i] > 1.f) {
                deriv[i] = img[i];
            }
        }
    }
    else {
        for (int i = 0; i < width * height; i++) {
            int r = 3*i+0, g = 3*i+1, b = 3*i+2;
            bool invalid = deriv[r] < 0.f || deriv[r] > 1.f || deriv[g] < 0.f || deriv[g] > 1.f || deriv[b] < 0.f || deriv[b] > 1.f;
            if (invalid) {
                deriv[r] = deriv[g] = deriv[b] = img[i];
            }
        }
    }
}


void FeatureFilter::GetFilteredDataAndDerivative(Buffer &output, Buffer &deriv, Buffer &output_spp) {
    GetFilteredDataAndDerivative(output, deriv);
    CUDA_CHECK_RETURN(cudaMemcpy(&(output_spp[0]), spp_filtered, size_img_bytes_1c, cudaMemcpyDeviceToHost));
}


void FeatureFilter::GetDataMean(float* dst, const float *src) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    int offset = 0;
    CUDA_CHECK_RETURN(cudaMemset(dst, 0, nchannels_data * size_img_bytes_1c));
    for (int i = 0; i < nbuffers; i++) {
        accumulate_scaled<<<grid, block>>>(width, height, dst, src+offset, nchannels_data, 1.f / nbuffers);
        offset += width * height * nchannels_data;
    }
}


void FeatureFilter::GetScaledSampleVariance(float* buf_var, const float *src, float *smp_var, int nchannels) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    int offset = 0;
    CUDA_CHECK_RETURN(cudaMemset(buf_var, 0, nchannels * size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMemset(tmp, 0, nchannels * size_img_bytes_1c));
    for (int i = 0; i < nbuffers; i++) {
        update_online_variance<<<grid, block>>>(width, height, tmp, buf_var, src+offset, nchannels, i+1);
        offset += width * height * nchannels;
    }
    scale_data<<<grid, block>>>(width, height, buf_var, 1.f / (nbuffers * (nbuffers-1)), nchannels);
    
    // filter both the buffer and sample variance
    conv_box<<<grid, block>>>(width, height, tmp, buf_var, 10, 1, 3);
    conv_box<<<grid, block>>>(width, height, buf_var, tmp, 10, width, 3);
    conv_box<<<grid, block>>>(width, height, tmp, smp_var, 10, 1, 3);
    conv_box<<<grid, block>>>(width, height, tmp4, tmp, 10, width, 3);
    
    scale_sample_variance<<<grid, block>>>(width, height, buf_var, buf_var, tmp4, smp_var, nchannels);
}


void FeatureFilter::GetDataMeanVariance(float* dst, const float *src, float k) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    int offset = 0;
    CUDA_CHECK_RETURN(cudaMemset(dst, 0, nchannels_data * size_img_bytes_1c));
    CUDA_CHECK_RETURN(cudaMemset(tmp, 0, nchannels_data * size_img_bytes_1c));
    for (int i = 0; i < nbuffers; i++) {
        update_online_variance<<<grid, block>>>(width, height, tmp, dst, src+offset, nchannels_data, i+1);
        offset += width * height * nchannels_data;
    }
    scale_data<<<grid, block>>>(width, height, dst, 1.f / (nbuffers * (nbuffers-1)), nchannels_data);
    conv_pyr<<<grid, block>>>(width, height, tmp, dst, 1, nchannels_data, k);
    conv_pyr<<<grid, block>>>(width, height, dst, tmp, width, nchannels_data, k);
}


void FeatureFilter::UpdateWeights(
    const float *data_mean,
    const float *data_mean_var_num,
    const float *data_mean_var_den,
    float var_num_scale,
    float var_den_scale,
    int ptc_rad,                // patch radius for distance filtering
    int ptc_rad_wgt,            // patch radius for weight filtering
    int dx, int dy,             // offset to first neighbor
    int nchannels,
    bool use_diff_var
) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    distance<<<grid, block>>>(width, height, d2_n1, d2_n2, d2_ns, data_mean, data_mean_var_num, data_mean_var_den, var_num_scale, var_den_scale, dx, dy, nchannels, use_diff_var);
    if (ptc_rad > 0) {
        conv_box_all<<<grid, block>>>(width, height, tmp_n1, d2_n1, tmp_n2, d2_n2, tmp_ns, d2_ns, ptc_rad, 1, 1);
        conv_box_all<<<grid, block>>>(width, height, d2_n1, tmp_n1, d2_n2, tmp_n2, d2_ns, tmp_ns, ptc_rad, width, 1);
    }
    
    // compute all weights
    get_weights<<<grid, block>>>(width, height, wgt_n1_tmp, wgt_n2_tmp, wgt_ns_tmp, d2_n1, d2_n2, d2_ns);
    if (ptc_rad_wgt > 0) {
        conv_box_all<<<grid, block>>>(width, height, tmp_n1, wgt_n1_tmp, tmp_n2, wgt_n2_tmp, tmp_ns, wgt_ns_tmp, ptc_rad_wgt, 1, 1);
        conv_box_all<<<grid, block>>>(width, height, wgt_n1_tmp, tmp_n1, wgt_n2_tmp, tmp_n2, wgt_ns_tmp, tmp_ns, ptc_rad_wgt, width, 1);
    }
    
    // use symmetric weights if we can
    keep_valid_sym_weights<<<grid, block>>>(width, height, wgt_n1_tmp, wgt_n2_tmp, wgt_ns_tmp, d2_n1, d2_n2, MAX_D2);
    // update the current weights
    keep_min_weights<<<grid, block>>>(width, height, wgt_n1, wgt_n1_tmp);
    keep_min_weights<<<grid, block>>>(width, height, wgt_n2, wgt_n2_tmp);
}


void FeatureFilter::UpdateWeightsFeat(
    const float *data_mean,
    const float *data_mean_var_num,
    const float *data_mean_var_den,
    float var_num_scale,
    float var_den_scale,
    int ptc_rad,                // patch radius for distance filtering
    int ptc_rad_wgt,            // patch radius for weight filtering
    int dx, int dy,             // offset to first neighbor
    int nchannels,
    bool use_diff_var
) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    dist_wgt_feat<<<grid, block>>>(width, height, wgt_n1, wgt_n2, wgt_n1_s, wgt_n2_s,
        data_mean, data_mean_var_num, data_mean_var_den,
        var_num_scale, var_den_scale, dx, dy, nchannels, use_diff_var, MAX_D2);
}


void FeatureFilter::UpdateWeightsDelta(
    const float *data_mean,
    const float *data_mean_var_num,
    const float *data_mean_var_den,
    float var_num_scale,
    float var_den_scale,
    int ptc_rad,                // patch radius for distance filtering
    int dx, int dy,             // offset to first neighbor
    int nchannels
) {
    dim3 block(BLOCK_Y, BLOCK_X, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    distance_delta<<<grid, block>>>(width, height,
        d2_n1_b, d2_n2_b, d2_ns_b, d2_n1_s, d2_n2_s, d2_ns_s, 
        data_mean, data_mean_var_num, data_mean_var_den, var_num_scale, var_den_scale, dx, dy, nchannels, true);
    if (ptc_rad > 0) {
        conv_box_all<<<grid, block>>>(width, height, tmp_n1, d2_n1_b, tmp_n2, d2_n2_b, tmp_ns, d2_ns_b, ptc_rad, 1, 1);
        conv_box_all<<<grid, block>>>(width, height, d2_n1, tmp_n1, d2_n2, tmp_n2, d2_ns, tmp_ns, ptc_rad, width, 1);
    }
    
//    // update both standard and delta weights
//    float wgt_center = 1.f / (1+2*ptc_rad) / (1+2*ptc_rad);
//    update_delta_dist<<<grid, block>>>(width, height, d2_n1_s, d2_n2_s, d2_ns_s,
//        d2_n1_b, d2_n2_b, d2_ns_b, d2_n1, d2_n2, d2_ns, wgt_center, false);//abs(dx) <= ptc_rad && abs(dy) <= ptc_rad );
//    
//    // compute all weights
//    get_weights<<<grid, block>>>(width, height, wgt_n1_tmp,   wgt_n2_tmp,   wgt_ns_tmp,   d2_n1,   d2_n2,   d2_ns);
//    get_weights<<<grid, block>>>(width, height, wgt_n1_tmp_s, wgt_n2_tmp_s, wgt_ns_tmp_s, d2_n1_s, d2_n2_s, d2_ns_s);
//    
//    // use symmetric weights if we can
//    keep_valid_sym_weights<<<grid, block>>>(width, height, wgt_n1_tmp,   wgt_n2_tmp,   wgt_ns_tmp,   d2_n1,   d2_n2, MAX_D2);
//    keep_valid_sym_weights<<<grid, block>>>(width, height, wgt_n1_tmp_s, wgt_n2_tmp_s, wgt_ns_tmp_s, d2_n1_s, d2_n2_s, MAX_D2);
//    // update the current weights
//    keep_min_weights<<<grid, block>>>(width, height, wgt_n1,   wgt_n1_tmp);
//    keep_min_weights<<<grid, block>>>(width, height, wgt_n2,   wgt_n2_tmp);
//    keep_min_weights<<<grid, block>>>(width, height, wgt_n1_s, wgt_n1_tmp_s);
//    keep_min_weights<<<grid, block>>>(width, height, wgt_n2_s, wgt_n2_tmp_s);
    
    // update both standard and delta weights
    float wgt_center = 1.f / (1+2*ptc_rad) / (1+2*ptc_rad);
    update_weights_delta<<<grid, block>>>(width, height,
        wgt_n1, wgt_n2, wgt_n1_s, wgt_n2_s,
        d2_n1, d2_n2, d2_ns, d2_n1_b, d2_n2_b, d2_ns_b, d2_n1_s, d2_n2_s, d2_ns_s,
        dx, dy, wgt_center, false, MAX_D2); //abs(dx) <= ptc_rad && abs(dy) <= ptc_rad );
}


void FeatureFilter::DumpMap(Buffer &data, const string &tag, int buf) {
    int npixels = width * height;
    Buffer tmp(3*npixels);
    for (int i = 0; i < npixels; i++) {
        tmp[3*i+0] = tmp[3*i+1] = tmp[3*i+2] = data[i];
    }
    DumpImageRGB(tmp, tag, buf);
}


void FeatureFilter::DumpImageRGB(vector<float> &data, const string &tag, int buf) {
    // Generate output filename
    char name[256];
    if (buf >= 0) {
        sprintf(name, "%s_b%d.exr", tag.c_str(), buf);
    }
    else {
        sprintf(name, "%s.exr", tag.c_str());
    }

    // Write to disk
//    ::WriteImage(name, &data[0], NULL, width, height, width, height, 0, 0);
}


void FeatureFilter::GetDataMeanVar(Buffer &mean_var) {
    mean_var.resize(nchannels_data*width*height);
    CUDA_CHECK_RETURN(cudaMemcpy(&(mean_var[0]), data_mean_var_num, nchannels_data * size_img_bytes_1c, cudaMemcpyDeviceToHost));    
}


