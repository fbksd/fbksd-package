
/*
    SBF source code Copyright(c) 2012-2013 Tzu-Mao Li

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


#if defined(_MSC_VER)
#pragma once
#endif

#ifndef SBF_VECTOR_NF_H__
#define SBF_VECTOR_NF_H__

#include <iostream>
#include <cassert>
#include <limits>
#include <algorithm>

using namespace std;

/**
 *  @ VectorNf: a N-d float vector with mathematical operators
 */
template<int N>
class VectorNf {
public:
    VectorNf() {
        for(int i = 0; i < N; i++)
            data[i] = 0.f;
    }

    VectorNf(float v) {
        for(int i = 0; i < N; i++)
            data[i] = v;
    }

    VectorNf(const float v[N]) {       
        for(int i = 0; i < N; i++)
            data[i] = v[i];
    }

    int Size() const {return N;}

    VectorNf operator+(const VectorNf &vn) const {
        VectorNf result;
        for(int i = 0; i < N; i++)
            result[i] = data[i] + vn[i];
        return result;
    }

    VectorNf& operator+=(const VectorNf &vn) {
        for(int i = 0; i < N; i++)
            data[i] += vn[i];
        return *this;
    }

    VectorNf operator+(float v) const {
        VectorNf<N> result;
        for(int i = 0; i < N; i++)
            result[i] = data[i] + v;
        return result;
    }

    VectorNf& operator+=(float v) {
        for(int i = 0; i < N; i++)
            data[i] += v;
        return *this;
    }

    VectorNf operator-() const {
        VectorNf result;
        for(int i = 0; i < N; i++)
            result[i] = -data[i];
        return result;
    }

    VectorNf operator-(const VectorNf &vn) const {
        VectorNf result;
        for(int i = 0; i < N; i++)
            result[i] = data[i] - vn[i];
        return result;
    }    

    VectorNf& operator-=(const VectorNf &vn) {
        for(int i = 0; i < N; i++)
            data[i] -= vn[i];
        return *this;
    }

    VectorNf operator-(float v) const {
        VectorNf result;
        for(int i = 0; i < N; i++)
            result[i] = data[i] - v;
        return result;
    }

    VectorNf& operator-=(float v) {
        for(int i = 0; i < N; i++)
            data[i] -= v;
        return *this;
    }

    VectorNf operator*(const VectorNf &vn) const {
        VectorNf result;
        for(int i = 0; i < N; i++)
            result[i] = data[i] * vn[i];
        return result;
    }

    VectorNf& operator*=(const VectorNf &vn) {
        for(int i = 0; i < N; i++)
            data[i] *= vn[i];
        return *this;
    }

    VectorNf operator*(float v) const {
        VectorNf result;
        for(int i = 0; i < N; i++)
            result[i] = data[i] * v;
        return result;
    }

    VectorNf& operator*=(float v) {
        for(int i = 0; i < N; i++)
            data[i] *= v;
        return *this;
    }

    VectorNf operator/(const VectorNf &vn) const {
        VectorNf result;
        for(int i = 0; i < N; i++)
            result[i] = data[i] / vn[i];
        return result;
    }

    VectorNf& operator/=(const VectorNf &vn) {
        for(int i = 0; i < N; i++)
            data[i] /= vn[i];
        return *this;
    }

    VectorNf operator/(float v) const {
        float invV = 1.f/v;
        VectorNf result;
        for(int i = 0; i < N; i++)
            result[i] = data[i] * invV;
        return result;
    }

    VectorNf& operator/=(float v) {
        float invV = 1.f/v;
        for(int i = 0; i < N; i++)
            data[i] *= invV;
        return *this;
    }

    float operator[](int i) const {
        assert(i >= 0 && i < N);
        return data[i];
    }    

    float& operator[](int i) {
        assert(i >= 0 && i < N);
        return data[i];
    }

    bool HasNaNs() const {
        for(int i = 0; i < N; i++)
            if(data[i] == numeric_limits<float>::quiet_NaN())
                return true;
        return false;
    }

    float Sum() const {
        float sum = 0.f;
        for(int i = 0; i < N; i++)
            sum += data[i];
        return sum;
    }

    float Avg() const {
        return Sum()/(float)N;
    }

    VectorNf<N> Max(float val) const {
        VectorNf<N> result;
        for(int i = 0; i < N; i++)
            result[i] = max(data[i], val);
        return result;
    }

    VectorNf<N> Min(float val) {
        VectorNf<N> result;
        for(int i = 0; i < N; i++)
            result[i] = min(data[i], val);
        return result;
    }

    VectorNf<N> Clamp(float low, float up) {
        VectorNf<N> result;
        for(int i = 0; i < N; i++)
            result[i] = max(min(data[i], up), low);
        return result;
    }

    float *GetRawPtr() { return data; }
    const float *GetRawPtr() const { return data; }

protected:
    float data[N];
};

template<int N>
VectorNf<N> operator*(float f, const VectorNf<N> &c) {
    return c*f;
}

template<int N>
VectorNf<N> operator-(float f, const VectorNf<N> &c) {
    return -(c-f);
}

// Assume RGB color space
class Color : public VectorNf<3> {
public:
    Color() {
        data[0] = data[1] = data[2] = 0.f;
    }

    Color(const VectorNf<3> &v)
        : VectorNf<3>(v) {}

    Color(float v) {
        data[0] = data[1] = data[2] = v;
    }

    Color(float x, float y, float z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }

    Color(float c[3]) {
        data[0] = c[0];
        data[1] = c[1];
        data[2] = c[2];

    }

    float Y() const {
        // Only for RGB color        
        const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
        return YWeight[0] * data[0] + YWeight[1] * data[1] + YWeight[2] * data[2];
    }
};

inline float Sum(float v) {
    return v;
}

inline float Avg(float v) {
    return v;
}

template<int N>
float Sum(const VectorNf<N> &v) {
    return v.Sum();
}

template<int N>
float Avg(const VectorNf<N> &v) {
    return v.Avg();
}

#endif //#ifndef SBF_VECTOR_NF_H__
