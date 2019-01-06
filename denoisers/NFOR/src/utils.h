#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdio>
#include <cstdarg>

enum ESample
{
    IMAGE_X,
    IMAGE_Y,
    COLOR_R,
    COLOR_G,
    COLOR_B,
    NORMAL_X,
    NORMAL_Y,
    NORMAL_Z,
    TEXTURE_R,
    TEXTURE_G,
    TEXTURE_B,
    DEPTH,
    DIRECT_R,

    SAMPLE_SIZE
};


inline float Lerp(float t, float v1, float v2) {
    return (1.f - t) * v1 + t * v2;
}


inline float Clamp(float val, float low, float high) {
    if (val < low) return low;
    else if (val > high) return high;
    else return val;
}


inline int Clamp(int val, int low, int high) {
    if (val < low) return low;
    else if (val > high) return high;
    else return val;
}


inline int Mod(int a, int b) {
    int n = int(a/b);
    a -= n*b;
    if (a < 0) a += b;
    return a;
}


inline float Radians(float deg) {
    return ((float)M_PI/180.f) * deg;
}


inline float Degrees(float rad) {
    return (180.f/(float)M_PI) * rad;
}


inline float Log2(float x) {
    static float invLog2 = 1.f / logf(2.f);
    return logf(x) * invLog2;
}


inline int Floor2Int(float val);
inline int Log2Int(float v) {
    return Floor2Int(Log2(v));
}


inline bool IsPowerOf2(int v) {
    return (v & (v - 1)) == 0;
}


inline uint32_t RoundUpPow2(uint32_t v) {
    v--;
    v |= v >> 1;    v |= v >> 2;
    v |= v >> 4;    v |= v >> 8;
    v |= v >> 16;
    return v+1;
}


inline int Floor2Int(float val) {
    return (int)floorf(val);
}


inline int Round2Int(float val) {
    return Floor2Int(val + 0.5f);
}


inline int Float2Int(float val) {
    return (int)val;
}


inline int Ceil2Int(float val) {
    return (int)ceilf(val);
}

inline void Info(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    printf(fmt, args);
}

inline void Warning(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    printf(fmt, args);
}

inline void Error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    printf(fmt, args);
}

inline void Severe(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    printf(fmt, args);
}


#endif // UTILS_H
