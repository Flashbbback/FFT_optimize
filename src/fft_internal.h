// 内部头文件，仅供src目录下的代码使用
#ifndef FFT_INTERNAL_H
#define FFT_INTERNAL_H

#include <immintrin.h>
#include <math.h>
#include "../include/fft_lib.h"

// 结构体具体定义放在这里，而不是 .c 文件里
struct FFTContext{
    float *cos_table;
    float *sin_table;
    float *shuffled_cos_table;
    float *shuffled_sin_table;
    int16_t *cos_t;
    int16_t *sin_t;
    int16_t *shuffled_cos_t;
    int16_t *shuffled_sin_t;
    int *stage_offsets;
    int* pos;
    int size;

} ;

// 通用宏
#define M_PI 3.14159265358979323846
#define SQRT2_2 0.70710678118654752440f
#define Q 15                  // Q15格式：15位小数位
#define SCALE (1 << Q)        // 缩放因子：32768
#define INT16_MAX 32767       // int16_t最大值
#define INT16_MIN (-32768)    // int16_t最小值

#define FLOAT_TO_Q15(x) ({ \
    float _temp = (x) * SCALE; \
    _temp += (x >= 0 ? 0.5f : -0.5f); \
    if (_temp > INT16_MAX) _temp = INT16_MAX; \
    if (_temp < INT16_MIN) _temp = INT16_MIN; \
    (int16_t)_temp; \
})

#define Q15_TO_FLOAT(x) ((float)(x) / SCALE)
// ... 其他宏 ...



#endif