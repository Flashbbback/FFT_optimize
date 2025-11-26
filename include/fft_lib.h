// 对外接口：所有外部需要调用的函数声明
#ifndef FFT_LIB_H
#define FFT_LIB_H

#include <stdint.h>

typedef struct FFTContext FFTContext; // 不透明指针，隐藏内部成员

// 初始化与销毁
FFTContext* trig_table(int max_size);
void free_trig_table(FFTContext* ctx);

// 执行接口
void fft_AVX(float* real, float* imag, int N, FFTContext* ctx);
void fft_AVX_fixedP(int16_t* real, int16_t* imag, int N, FFTContext* ctx);
void fft_diedai(float* real, float* imag, int N, FFTContext* ctx);
#endif