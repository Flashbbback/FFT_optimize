#pragma once
#ifndef FFT_H
#define FFT_H
#include <stdint.h>

typedef struct{
    float *cos_table;
    float *sin_table;
    int16_t *cos_t;
    int16_t *sin_t;
    int* pos;
    int size;

} FFTContext;

FFTContext* trig_table(int max_size);
void free_trig_table(FFTContext* ctx);

void fft_diedai(float real[], float imag[], int N, FFTContext* ctx);
void signal_gen(float real[], float imag[], int N);
void signal_gen_q15(int16_t real[], int16_t imag[], int N);
void fft_digui(float real[], float imag[], int N, FFTContext* ctx);
// void bit_reverse(float real[], float imag[], int N, FFTContext* ctx);
// void bit_reverse_q15(int16_t real[], int16_t imag[], int N, FFTContext* ctx);
void fft_AVX(float real[], float imag[], int N, FFTContext* ctx);
void fft_AVX_fixedP(int16_t real[], int16_t imag[], int N, FFTContext *ctx);
#endif // FFT_H
