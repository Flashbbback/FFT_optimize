#pragma once
#ifndef FFT_H
#define FFT_H

typedef struct{
    float *cos_table;
    float *sin_table;
    int size;

} FFTContext;

FFTContext* trig_table(int max_size);
void free_trig_table(FFTContext* ctx);

void fft_diedai(float real[], float imag[], int N, FFTContext* ctx);
void signal_gen(float real[], float imag[], int N);
void fft_digui(float real[], float imag[], int N, FFTContext* ctx);
void bit_reverse(float real[], float imag[], int N);
#endif // FFT_H