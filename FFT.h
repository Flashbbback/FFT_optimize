#pragma once
#ifndef FFT_H
#define FFT_H
void fft_diedai(int32_t real[], int32_t imag[], int N);
void signal_gen(int32_t real[], int32_t imag[], int N);
void fft_digui(int32_t real[], int32_t imag[], int N);
void bit_reverse(int32_t real[], int32_t imag[], int N);
#endif // FFT_H