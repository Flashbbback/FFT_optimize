#pragma once
#ifndef FFT_H
#define FFT_H
void fft_diedai(float real[], float imag[], int N);
void signal_gen(float real[], float imag[], int N);
void fft_digui(float real[], float imag[], int N);
void bit_reverse(float real[], float imag[], int N);
#endif // FFT_H
