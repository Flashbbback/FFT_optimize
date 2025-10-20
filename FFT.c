#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <stdlib.h>  // 新增：用于malloc和free

#define M_PI 3.14159265358979323846


void signal_gen(float real[], float imag[], int N) {
    for (int n = 0; n < N; n++) {
        real[n] = sinf(2 * M_PI * n * 10 / N);
        imag[n] = 0.0f;
    }
}

void fft_digui(float real[], float imag[], int N) {
    if (N <= 1) return;
    // 堆上分配数组，避免栈溢出
    float* even_real = (float*)malloc(N/2 * sizeof(float));
    float* even_imag = (float*)malloc(N/2 * sizeof(float));
    float* odd_real = (float*)malloc(N/2 * sizeof(float));
    float* odd_imag = (float*)malloc(N/2 * sizeof(float));

    for (int k = 0; k < N/2; k++) {
        even_real[k] = real[2*k];
        even_imag[k] = imag[2*k];
        odd_real[k] = real[2*k+1];
        odd_imag[k] = imag[2*k+1];
    }
    fft_digui(even_real, even_imag, N/2);
    fft_digui(odd_real, odd_imag, N/2);

    float WN_real = 1.0f, WN_imag = 0.0f;
    for (int i = 0; i < N/2; i++) {
        real[i] = even_real[i] + odd_real[i]*WN_real - odd_imag[i]*WN_imag;
        imag[i] = even_imag[i] + odd_imag[i]*WN_real + odd_real[i]*WN_imag;
        real[i+N/2] = even_real[i] - (odd_real[i]*WN_real - odd_imag[i]*WN_imag);
        imag[i+N/2] = even_imag[i] - (odd_imag[i]*WN_real + odd_real[i]*WN_imag);

        float temp_real = WN_real;
        WN_real = WN_real * cosf(2*M_PI/N) - WN_imag * sinf(2*M_PI/N);  
        WN_imag = WN_imag * cosf(2*M_PI/N) + temp_real * sinf(2*M_PI/N);  
    }

    // 释放堆内存
    free(even_real);
    free(even_imag);
    free(odd_real);
    free(odd_imag);
}

void bit_reverse(float real[], float imag[], int N)
{
    // 堆上分配pos数组，避免栈溢出
    int* pos = (int*)malloc(N * sizeof(int));
    if (pos == NULL) {
        printf("Memory allocation failed in bit_reverse!\n");
        return; // 处理内存分配失败的情况
    }

    pos[0] = 0;
    for (int i = 1; i < N; i++) {
        pos[i] = pos[i / 2] / 2 + (i % 2) * N / 2;
    }
    for (int i = 0; i < N; i++) {
        if (i < pos[i]) {
            float temp_real = real[i];
            float temp_imag = imag[i];
            real[i] = real[pos[i]];
            imag[i] = imag[pos[i]];
            real[pos[i]] = temp_real;
            imag[pos[i]] = temp_imag;
        }
    }

    free(pos); // 释放堆内存
    return;
}

void fft_diedai(float real[],float imag[],int N)
{
    bit_reverse(real,imag,N);
    int m = log2(N);
    for (int s = 1;s<=m;s++)
    {
        int M = 1<<s;
        for(int k=0;k<N;k+=M)
        {
            float WN_real = 1.0f;
            float WN_imag = 0.0f;
            for (int j = 0;j<M/2;j++)
            {
                int t = k+j;
                int u = k+j + M/2;
                float temp_real1 = real[t] + real[u]*WN_real - imag[u]*WN_imag;
                float temp_imag1 = imag[t] + imag[u]*WN_real + real[u]*WN_imag;
                float temp_real2 = real[t] - (real[u]*WN_real - imag[u]*WN_imag);
                float temp_imag2 = imag[t] - (imag[u]*WN_real + real[u]*WN_imag);
                real[t] = temp_real1;
                imag[t] = temp_imag1;   
                real[u] = temp_real2;
                imag[u] = temp_imag2;
                float temp_WN_real = WN_real;
                WN_real = temp_WN_real * cosf(2*M_PI/M) + WN_imag * sinf(2*M_PI/M);  
                WN_imag = WN_imag * cosf(2*M_PI/M) - temp_WN_real * sinf(2*M_PI/M);  
            }

        }
    }
return; 
}