// 工具层：通用函数，如位反转 (Bit Reverse)
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "fft_internal.h"

static inline void bit_reverse(float real[], float imag[], int N,FFTContext* ctx)
{
    for (int i = 0; i < N; i++) {
        if (i < ctx->pos[i]) {
            float temp_real = real[i];
            float temp_imag = imag[i];
            real[i] = real[ctx->pos[i]];
            imag[i] = imag[ctx->pos[i]];
            real[ctx->pos[i]] = temp_real;
            imag[ctx->pos[i]] = temp_imag;

        }
    }

    return;
}

void fft_diedai(float *real,float *imag,int N, FFTContext* ctx)
{
    bit_reverse(real,imag,N,ctx);
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
                int idx = j * (ctx->size / M);
                WN_real = ctx->cos_table[idx];
                WN_imag = -ctx->sin_table[idx];
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
                // float temp_WN_real = WN_real;
                // WN_real = temp_WN_real * cosf(2*M_PI/M) + WN_imag * sinf(2*M_PI/M);  
                // WN_imag = WN_imag * cosf(2*M_PI/M) - temp_WN_real * sinf(2*M_PI/M);  
            }

        }
    }
return; 
}

void signal_gen(float real[], float imag[], int N) {
    for (int n = 0; n < N; n++) {
        real[n] = sinf(2 * M_PI * 10* n  / N)/N;
        imag[n] = 0.0f;
    }
}

// --- 2. 16位定点数（Q15）版本（内联+适配）---
// static inline 实现内联，减少函数调用开销
void signal_gen_q15(int16_t real[], int16_t imag[], int N) {
    for (int n = 0; n < N; n++) {
        // 步骤1：生成浮点数正弦信号（同浮点数版本）
        float sin_val = sinf(2 * M_PI * n * 10 / N)/N;
        // 步骤2：转换为Q15定点数（范围[-32768, 32767]，对应浮点[-1, 1)）
        // 注意：sin值范围是[-1,1]，需确保转换后不溢出（-1→-32768，1→32767）
        real[n] = FLOAT_TO_Q15(sin_val);
        imag[n] = 0;  // 虚部为0（定点数0）
    }
}