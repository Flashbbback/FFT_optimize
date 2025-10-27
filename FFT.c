#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <stdlib.h>  // 新增：用于malloc和free
#include <immintrin.h>
#include "FFT.h"

#define M_PI 3.14159265358979323846



FFTContext* trig_table(int max_size)
{
    FFTContext* ctx = (FFTContext*)malloc(sizeof(FFTContext));
    ctx->size = max_size;
    ctx->cos_table = (float*)malloc((max_size/2) * sizeof(float));
    ctx->sin_table = (float*)malloc((max_size/2) * sizeof(float));
    for (int i = 0; i < max_size/2; i++) {
        ctx->cos_table[i] = cosf(2 * M_PI * i / max_size);
        ctx->sin_table[i] = sinf(2 * M_PI * i / max_size);
    }
    return ctx;
}

void free_trig_table(FFTContext* ctx) {
    if (ctx) {
        free(ctx->cos_table);
        free(ctx->sin_table);
        free(ctx);
    }
}

void signal_gen(float real[], float imag[], int N) {
    for (int n = 0; n < N; n++) {
        real[n] = sinf(2 * M_PI * n * 10 / N);
        imag[n] = 0.0f;
    }
}

void fft_digui(float real[], float imag[], int N, FFTContext* ctx) {
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
    fft_digui(even_real, even_imag, N/2,ctx);
    fft_digui(odd_real, odd_imag, N/2,ctx);

    float WN_real = 1.0f, WN_imag = 0.0f;
    for (int i = 0; i < N/2; i++) {
        int idx = i * (ctx->size / N);
        WN_real = ctx->cos_table[idx];
        WN_imag = -ctx->sin_table[idx];
        real[i] = even_real[i] + odd_real[i]*WN_real - odd_imag[i]*WN_imag;
        imag[i] = even_imag[i] + odd_imag[i]*WN_real + odd_real[i]*WN_imag;
        real[i+N/2] = even_real[i] - (odd_real[i]*WN_real - odd_imag[i]*WN_imag);
        imag[i+N/2] = even_imag[i] - (odd_imag[i]*WN_real + odd_real[i]*WN_imag);

        // float temp_real = WN_real;
        // WN_real = WN_real * cosf(2*M_PI/N) - WN_imag * sinf(2*M_PI/N);  
        // WN_imag = WN_imag * cosf(2*M_PI/N) + temp_real * sinf(2*M_PI/N);  
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

void fft_diedai(float real[],float imag[],int N, FFTContext* ctx)
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




void fft_AVX(float real[], float imag[], int N, FFTContext *ctx)
{
    bit_reverse(real, imag, N);
     __m256 *real_vec = (__m256 *)real;
     __m256 *imag_vec = (__m256 *)imag;
    int m = log2(N);
    for (int s = 1;s<=m;s++)
    {
        int M = 1<<s;
        //M = 2时 cos = 1,sin = 0;
        if (M == 2)
        {
            for (int j = 0; j < N/8; j++)
            {
                __m256 real_val = real_vec[j];
                __m256 imag_val = imag_vec[j];
                
                // 提取偶数和奇数元素
                __m256 even_real = _mm256_shuffle_ps(real_val, real_val, _MM_SHUFFLE(2,0,2,0));
                __m256 odd_real = _mm256_shuffle_ps(real_val, real_val, _MM_SHUFFLE(3,1,3,1));
                __m256 even_imag = _mm256_shuffle_ps(imag_val, imag_val, _MM_SHUFFLE(2,0,2,0));
                __m256 odd_imag = _mm256_shuffle_ps(imag_val, imag_val, _MM_SHUFFLE(3,1,3,1));
                
                // 蝶形运算
                __m256 result_real = _mm256_add_ps(even_real, odd_real);
                __m256 result_imag = _mm256_add_ps(even_imag, odd_imag);
                __m256 result_real2 = _mm256_sub_ps(even_real, odd_real);
                __m256 result_imag2 = _mm256_sub_ps(even_imag, odd_imag);
                
                // 交错存储结果
                real_vec[j] = _mm256_unpacklo_ps(result_real, result_real2);
                imag_vec[j] = _mm256_unpacklo_ps(result_imag, result_imag2);
            }
        }

        else if(M == 4)
        {
            __m256 w_real = _mm256_set_ps(0,-1,0,1,0,-1,0,1);
            __m256 w_imag = _mm256_set_ps(1,0,-1,0,1,0,-1,0);
            
            for(int j = 0;j<N/8;j++)
            {
                __m256 real_val = real_vec[j];
                __m256 imag_val = imag_vec[j];

                __m256 a_real = _mm256_shuffle_ps(real_val, real_val, _MM_SHUFFLE(1,0,1,0));
                __m256 b_real = _mm256_shuffle_ps(real_val, real_val, _MM_SHUFFLE(3,2,3,2));
                __m256 a_imag = _mm256_shuffle_ps(imag_val, imag_val, _MM_SHUFFLE(1,0,1,0));
                __m256 b_imag = _mm256_shuffle_ps(imag_val, imag_val, _MM_SHUFFLE(3,2,3,2));

                __m256 r_real = _mm256_sub_ps(_mm256_mul_ps(b_real,w_real),_mm256_mul_ps(b_imag,w_imag));//可用FMA优化 
                __m256 i_imag = _mm256_add_ps(_mm256_mul_ps(b_real,w_imag),_mm256_mul_ps(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm256_add_ps(a_real,r_real);
                imag_vec[j] = _mm256_add_ps(a_imag,i_imag);
            }



        }

        else if(M == 8)
        {
            __m256 w_real = _mm256_set_ps(-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0));//低位正，高位负
            __m256 w_imag = _mm256_set_ps(-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0));
            for(int j = 0;j<N/8;j++)
            {
                __m256 real_val = real_vec[j];
                __m256 imag_val = imag_vec[j];

                __m256 a_real = _mm256_permute2f128_ps(real_val,real_val,0x00);
                __m256 b_real = _mm256_permute2f128_ps(real_val,real_val,0x11);
                __m256 a_imag = _mm256_permute2f128_ps(imag_val,imag_val,0x00);
                __m256 b_imag = _mm256_permute2f128_ps(imag_val,imag_val,0x11);

                __m256 r_real = _mm256_sub_ps(_mm256_mul_ps(b_real,w_real),_mm256_mul_ps(b_imag,w_imag));//可用FMA优化 
                __m256 i_imag = _mm256_add_ps(_mm256_mul_ps(b_real,w_imag),_mm256_mul_ps(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm256_add_ps(a_real,r_real);
                imag_vec[j] = _mm256_add_ps(a_imag,i_imag);



            }



        }

        else
        {

            for(int k = 0;k<N/8;k+=M/8)
            {

                for(int j = 0;j<M/2/8;j++)
                {
                    int idx = j * (ctx->size / M);

                    __m256 w_real = _mm256_set_ps(ctx->cos_table[idx+7],ctx->cos_table[idx+6],ctx->cos_table[idx+5],ctx->cos_table[idx+4],ctx->cos_table[idx+3],ctx->cos_table[idx+2],ctx->cos_table[idx+1],ctx->cos_table[idx]);
                    __m256 w_imag = _mm256_set_ps(-ctx->sin_table[idx+7],-ctx->sin_table[idx+6],-ctx->sin_table[idx+5],-ctx->sin_table[idx+4],-ctx->sin_table[idx+3],-ctx->sin_table[idx+2],-ctx->sin_table[idx+1],-ctx->sin_table[idx]);


                    __m256 real_val1 = real_vec[k+j];
                    __m256 imag_val1 = imag_vec[k+j];
                    __m256 real_val2 = real_vec[k+j+M/2/8];
                    __m256 imag_val2 = imag_vec[k+j+M/2/8];


                    __m256 r_real = _mm256_sub_ps(_mm256_mul_ps(real_val2,w_real),_mm256_mul_ps(imag_val2,w_imag));//可用FMA优化 
                    __m256 i_imag = _mm256_add_ps(_mm256_mul_ps(real_val2,w_imag),_mm256_mul_ps(imag_val2,w_real));//可用FMA优化

                    real_vec[k+j] = _mm256_add_ps(real_val1,r_real);
                    imag_vec[k+j] = _mm256_add_ps(imag_val1,i_imag);
                    real_vec[k+j+M/2/8] = _mm256_sub_ps(real_val1,r_real);
                    imag_vec[k+j+M/2/8] = _mm256_sub_ps(imag_val1,i_imag);



                }
            }






        }



    }
}



