
#include <math.h>
#include <windows.h>
#include <stdlib.h>  // 新增：用于malloc和free
#include <immintrin.h>
#include <stdio.h>
#include "FFT.h"
#include <stdint.h>

#define M_PI 3.14159265358979323846
#define Q 15                  // Q15格式：15位小数位
#define SCALE (1 << Q)        // 缩放因子：32768
#define INT16_MAX 32767       // int16_t最大值
#define INT16_MIN (-32768)    // int16_t最小值

// 修复后的宏：支持作为表达式赋值
#define FLOAT_TO_Q15(x) ({ \
    float _temp = (x) * SCALE; \
    _temp += (x >= 0 ? 0.5f : -0.5f); \
    if (_temp > INT16_MAX) _temp = INT16_MAX; \
    if (_temp < INT16_MIN) _temp = INT16_MIN; \
    (int16_t)_temp; \
})

#define Q15_TO_FLOAT(x) ((float)(x) / SCALE)


FFTContext* trig_table(int max_size)
{
    FFTContext* ctx = (FFTContext*)malloc(sizeof(FFTContext));
    ctx->size = max_size;
    ctx->cos_table = (float*)_aligned_malloc((max_size/2) * sizeof(float),32);
    ctx->sin_table = (float*)_aligned_malloc((max_size/2) * sizeof(float),32);
    ctx->cos_t = (int16_t*)_aligned_malloc((max_size/2) * sizeof(int16_t),32);
    ctx->sin_t = (int16_t*)_aligned_malloc((max_size/2) * sizeof(int16_t),32);
    ctx->pos = (int*)_aligned_malloc(max_size * sizeof(int),32);
    ctx->pos[0] = 0;
    for (int i = 1; i < max_size; i++) {
        ctx->pos[i] = ctx->pos[i / 2] / 2 + (i % 2) * max_size / 2;
    }
    for (int i = 0; i < max_size/2; i++) {
        ctx->cos_table[i] = cosf(2 * M_PI * i / max_size);
        ctx->sin_table[i] = sinf(2 * M_PI * i / max_size);
        ctx->cos_t[i] = FLOAT_TO_Q15(ctx->cos_table[i]);
        ctx->sin_t[i] = FLOAT_TO_Q15(ctx->sin_table[i]);

    }
    return ctx;
}

void free_trig_table(FFTContext* ctx) {
    if (ctx) {
        _aligned_free(ctx->cos_table);
        _aligned_free(ctx->sin_table);
        _aligned_free(ctx->cos_t);
        _aligned_free(ctx->sin_t);
        _aligned_free(ctx->pos);
        free(ctx);
    }
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

static void bit_reverse(float real[], float imag[], int N,FFTContext* ctx)
{
    // 堆上分配pos数组，避免栈溢出
    // int* pos = (int*)malloc(N * sizeof(int));
    // if (pos == NULL) {
    //     // printf("Memory allocation failed in bit_reverse!\n");
    //     return; // 处理内存分配失败的情况
    // }

    // pos[0] = 0;
    // for (int i = 1; i < N; i++) {
    //     pos[i] = pos[i / 2] / 2 + (i % 2) * N / 2;
    // }
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

static void bit_reverse_q15(int16_t real[], int16_t imag[], int N,FFTContext* ctx)
{
    // 堆上分配pos数组，避免栈溢出
    // int* pos = (int*)malloc(N * sizeof(int));
    // if (pos == NULL) {
    //     // printf("Memory allocation failed in bit_reverse!\n");
    //     return; // 处理内存分配失败的情况
    // }

    // pos[0] = 0;
    // for (int i = 1; i < N; i++) {
    //     pos[i] = pos[i / 2] / 2 + (i % 2) * N / 2;
    // }
    for (int i = 0; i < N; i++) {
        if (i < ctx->pos[i]) {
            int16_t temp_real = real[i];
            int16_t temp_imag = imag[i];
            real[i] = real[ctx->pos[i]];
            imag[i] = imag[ctx->pos[i]];
            real[ctx->pos[i]] = temp_real;
            imag[ctx->pos[i]] = temp_imag;

        }
    }

    return;
}

void fft_diedai(float real[],float imag[],int N, FFTContext* ctx)
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

void fft_AVX(float real[], float imag[], int N, FFTContext *ctx)
{
    bit_reverse(real, imag, N,ctx);
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
                    int step = (ctx->size / M);
                    int idx = j * 8 * step;

                    __m256 w_real = _mm256_set_ps(ctx->cos_table[idx+step*7],ctx->cos_table[idx+step*6],ctx->cos_table[idx+step*5],ctx->cos_table[idx+step*4],ctx->cos_table[idx+step*3],ctx->cos_table[idx+step*2],ctx->cos_table[idx+step*1],ctx->cos_table[idx]);
                    __m256 w_imag = _mm256_set_ps(-ctx->sin_table[idx+step*7],-ctx->sin_table[idx+step*6],-ctx->sin_table[idx+step*5],-ctx->sin_table[idx+step*4],-ctx->sin_table[idx+step*3],-ctx->sin_table[idx+step*2],-ctx->sin_table[idx+step*1],-ctx->sin_table[idx]);




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

void fft_AVX_fixedP(int16_t real[], int16_t imag[], int N, FFTContext *ctx)
{
    bit_reverse_q15(real, imag, N,ctx);
     __m256i *real_vec = (__m256i *)real;
     __m256i *imag_vec = (__m256i *)imag;
    int m = log2(N);
    for (int s = 1;s<=m;s++)
    {
        int M = 1<<s;
        //M = 2时 cos = 1,sin = 0;
        if (M == 2)
        {
            for (int j = 0; j < N/16; j++)
            {
                __m256i real_val = real_vec[j];
                __m256i imag_val = imag_vec[j];

                
                
                // // 提取偶数和奇数元素
                __m256i even_mask = _mm256_set_epi8(
                        13,12,9,8,5,4,1,0,13,12,9,8,5,4,1,0,
                        13,12,9,8,5,4,1,0,13,12,9,8,5,4,1,0
                );


                __m256i odd_mask = _mm256_set_epi8(
                        15,14,11,10,7,6,3,2,15,14,11,10,7,6,3,2,
                        15,14,11,10,7,6,3,2,15,14,11,10,7,6,3,2
                );

                // 0 2 4 6 0 2 4 6/8 10 12 14 8 10 12 14
                // 1 3 5 7 1 3 5 7/9 11 13 15 9 11 13 15 
                __m256i even_real = _mm256_shuffle_epi8(real_val,even_mask);
                __m256i even_imag = _mm256_shuffle_epi8(imag_val,even_mask);
                __m256i odd_real = _mm256_shuffle_epi8(real_val,odd_mask);
                __m256i odd_imag = _mm256_shuffle_epi8(imag_val,odd_mask);


                // __m256i even_real = _mm256_shuffle_epi32(_mm256_shufflelo_epi16(_mm256_shufflehi_epi16(real_val, _MM_SHUFFLE(2,0,2,0)),_MM_SHUFFLE(2,0,2,0)),_MM_SHUFFLE(2,0,2,0));
                // __m256i even_imag = _mm256_shuffle_epi32(_mm256_shufflelo_epi16(_mm256_shufflehi_epi16(imag_val, _MM_SHUFFLE(2,0,2,0)),_MM_SHUFFLE(2,0,2,0)),_MM_SHUFFLE(2,0,2,0));
                // __m256i odd_real = _mm256_shuffle_epi32(_mm256_shufflelo_epi16(_mm256_shufflehi_epi16(real_val, _MM_SHUFFLE(3,1,3,1)),_MM_SHUFFLE(3,1,3,1)),_MM_SHUFFLE(3,1,3,1));
                // __m256i odd_imag = _mm256_shuffle_epi32(_mm256_shufflelo_epi16(_mm256_shufflehi_epi16(imag_val, _MM_SHUFFLE(3,1,3,1)),_MM_SHUFFLE(3,1,3,1)),_MM_SHUFFLE(3,1,3,1));


                // __m256 even_real = _mm256_shuffle_ps(real_val, real_val, _MM_SHUFFLE(2,0,2,0));
                // __m256 odd_real = _mm256_shuffle_ps(real_val, real_val, _MM_SHUFFLE(3,1,3,1));
                // __m256 even_imag = _mm256_shuffle_ps(imag_val, imag_val, _MM_SHUFFLE(2,0,2,0));
                // __m256 odd_imag = _mm256_shuffle_ps(imag_val, imag_val, _MM_SHUFFLE(3,1,3,1));
                
                // 蝶形运算
                //0+1 2+3 4+5 6+7 0+1 2+3 4+5 6+7 8+9 10+11 12+13 14+15 8+9 10+11 12+13 14+15
                //0-1 2-3 4-5 6-7 0-1 2-3 4-5 6-7 8-9 10-11 12-13 14-15 8-9 10-11 12-13 14-15
                __m256i result_real = _mm256_add_epi16(even_real, odd_real);
                __m256i result_imag = _mm256_add_epi16(even_imag, odd_imag);
                __m256i result_real2 = _mm256_sub_epi16(even_real, odd_real);
                __m256i result_imag2 = _mm256_sub_epi16(even_imag, odd_imag);
                
                // 交错存储结果
                real_vec[j] = _mm256_unpacklo_epi16(result_real, result_real2);
                imag_vec[j] = _mm256_unpacklo_epi16(result_imag, result_imag2);
            }
        }

        else if(M == 4)
        {
            // __m256i w_real = _mm256_setr_epi16(1,0,-1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0);
            // __m256i w_imag = _mm256_setr_epi16(0,-1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0,1);

            int step = ctx->size / M;
            __m256i w_real = _mm256_set_epi16(ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0]);
            __m256i w_imag = _mm256_set_epi16(ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0]);


            // 提取对应元素
            //0 1 0 1  4 5 4 5 8 9 8 9  12 13 12 13
            __m256i a_mask = _mm256_setr_epi8(
                0,1,2,3,0,1,2,3,8,9,10,11,8,9,10,11,
                0,1,2,3,0,1,2,3,8,9,10,11,8,9,10,11
            );

            // 2 3 2 3 6 7 6 7 10 11 10 11 14 15 14 15
            __m256i b_mask = _mm256_setr_epi8(
                4,5,6,7,4,5,6,7,12,13,14,15,12,13,14,15,
                4,5,6,7,4,5,6,7,12,13,14,15,12,13,14,15
            );
            
            for(int j = 0;j<N/16;j++)
            {
                __m256i real_val = real_vec[j];
                __m256i imag_val = imag_vec[j];

                __m256i a_real = _mm256_shuffle_epi8(real_val,a_mask);
                __m256i a_imag = _mm256_shuffle_epi8(imag_val,a_mask);
                __m256i b_real = _mm256_shuffle_epi8(real_val,b_mask);
                __m256i b_imag = _mm256_shuffle_epi8(imag_val,b_mask);
                //0+2 1+3 0-2 1-3 4+6 5+7 4-6 5-7...
                __m256i r_real = _mm256_sub_epi16(_mm256_mulhrs_epi16(b_real,w_real),_mm256_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m256i i_imag = _mm256_add_epi16(_mm256_mulhrs_epi16(b_real,w_imag),_mm256_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm256_add_epi16(a_real,r_real);
                imag_vec[j] = _mm256_add_epi16(a_imag,i_imag);
            }



        }

        else if(M == 8)
        {
            // __m256i w_real = _mm256_set_epi16(-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0),-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0));//低位正，高位负
            // __m256i w_imag = _mm256_set_epi16(-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0),-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0));

            int step = ctx->size / M;
            __m256i w_real = _mm256_set_epi16(-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0]);
            __m256i w_imag = _mm256_set_epi16(ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0]);


            //0 1 2 3 0 1 2 3 2 9 10 11 8 9 10 11
            __m256i a_mask = _mm256_setr_epi8(
                0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,
                0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7
            );

            //4 5 6 7 4 5 6 7 12 13 14 15 12 13 14 15
            __m256i b_mask = _mm256_setr_epi8(
                8,9,10,11,12,13,14,15,8,9,10,11,12,13,14,15,
                8,9,10,11,12,13,14,15,8,9,10,11,12,13,14,15
            );
            for(int j = 0;j<N/16;j++)
            {
                __m256i real_val = real_vec[j];
                __m256i imag_val = imag_vec[j];

                __m256i a_real = _mm256_shuffle_epi8(real_val,a_mask);
                __m256i a_imag = _mm256_shuffle_epi8(imag_val,a_mask);
                __m256i b_real = _mm256_shuffle_epi8(real_val,b_mask);
                __m256i b_imag = _mm256_shuffle_epi8(imag_val,b_mask);

                __m256i r_real = _mm256_sub_epi16(_mm256_mulhrs_epi16(b_real,w_real),_mm256_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m256i i_imag = _mm256_add_epi16(_mm256_mulhrs_epi16(b_real,w_imag),_mm256_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm256_add_epi16(a_real,r_real);
                imag_vec[j] = _mm256_add_epi16(a_imag,i_imag);
            }



        }

        else if(M == 16)
        {

            int step = ctx->size/M;
            __m256i w_real = _mm256_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/M*7)),FLOAT_TO_Q15(-cosf(2*M_PI/M*6)),FLOAT_TO_Q15(-cosf(2*M_PI/M*5)),FLOAT_TO_Q15(-cosf(2*M_PI/M*4)),FLOAT_TO_Q15(-cosf(2*M_PI/M*3)),FLOAT_TO_Q15(-cosf(2*M_PI/M*2)),FLOAT_TO_Q15(-cosf(2*M_PI/M*1)),FLOAT_TO_Q15(-cosf(2*M_PI/M*0)),FLOAT_TO_Q15(cosf(2*M_PI/M*7)),FLOAT_TO_Q15(cosf(2*M_PI/M*6)),FLOAT_TO_Q15(cosf(2*M_PI/M*5)),FLOAT_TO_Q15(cosf(2*M_PI/M*4)),FLOAT_TO_Q15(cosf(2*M_PI/M*3)),FLOAT_TO_Q15(cosf(2*M_PI/M*2)),FLOAT_TO_Q15(cosf(2*M_PI/M*1)),FLOAT_TO_Q15(cosf(2*M_PI/M*0)));//低位正，高位负
            __m256i w_imag = _mm256_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*0)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*0)));

            // __m256i w_real = _mm256_set_epi16(-ctx->cos_t[step*7],-ctx->cos_t[step*6],-ctx->cos_t[step*5],-ctx->cos_t[step*4],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*7],ctx->cos_t[step*6],ctx->cos_t[step*5],ctx->cos_t[step*4],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[0]);
            // __m256i w_imag = _mm256_set_epi16(ctx->sin_t[step*7],ctx->sin_t[step*6],ctx->sin_t[step*5],ctx->sin_t[step*4],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*7],-ctx->sin_t[step*6],-ctx->sin_t[step*5],-ctx->sin_t[step*4],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[0]);

            for(int j = 0;j<N/16;j++)
            {
                __m256i real_val = real_vec[j];
                __m256i imag_val = imag_vec[j];
                //0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
                __m256i a_real = _mm256_permute2x128_si256(real_val,real_val,0x00);
                __m256i a_imag = _mm256_permute2x128_si256(imag_val,imag_val,0x00);
                //8 9 10 11 12 13 14 15 8 9 10 11 12 13 14 15
                __m256i b_real = _mm256_permute2x128_si256(real_val,real_val,0x11);
                __m256i b_imag = _mm256_permute2x128_si256(imag_val,imag_val,0x11);

                __m256i r_real = _mm256_sub_epi16(_mm256_mulhrs_epi16(b_real,w_real),_mm256_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m256i i_imag = _mm256_add_epi16(_mm256_mulhrs_epi16(b_real,w_imag),_mm256_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm256_add_epi16(a_real,r_real);
                imag_vec[j] = _mm256_add_epi16(a_imag,i_imag);
            }





        }


        else{

            for(int k = 0;k<N/16;k+=M/16)
            {

                for(int j = 0;j<M/2/16;j++)
                {
                    int step = (ctx->size / M);
                    int idx = j * 16 * step;

                    __m256i w_real = _mm256_set_epi16(ctx->cos_t[idx+step*15],ctx->cos_t[idx+step*14],ctx->cos_t[idx+step*13],ctx->cos_t[idx+step*12],ctx->cos_t[idx+step*11],ctx->cos_t[idx+step*10],ctx->cos_t[idx+step*9],ctx->cos_t[idx+step*8],ctx->cos_t[idx+step*7],ctx->cos_t[idx+step*6],ctx->cos_t[idx+step*5],ctx->cos_t[idx+step*4],ctx->cos_t[idx+step*3],ctx->cos_t[idx+step*2],ctx->cos_t[idx+step*1],ctx->cos_t[0]);
                    __m256i w_imag = _mm256_set_epi16(-ctx->sin_t[idx+step*15],-ctx->sin_t[idx+step*14],-ctx->sin_t[idx+step*13],-ctx->sin_t[idx+step*12],-ctx->sin_t[idx+step*11],-ctx->sin_t[idx+step*10],-ctx->sin_t[idx+step*9],-ctx->sin_t[idx+step*8],-ctx->sin_t[idx+step*7],-ctx->sin_t[idx+step*6],-ctx->sin_t[idx+step*5],-ctx->sin_t[idx+step*4],-ctx->sin_t[idx+step*3],-ctx->sin_t[idx+step*2],-ctx->sin_t[idx+step*1],-ctx->sin_t[0]);


                    // __m256i w_real = _mm256_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/M*7)),FLOAT_TO_Q15(-cosf(2*M_PI/M*6)),FLOAT_TO_Q15(-cosf(2*M_PI/M*5)),FLOAT_TO_Q15(-cosf(2*M_PI/M*4)),FLOAT_TO_Q15(-cosf(2*M_PI/M*3)),FLOAT_TO_Q15(-cosf(2*M_PI/M*2)),FLOAT_TO_Q15(-cosf(2*M_PI/M*1)),FLOAT_TO_Q15(-cosf(2*M_PI/M*0)),FLOAT_TO_Q15(cosf(2*M_PI/M*7)),FLOAT_TO_Q15(cosf(2*M_PI/M*6)),FLOAT_TO_Q15(cosf(2*M_PI/M*5)),FLOAT_TO_Q15(cosf(2*M_PI/M*4)),FLOAT_TO_Q15(cosf(2*M_PI/M*3)),FLOAT_TO_Q15(cosf(2*M_PI/M*2)),FLOAT_TO_Q15(cosf(2*M_PI/M*1)),FLOAT_TO_Q15(cosf(2*M_PI/M*0)));//低位正，高位负
                    // __m256i w_imag = _mm256_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*0)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*0)));

                    __m256i real_val1 = real_vec[k+j];
                    __m256i imag_val1 = imag_vec[k+j];
                    __m256i real_val2 = real_vec[k+j+M/2/16];
                    __m256i imag_val2 = imag_vec[k+j+M/2/16];


                    __m256i r_real = _mm256_sub_epi16(_mm256_mulhrs_epi16(real_val2,w_real),_mm256_mulhrs_epi16(imag_val2,w_imag));//可用FMA优化 
                    __m256i i_imag = _mm256_add_epi16(_mm256_mulhrs_epi16(real_val2,w_imag),_mm256_mulhrs_epi16(imag_val2,w_real));//可用FMA优化

                    real_vec[k+j] = _mm256_add_epi16(real_val1,r_real);
                    imag_vec[k+j] = _mm256_add_epi16(imag_val1,i_imag);
                    real_vec[k+j+M/2/16] = _mm256_sub_epi16(real_val1,r_real);
                    imag_vec[k+j+M/2/16] = _mm256_sub_epi16(imag_val1,i_imag);




                }
            }






        }



    }
}
