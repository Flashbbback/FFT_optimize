
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
    //重排旋转因子 用于FFT中顺序读取
    int m = log2(max_size);
    ctx->stage_offsets = (int*)malloc((m + 1) * sizeof(int));
    memset(ctx->stage_offsets, -1, (m + 1) * sizeof(int));
    // 计算重排表所需的总大小
    size_t total_twiddles = 0;
    for (int s = 4; s <= m; s++) { // 从 M=16 (s=4) 开始需要通用加载
        int M = 1 << s;
        // 在这一阶段，内层循环(j)会执行 N/M * (M/2/8) = N/16 次加载
        total_twiddles += (max_size / 16); 
    }
    
    ctx->shuffled_cos_table = (float*)_aligned_malloc(total_twiddles * 8 * sizeof(float), 32);
    ctx->shuffled_sin_table = (float*)_aligned_malloc(total_twiddles * 8 * sizeof(float), 32);
    ctx->shuffled_cos_t = (int16_t*)_aligned_malloc(total_twiddles * 8 * sizeof(int16_t), 32);
    ctx->shuffled_sin_t = (int16_t*)_aligned_malloc(total_twiddles * 8 * sizeof(int16_t), 32);
    // 开始填充重排表
    size_t current_offset = 0;
    ctx->stage_offsets[0] = 0;
    for (int s = 1; s <= m; s++) {
        ctx->stage_offsets[s] = current_offset;
        
        // 我们只为需要跨步访问的通用阶段进行重排
        // M=2, 4, 8 可以硬编码，不需要查表
        if (s <= 3) continue; // M <= 8

        int M = 1 << s;
        int step = max_size / M;

        // 模拟fft_AVX的循环来填充数据
        for (int k = 0; k < max_size / 8; k += M / 8) {
            for (int j = 0; j < M / 2 / 8; j++) {
                int idx = j * 8 * step;

                // 把将来要访问的8个跨步值，现在就连续存起来
                for (int i = 0; i < 8; i++) {
                    ctx->shuffled_cos_table[current_offset + i] = ctx->cos_table[idx + i * step];
                    // 存 -sin 的值，这样在fft中就不用再取反了
                    ctx->shuffled_sin_table[current_offset + i] = -ctx->sin_table[idx + i * step];
                    ctx->shuffled_cos_t[current_offset + i] = FLOAT_TO_Q15(ctx->cos_table[idx + i * step]);
                    // 存 -sin 的值，这样在fft中就不用再取反了
                    ctx->shuffled_sin_t[current_offset + i] = FLOAT_TO_Q15(-ctx->sin_table[idx + i * step]);
                }
                current_offset += 8; // 移动到下一个存储位置
            }
        }
    }
    //




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
        size_t offset_ptr = ctx->stage_offsets[s];

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

                    // __m256 w_real = _mm256_set_ps(ctx->cos_table[idx+step*7],ctx->cos_table[idx+step*6],ctx->cos_table[idx+step*5],ctx->cos_table[idx+step*4],ctx->cos_table[idx+step*3],ctx->cos_table[idx+step*2],ctx->cos_table[idx+step*1],ctx->cos_table[idx]);
                    // __m256 w_imag = _mm256_set_ps(-ctx->sin_table[idx+step*7],-ctx->sin_table[idx+step*6],-ctx->sin_table[idx+step*5],-ctx->sin_table[idx+step*4],-ctx->sin_table[idx+step*3],-ctx->sin_table[idx+step*2],-ctx->sin_table[idx+step*1],-ctx->sin_table[idx]);

                    __m256 w_real = _mm256_load_ps(&ctx->shuffled_cos_table[offset_ptr]);
                    __m256 w_imag = _mm256_load_ps(&ctx->shuffled_sin_table[offset_ptr]);
                    offset_ptr += 8; // 简单地移动到下一组旋转因子



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
        size_t offset_ptr = ctx->stage_offsets[s];
        //M = 2时 cos = 1,sin = 0;
        if (M == 2)
        {
                        // // 提取偶数和奇数元素
            __m256i even_mask = _mm256_set_epi8(
                    13,12,9,8,5,4,1,0,13,12,9,8,5,4,1,0,
                    13,12,9,8,5,4,1,0,13,12,9,8,5,4,1,0
            );


            __m256i odd_mask = _mm256_set_epi8(
                    15,14,11,10,7,6,3,2,15,14,11,10,7,6,3,2,
                    15,14,11,10,7,6,3,2,15,14,11,10,7,6,3,2
            );
            for (int j = 0; j < N/16; j++)
            {
                __m256i real_val = real_vec[j];
                __m256i imag_val = imag_vec[j];

                
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
            // __m256i w_real = _mm256_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/M*7)),FLOAT_TO_Q15(-cosf(2*M_PI/M*6)),FLOAT_TO_Q15(-cosf(2*M_PI/M*5)),FLOAT_TO_Q15(-cosf(2*M_PI/M*4)),FLOAT_TO_Q15(-cosf(2*M_PI/M*3)),FLOAT_TO_Q15(-cosf(2*M_PI/M*2)),FLOAT_TO_Q15(-cosf(2*M_PI/M*1)),FLOAT_TO_Q15(-cosf(2*M_PI/M*0)),FLOAT_TO_Q15(cosf(2*M_PI/M*7)),FLOAT_TO_Q15(cosf(2*M_PI/M*6)),FLOAT_TO_Q15(cosf(2*M_PI/M*5)),FLOAT_TO_Q15(cosf(2*M_PI/M*4)),FLOAT_TO_Q15(cosf(2*M_PI/M*3)),FLOAT_TO_Q15(cosf(2*M_PI/M*2)),FLOAT_TO_Q15(cosf(2*M_PI/M*1)),FLOAT_TO_Q15(cosf(2*M_PI/M*0)));//低位正，高位负
            // __m256i w_imag = _mm256_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*0)),FLOAT_TO_Q15(sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(sinf(-2*M_PI/M*0)));

            __m256i w_real = _mm256_set_epi16(-ctx->cos_t[step*7],-ctx->cos_t[step*6],-ctx->cos_t[step*5],-ctx->cos_t[step*4],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*7],ctx->cos_t[step*6],ctx->cos_t[step*5],ctx->cos_t[step*4],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[0]);
            __m256i w_imag = _mm256_set_epi16(ctx->sin_t[step*7],ctx->sin_t[step*6],ctx->sin_t[step*5],ctx->sin_t[step*4],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*7],-ctx->sin_t[step*6],-ctx->sin_t[step*5],-ctx->sin_t[step*4],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[0]);

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

                    __m256i w_real = _mm256_load_si256((__m256i*)&ctx->shuffled_cos_t[offset_ptr]);
                    __m256i w_imag = _mm256_load_si256((__m256i*)&ctx->shuffled_sin_t[offset_ptr]);
                    
                    // 3. 更新偏移量，为下一次迭代做准备
                    offset_ptr += 16; // 因为我们一次处理了16个 int16_t

                    // __m256i w_real = _mm256_set_epi16(ctx->cos_t[idx+step*15],ctx->cos_t[idx+step*14],ctx->cos_t[idx+step*13],ctx->cos_t[idx+step*12],ctx->cos_t[idx+step*11],ctx->cos_t[idx+step*10],ctx->cos_t[idx+step*9],ctx->cos_t[idx+step*8],ctx->cos_t[idx+step*7],ctx->cos_t[idx+step*6],ctx->cos_t[idx+step*5],ctx->cos_t[idx+step*4],ctx->cos_t[idx+step*3],ctx->cos_t[idx+step*2],ctx->cos_t[idx+step*1],ctx->cos_t[0]);
                    // __m256i w_imag = _mm256_set_epi16(-ctx->sin_t[idx+step*15],-ctx->sin_t[idx+step*14],-ctx->sin_t[idx+step*13],-ctx->sin_t[idx+step*12],-ctx->sin_t[idx+step*11],-ctx->sin_t[idx+step*10],-ctx->sin_t[idx+step*9],-ctx->sin_t[idx+step*8],-ctx->sin_t[idx+step*7],-ctx->sin_t[idx+step*6],-ctx->sin_t[idx+step*5],-ctx->sin_t[idx+step*4],-ctx->sin_t[idx+step*3],-ctx->sin_t[idx+step*2],-ctx->sin_t[idx+step*1],-ctx->sin_t[0]);


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


void fft_AVX512_fixedP(int16_t real[], int16_t imag[], int N, FFTContext *ctx)
{
    bit_reverse_q15(real, imag, N,ctx);
     __m512i *real_vec = (__m512i *)real;
     __m512i *imag_vec = (__m512i *)imag;
    int m = log2(N);
    for (int s = 1;s<=m;s++)
    {
        int M = 1<<s;
        size_t offset_ptr = ctx->stage_offsets[s];
        //M = 2时 cos = 1,sin = 0;
        if (M == 2)
        {


            //提取偶数和奇数元素
            __m512i even_mask = _mm512_set_epi8(
                0x3D,0x3C,0x39,0x38,0x35,0x34,0x31,0x30,
                0x3D,0x3C,0x39,0x38,0x35,0x34,0x31,0x30,
                0x2D,0x2C,0x29,0x28,0x25,0x24,0x21,0x20,
                0x2D,0x2C,0x29,0x28,0x25,0x24,0x21,0x20,
                0x1D,0x1C,0x19,0x18,0x15,0x14,0x11,0x10,
                0x1D,0x1C,0x19,0x18,0x15,0x14,0x11,0x10,
                0x0D,0x0C,0x09,0x08,0x05,0x04,0x01,0x00,
                0x0D,0x0C,0x09,0x08,0x05,0x04,0x01,0x00
            );


            __m512i odd_mask = _mm512_set_epi8(
                0x3F,0x3E,0x3B,0x3A,0x37,0x36,0x33,0x32,
                0x3F,0x3E,0x3B,0x3A,0x37,0x36,0x33,0x32,
                0x2F,0x2E,0x2B,0x2A,0x27,0x26,0x23,0x22,
                0x2F,0x2E,0x2B,0x2A,0x27,0x26,0x23,0x22,
                0x1F,0x1E,0x1B,0x1A,0x17,0x16,0x13,0x12,
                0x1F,0x1E,0x1B,0x1A,0x17,0x16,0x13,0x12,
                0x0F,0x0E,0x0B,0x0A,0x07,0x06,0x03,0x02,
                0x0F,0x0E,0x0B,0x0A,0x07,0x06,0x03,0x02
            );
            for (int j = 0; j < N/32; j++)
            {
                __m512i real_val = real_vec[j];
                __m512i imag_val = imag_vec[j];

                __m512i even_real = _mm512_shuffle_epi8(real_val,even_mask);
                __m512i even_imag = _mm512_shuffle_epi8(imag_val,even_mask);
                __m512i odd_real = _mm512_shuffle_epi8(real_val,odd_mask);
                __m512i odd_imag = _mm512_shuffle_epi8(imag_val,odd_mask);

                __m512i result_real = _mm512_add_epi16(even_real, odd_real);
                __m512i result_imag = _mm512_add_epi16(even_imag, odd_imag);
                __m512i result_real2 = _mm512_sub_epi16(even_real, odd_real);
                __m512i result_imag2 = _mm512_sub_epi16(even_imag, odd_imag);
                
                // 交错存储结果
                real_vec[j] = _mm512_unpacklo_epi16(result_real, result_real2);
                imag_vec[j] = _mm512_unpacklo_epi16(result_imag, result_imag2);
            }
        }

        else if(M == 4)
        {
            // __m512i w_real = _mm512_setr_epi16(1,0,-1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0);
            // __m512i w_imag = _mm512_setr_epi16(0,-1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0,1);

            int step = ctx->size / M;
            __m512i w_real = _mm512_set_epi16(ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0]);
            __m512i w_imag = _mm512_set_epi16(ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0]);


            // 提取对应元素
            //0 1 0 1  4 5 4 5 8 9 8 9  12 13 12 13
            __m512i a_mask = _mm512_set_epi8(
                // m63~m48（d24~d31的高/低字节）
                0x3D, 0x3C, 0x3B, 0x3A, 0x3D, 0x3C, 0x3B, 0x3A,
                0x35, 0x34, 0x33, 0x32, 0x35, 0x34, 0x33, 0x32,
                // m47~m32（d16~d23的高/低字节）
                0x2D, 0x2C, 0x2B, 0x2A, 0x2D, 0x2C, 0x2B, 0x2A,
                0x25, 0x24, 0x23, 0x22, 0x25, 0x24, 0x23, 0x22,
                // m31~m16（d8~d15的高/低字节）
                0x1D, 0x1C, 0x1B, 0x1A, 0x1D, 0x1C, 0x1B, 0x1A,
                0x15, 0x14, 0x13, 0x12, 0x15, 0x14, 0x13, 0x12,
                // m15~m0（d0~d7的高/低字节）
                0x0B, 0x0A, 0x09, 0x08, 0x0B, 0x0A, 0x09, 0x08,
                0x03, 0x02, 0x01, 0x00, 0x03, 0x02, 0x01, 0x00
            );

            // 2 3 2 3 6 7 6 7 10 11 10 11 14 15 14 15
            __m512i b_mask = _mm512_set_epi8(
                // m63~m48（d24~d31的高/低字节）
                0x3F, 0x3E, 0x3D, 0x3C, 0x3F, 0x3E, 0x3D, 0x3C,
                0x37, 0x36, 0x35, 0x34, 0x37, 0x36, 0x35, 0x34,
                // m47~m32（d16~d23的高/低字节）
                0x2F, 0x2E, 0x2D, 0x2C, 0x2F, 0x2E, 0x2D, 0x2C,
                0x27, 0x26, 0x25, 0x24, 0x27, 0x26, 0x25, 0x24,
                // m31~m16（d8~d15的高/低字节）
                0x1F, 0x1E, 0x1D, 0x1C, 0x1F, 0x1E, 0x1D, 0x1C,
                0x17, 0x16, 0x15, 0x14, 0x17, 0x16, 0x15, 0x14,
                // m15~m0（d0~d7的高/低字节）
                0x0F, 0x0E, 0x0D, 0x0C, 0x0F, 0x0E, 0x0D, 0x0C,
                0x07, 0x06, 0x05, 0x04, 0x07, 0x06, 0x05, 0x04
            );
            
            for(int j = 0;j<N/32;j++)
            {
                __m512i real_val = real_vec[j];
                __m512i imag_val = imag_vec[j];

                __m512i a_real = _mm512_shuffle_epi8(real_val,a_mask);
                __m512i a_imag = _mm512_shuffle_epi8(imag_val,a_mask);
                __m512i b_real = _mm512_shuffle_epi8(real_val,b_mask);
                __m512i b_imag = _mm512_shuffle_epi8(imag_val,b_mask);
                //0+2 1+3 0-2 1-3 4+6 5+7 4-6 5-7...
                __m512i r_real = _mm512_sub_epi16(_mm512_mulhrs_epi16(b_real,w_real),_mm512_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m512i i_imag = _mm512_add_epi16(_mm512_mulhrs_epi16(b_real,w_imag),_mm512_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm512_add_epi16(a_real,r_real);
                imag_vec[j] = _mm512_add_epi16(a_imag,i_imag);
            }



        }

        else if(M == 8)
        {
            // __m512i w_real = _mm512_set_epi16(-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0),-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0));//低位正，高位负
            // __m512i w_imag = _mm512_set_epi16(-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0),-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0));

            int step = ctx->size / M;
            __m512i w_real = _mm512_set_epi16(-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0]);
            __m512i w_imag = _mm512_set_epi16(ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0]);


            //0 1 2 3 0 1 2 3 2 9 10 11 8 9 10 11
            __m512i a_mask = _mm512_set_epi8(
            // m63~m48（d24~d31的高/低字节）
                0x3D,0x3C,0x3B,0x3A,0x39,0x38,0x37,0x36,
                0x3D,0x3C,0x3B,0x3A,0x39,0x38,0x37,0x36,
                // m47~m32（d16~d23的高/低字节）
                0x2D,0x2C,0x2B,0x2A,0x29,0x28,0x27,0x26,
                0x2D,0x2C,0x2B,0x2A,0x29,0x28,0x27,0x26,
                // m31~m16（d8~d15的高/低字节）
                0x1D,0x1C,0x1B,0x1A,0x19,0x18,0x17,0x16,
                0x1D,0x1C,0x1B,0x1A,0x19,0x18,0x17,0x16,
                // m15~m0（d0~d7的高/低字节）
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00
            );

            //4 5 6 7 4 5 6 7 12 13 14 15 12 13 14 15
            __m512i b_mask = _mm512_set_epi8(
                // m63~m48（d24~d31的高/低字节）
                0x3F,0x3E,0x3D,0x3C,0x3B,0x3A,0x39,0x38,
                0x3F,0x3E,0x3D,0x3C,0x3B,0x3A,0x39,0x38,
                // m47~m32（d16~d23的高/低字节）
                0x2F,0x2E,0x2D,0x2C,0x2B,0x2A,0x29,0x28,
                0x2F,0x2E,0x2D,0x2C,0x2B,0x2A,0x29,0x28,
                // m31~m16（d8~d15的高/低字节）
                0x1F,0x1E,0x1D,0x1C,0x1B,0x1A,0x19,0x18,
                0x1F,0x1E,0x1D,0x1C,0x1B,0x1A,0x19,0x18,
                // m15~m0（d0~d7的高/低字节）
                0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08,
                0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08
            );
            for(int j = 0;j<N/32;j++)
            {
                __m512i real_val = real_vec[j];
                __m512i imag_val = imag_vec[j];

                __m512i a_real = _mm512_shuffle_epi8(real_val,a_mask);
                __m512i a_imag = _mm512_shuffle_epi8(imag_val,a_mask);
                __m512i b_real = _mm512_shuffle_epi8(real_val,b_mask);
                __m512i b_imag = _mm512_shuffle_epi8(imag_val,b_mask);

                __m512i r_real = _mm512_sub_epi16(_mm512_mulhrs_epi16(b_real,w_real),_mm512_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m512i i_imag = _mm512_add_epi16(_mm512_mulhrs_epi16(b_real,w_imag),_mm512_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm512_add_epi16(a_real,r_real);
                imag_vec[j] = _mm512_add_epi16(a_imag,i_imag);
            }



        }

        else if(M == 16)
        {

            int step = ctx->size/M;
            // __m512i w_real = _mm512_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/M*7)),FLOAT_TO_Q15(-cosf(2*M_PI/M*6)),FLOAT_TO_Q15(-cosf(2*M_PI/M*5)),FLOAT_TO_Q15(-cosf(2*M_PI/M*4)),FLOAT_TO_Q15(-cosf(2*M_PI/M*3)),FLOAT_TO_Q15(-cosf(2*M_PI/M*2)),FLOAT_TO_Q15(-cosf(2*M_PI/M*1)),FLOAT_TO_Q15(-cosf(2*M_PI/M*0)),FLOAT_TO_Q15(cosf(2*M_PI/M*7)),FLOAT_TO_Q15(cosf(2*M_PI/M*6)),FLOAT_TO_Q15(cosf(2*M_PI/M*5)),FLOAT_TO_Q15(cosf(2*M_PI/M*4)),FLOAT_TO_Q15(cosf(2*M_PI/M*3)),FLOAT_TO_Q15(cosf(2*M_PI/M*2)),FLOAT_TO_Q15(cosf(2*M_PI/M*1)),FLOAT_TO_Q15(cosf(2*M_PI/M*0)));//低位正，高位负
            // __m512i w_imag = _mm512_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*0)),FLOAT_TO_Q15(sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(sinf(-2*M_PI/M*0)));
            // 序列1掩码：0~7重复、16~23重复
            __m512i a_mask= _mm512_set_epi8(
                // m63~m48（d24~d31的高/低字节：s16~23 → 字节索引32~47）
                0x2F,0x2E,0x2D,0x2C,0x2B,0x2A,0x29,0x28,
                0x27,0x26,0x25,0x24,0x23,0x22,0x21,0x20,
                // m47~m32（d16~d23的高/低字节：s16~23 → 字节索引32~47）
                0x2F,0x2E,0x2D,0x2C,0x2B,0x2A,0x29,0x28,
                0x27,0x26,0x25,0x24,0x23,0x22,0x21,0x20,
                // m31~m16（d8~d15的高/低字节：s0~7 → 字节索引0~15）
                0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08,
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,
                // m15~m0（d0~d7的高/低字节：s0~7 → 字节索引0~15）
                0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08,
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00
            );

            // 序列2掩码：8~15重复、24~31重复
            __m512i b_mask = _mm512_set_epi8(
                // m63~m48（d24~d31的高/低字节：s24~31 → 字节索引48~63）
                0x3F,0x3E,0x3D,0x3C,0x3B,0x3A,0x39,0x38,
                0x37,0x36,0x35,0x34,0x33,0x32,0x31,0x30,
                // m47~m32（d16~d23的高/低字节：s24~31 → 字节索引48~63）
                0x3F,0x3E,0x3D,0x3C,0x3B,0x3A,0x39,0x38,
                0x37,0x36,0x35,0x34,0x33,0x32,0x31,0x30,
                // m31~m16（d8~d15的高/低字节：s8~15 → 字节索引16~31）
                0x1F,0x1E,0x1D,0x1C,0x1B,0x1A,0x19,0x18,
                0x17,0x16,0x15,0x14,0x13,0x12,0x11,0x10,
                // m15~m0（d0~d7的高/低字节：s8~15 → 字节索引16~31）
                0x1F,0x1E,0x1D,0x1C,0x1B,0x1A,0x19,0x18,
                0x17,0x16,0x15,0x14,0x13,0x12,0x11,0x10
            );

            __m512i w_real = _mm512_set_epi16(-ctx->cos_t[step*7],-ctx->cos_t[step*6],-ctx->cos_t[step*5],-ctx->cos_t[step*4],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*7],ctx->cos_t[step*6],ctx->cos_t[step*5],ctx->cos_t[step*4],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[0],-ctx->cos_t[step*7],-ctx->cos_t[step*6],-ctx->cos_t[step*5],-ctx->cos_t[step*4],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*7],ctx->cos_t[step*6],ctx->cos_t[step*5],ctx->cos_t[step*4],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[0]);
            __m512i w_imag = _mm512_set_epi16(ctx->sin_t[step*7],ctx->sin_t[step*6],ctx->sin_t[step*5],ctx->sin_t[step*4],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*7],-ctx->sin_t[step*6],-ctx->sin_t[step*5],-ctx->sin_t[step*4],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[0],ctx->sin_t[step*7],ctx->sin_t[step*6],ctx->sin_t[step*5],ctx->sin_t[step*4],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*7],-ctx->sin_t[step*6],-ctx->sin_t[step*5],-ctx->sin_t[step*4],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[0]);

            for(int j = 0;j<N/32;j++)
            {
                __m512i real_val = real_vec[j];
                __m512i imag_val = imag_vec[j];

                __m512i a_real = _mm512_shuffle_epi8(real_val,a_mask);
                __m512i a_imag = _mm512_shuffle_epi8(imag_val,a_mask);
                __m512i b_real = _mm512_shuffle_epi8(real_val,b_mask);
                __m512i b_imag = _mm512_shuffle_epi8(imag_val,b_mask);

                __m512i r_real = _mm512_sub_epi16(_mm512_mulhrs_epi16(b_real,w_real),_mm512_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m512i i_imag = _mm512_add_epi16(_mm512_mulhrs_epi16(b_real,w_imag),_mm512_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm512_add_epi16(a_real,r_real);
                imag_vec[j] = _mm512_add_epi16(a_imag,i_imag);
            }

        }

        else if(M == 32)
        {

            int step = ctx->size/M;
            // __m512i w_real = _mm512_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/M*7)),FLOAT_TO_Q15(-cosf(2*M_PI/M*6)),FLOAT_TO_Q15(-cosf(2*M_PI/M*5)),FLOAT_TO_Q15(-cosf(2*M_PI/M*4)),FLOAT_TO_Q15(-cosf(2*M_PI/M*3)),FLOAT_TO_Q15(-cosf(2*M_PI/M*2)),FLOAT_TO_Q15(-cosf(2*M_PI/M*1)),FLOAT_TO_Q15(-cosf(2*M_PI/M*0)),FLOAT_TO_Q15(cosf(2*M_PI/M*7)),FLOAT_TO_Q15(cosf(2*M_PI/M*6)),FLOAT_TO_Q15(cosf(2*M_PI/M*5)),FLOAT_TO_Q15(cosf(2*M_PI/M*4)),FLOAT_TO_Q15(cosf(2*M_PI/M*3)),FLOAT_TO_Q15(cosf(2*M_PI/M*2)),FLOAT_TO_Q15(cosf(2*M_PI/M*1)),FLOAT_TO_Q15(cosf(2*M_PI/M*0)));//低位正，高位负
            // __m512i w_imag = _mm512_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*0)),FLOAT_TO_Q15(sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(sinf(-2*M_PI/M*0)));
            __m512i a_mask= _mm512_set_epi8(
            // 序列1掩码：字节索引0~31重复（对应s0~s15重复）
                0x1F,0x1E,0x1D,0x1C,0x1B,0x1A,0x19,0x18,  // m63~m56：s15的高/低字节 → 31,30,...,24
                0x17,0x16,0x15,0x14,0x13,0x12,0x11,0x10,  // m55~m48：s14~s8的高/低字节 → 23,22,...,16
                0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08,  // m47~m40：s7的高/低字节 → 15,14,...,8
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,  // m39~m32：s6~s0的高/低字节 → 7,6,...,0
                // 以下是0~31重复（m31~m0，与上面m63~m32完全一致）
                0x1F,0x1E,0x1D,0x1C,0x1B,0x1A,0x19,0x18,
                0x17,0x16,0x15,0x14,0x13,0x12,0x11,0x10,
                0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08,
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00
            );

            __m512i b_mask = _mm512_set_epi8(
            // 序列2掩码：字节索引32~63重复（对应s16~s31重复）
                0x3F,0x3E,0x3D,0x3C,0x3B,0x3A,0x39,0x38,  // m63~m56：s31的高/低字节 → 63,62,...,56
                0x37,0x36,0x35,0x34,0x33,0x32,0x31,0x30,  // m55~m48：s30~s24的高/低字节 → 55,54,...,48
                0x2F,0x2E,0x2D,0x2C,0x2B,0x2A,0x29,0x28,  // m47~m40：s23的高/低字节 → 47,46,...,40
                0x27,0x26,0x25,0x24,0x23,0x22,0x21,0x20,  // m39~m32：s22~s16的高/低字节 → 39,38,...,32
                // 以下是32~63重复（m31~m0，与上面m63~m32完全一致）
                0x3F,0x3E,0x3D,0x3C,0x3B,0x3A,0x39,0x38,
                0x37,0x36,0x35,0x34,0x33,0x32,0x31,0x30,
                0x2F,0x2E,0x2D,0x2C,0x2B,0x2A,0x29,0x28,
                0x27,0x26,0x25,0x24,0x23,0x22,0x21,0x20
            );

            __m512i w_real = _mm512_set_epi16(-ctx->cos_t[step*15],-ctx->cos_t[step*14],-ctx->cos_t[step*13],-ctx->cos_t[step*12],-ctx->cos_t[step*11],-ctx->cos_t[step*10],-ctx->cos_t[step*9],-ctx->cos_t[step*8],-ctx->cos_t[step*7],-ctx->cos_t[step*6],-ctx->cos_t[step*5],-ctx->cos_t[step*4],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[0],ctx->cos_t[step*15],ctx->cos_t[step*14],ctx->cos_t[step*13],ctx->cos_t[step*12],ctx->cos_t[step*11],ctx->cos_t[step*10],ctx->cos_t[step*9],ctx->cos_t[step*8],ctx->cos_t[step*7],ctx->cos_t[step*6],ctx->cos_t[step*5],ctx->cos_t[step*4],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[0]);
            __m512i w_imag = _mm512_set_epi16(ctx->sin_t[step*15],ctx->sin_t[step*14],ctx->sin_t[step*13],ctx->sin_t[step*12],ctx->sin_t[step*11],ctx->sin_t[step*10],ctx->sin_t[step*9],ctx->sin_t[step*8],ctx->sin_t[step*7],ctx->sin_t[step*6],ctx->sin_t[step*5],ctx->sin_t[step*4],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[0],-ctx->sin_t[step*15],-ctx->sin_t[step*14],-ctx->sin_t[step*13],-ctx->sin_t[step*12],-ctx->sin_t[step*11],-ctx->sin_t[step*10],-ctx->sin_t[step*9],-ctx->sin_t[step*8],-ctx->sin_t[step*7],-ctx->sin_t[step*6],-ctx->sin_t[step*5],-ctx->sin_t[step*4],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[0]);

            for(int j = 0;j<N/32;j++)
            {
                __m512i real_val = real_vec[j];
                __m512i imag_val = imag_vec[j];

                __m512i a_real = _mm512_shuffle_epi8(real_val,a_mask);
                __m512i a_imag = _mm512_shuffle_epi8(imag_val,a_mask);
                __m512i b_real = _mm512_shuffle_epi8(real_val,b_mask);
                __m512i b_imag = _mm512_shuffle_epi8(imag_val,b_mask);

                __m512i r_real = _mm512_sub_epi16(_mm512_mulhrs_epi16(b_real,w_real),_mm512_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m512i i_imag = _mm512_add_epi16(_mm512_mulhrs_epi16(b_real,w_imag),_mm512_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm512_add_epi16(a_real,r_real);
                imag_vec[j] = _mm512_add_epi16(a_imag,i_imag);
            }

        }

        else{

            for(int k = 0;k<N/32;k+=M/32)
            {

                for(int j = 0;j<M/2/32;j++)
                {
                    int step = (ctx->size / M);
                    int idx = j * 32 * step;

                    __m512i w_real = _mm512_stream_load_si512((__m512i*)&ctx->shuffled_cos_t[offset_ptr]);
                    __m512i w_imag = _mm512_stream_load_si512((__m512i*)&ctx->shuffled_sin_t[offset_ptr]);
                    
                    // 3. 更新偏移量，为下一次迭代做准备
                    offset_ptr += 32; // 因为我们一次处理了32个 int16_t

                    // __m512i w_real = _mm512_set_epi16(ctx->cos_t[idx+step*15],ctx->cos_t[idx+step*14],ctx->cos_t[idx+step*13],ctx->cos_t[idx+step*12],ctx->cos_t[idx+step*11],ctx->cos_t[idx+step*10],ctx->cos_t[idx+step*9],ctx->cos_t[idx+step*8],ctx->cos_t[idx+step*7],ctx->cos_t[idx+step*6],ctx->cos_t[idx+step*5],ctx->cos_t[idx+step*4],ctx->cos_t[idx+step*3],ctx->cos_t[idx+step*2],ctx->cos_t[idx+step*1],ctx->cos_t[0]);
                    // __m512i w_imag = _mm512_set_epi16(-ctx->sin_t[idx+step*15],-ctx->sin_t[idx+step*14],-ctx->sin_t[idx+step*13],-ctx->sin_t[idx+step*12],-ctx->sin_t[idx+step*11],-ctx->sin_t[idx+step*10],-ctx->sin_t[idx+step*9],-ctx->sin_t[idx+step*8],-ctx->sin_t[idx+step*7],-ctx->sin_t[idx+step*6],-ctx->sin_t[idx+step*5],-ctx->sin_t[idx+step*4],-ctx->sin_t[idx+step*3],-ctx->sin_t[idx+step*2],-ctx->sin_t[idx+step*1],-ctx->sin_t[0]);


                    // __m512i w_real = _mm512_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/M*7)),FLOAT_TO_Q15(-cosf(2*M_PI/M*6)),FLOAT_TO_Q15(-cosf(2*M_PI/M*5)),FLOAT_TO_Q15(-cosf(2*M_PI/M*4)),FLOAT_TO_Q15(-cosf(2*M_PI/M*3)),FLOAT_TO_Q15(-cosf(2*M_PI/M*2)),FLOAT_TO_Q15(-cosf(2*M_PI/M*1)),FLOAT_TO_Q15(-cosf(2*M_PI/M*0)),FLOAT_TO_Q15(cosf(2*M_PI/M*7)),FLOAT_TO_Q15(cosf(2*M_PI/M*6)),FLOAT_TO_Q15(cosf(2*M_PI/M*5)),FLOAT_TO_Q15(cosf(2*M_PI/M*4)),FLOAT_TO_Q15(cosf(2*M_PI/M*3)),FLOAT_TO_Q15(cosf(2*M_PI/M*2)),FLOAT_TO_Q15(cosf(2*M_PI/M*1)),FLOAT_TO_Q15(cosf(2*M_PI/M*0)));//低位正，高位负
                    // __m512i w_imag = _mm512_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*0)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/M*0)));

                    __m512i real_val1 = real_vec[k+j];
                    __m512i imag_val1 = imag_vec[k+j];
                    __m512i real_val2 = real_vec[k+j+M/2/32];
                    __m512i imag_val2 = imag_vec[k+j+M/2/32];


                    __m512i r_real = _mm512_sub_epi16(_mm512_mulhrs_epi16(real_val2,w_real),_mm512_mulhrs_epi16(imag_val2,w_imag));//可用FMA优化 
                    __m512i i_imag = _mm512_add_epi16(_mm512_mulhrs_epi16(real_val2,w_imag),_mm512_mulhrs_epi16(imag_val2,w_real));//可用FMA优化

                    real_vec[k+j] = _mm512_add_epi16(real_val1,r_real);
                    imag_vec[k+j] = _mm512_add_epi16(imag_val1,i_imag);
                    real_vec[k+j+M/2/32] = _mm512_sub_epi16(real_val1,r_real);
                    imag_vec[k+j+M/2/32] = _mm512_sub_epi16(imag_val1,i_imag);




                }
            }
        }
    }
}