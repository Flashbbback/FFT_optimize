#include <stdio.h>
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
// 计算核：AVX 单精度浮点版本实现
// =============================================================
// Stage 1: M = 2
// 寄存器内 shuffle 和 蝶形运算
// =============================================================
static inline void fft_stage_float_M2(__m256 *real_vec, __m256 *imag_vec, int num_vecs)
{
    for (int j = 0; j < num_vecs; j++)
    {
        __m256 r = real_vec[j];
        __m256 i = imag_vec[j];
        
        // 提取偶数和奇数元素
        // _MM_SHUFFLE(3,2,1,0) -> select indices
        __m256 even_real = _mm256_shuffle_ps(r, r, _MM_SHUFFLE(2,0,2,0));
        __m256 odd_real  = _mm256_shuffle_ps(r, r, _MM_SHUFFLE(3,1,3,1));
        __m256 even_imag = _mm256_shuffle_ps(i, i, _MM_SHUFFLE(2,0,2,0));
        __m256 odd_imag  = _mm256_shuffle_ps(i, i, _MM_SHUFFLE(3,1,3,1));
        
        // 蝶形运算
        __m256 result_real_sum = _mm256_add_ps(even_real, odd_real);
        __m256 result_imag_sum = _mm256_add_ps(even_imag, odd_imag);
        __m256 result_real_sub = _mm256_sub_ps(even_real, odd_real);
        __m256 result_imag_sub = _mm256_sub_ps(even_imag, odd_imag);
        
        // 交错存储结果
        real_vec[j] = _mm256_unpacklo_ps(result_real_sum, result_real_sub);
        imag_vec[j] = _mm256_unpacklo_ps(result_imag_sum, result_imag_sub);
    }
}

// =============================================================
// Stage 2: M = 4
// =============================================================
static inline void fft_stage_float_M4(__m256 *real_vec, __m256 *imag_vec, int num_vecs)
{

    __m256 w_real = _mm256_set_ps(0,-1,0,1,0,-1,0,1);
    __m256 w_imag = _mm256_set_ps(1,0,-1,0,1,0,-1,0);

    for(int j = 0; j < num_vecs; j++)
    {
        __m256 r = real_vec[j];
        __m256 i = imag_vec[j];

        __m256 a_real = _mm256_shuffle_ps(r, r, _MM_SHUFFLE(1,0,1,0));
        __m256 b_real = _mm256_shuffle_ps(r, r, _MM_SHUFFLE(3,2,3,2));
        __m256 a_imag = _mm256_shuffle_ps(i, i, _MM_SHUFFLE(1,0,1,0));
        __m256 b_imag = _mm256_shuffle_ps(i, i, _MM_SHUFFLE(3,2,3,2));

        // 复杂乘法 + 蝶形
        // (br + j*bi) * (wr + j*wi)
        __m256 term1 = _mm256_mul_ps(b_real, w_real);
        __m256 term2 = _mm256_mul_ps(b_imag, w_imag);
        __m256 r_real = _mm256_sub_ps(term1, term2);

        __m256 term3 = _mm256_mul_ps(b_real, w_imag);
        __m256 term4 = _mm256_mul_ps(b_imag, w_real);
        __m256 i_imag = _mm256_add_ps(term3, term4);

        real_vec[j] = _mm256_add_ps(a_real, r_real);
        imag_vec[j] = _mm256_add_ps(a_imag, i_imag);
    }
}

// =============================================================
// Stage 3: M = 8
// =============================================================
static inline void fft_stage_float_M8(__m256 *real_vec, __m256 *imag_vec, int num_vecs)
{
    
    static const __m256 w_real = {
         1.0f,  SQRT2_2,  0.0f, -SQRT2_2, 
         -1.0f, -SQRT2_2,  0.0f, SQRT2_2
    };
    
    static const __m256 w_imag = {
         0.0f, -SQRT2_2, -1.0f, -SQRT2_2,
         0.0f, SQRT2_2, 1.0f, SQRT2_2
    };

    for(int j = 0; j < num_vecs; j++)
    {
        __m256 r = real_vec[j];
        __m256 i = imag_vec[j];

        // 0x00: Extract lower 128 bits to both halves
        // 0x11: Extract upper 128 bits to both halves
        __m256 a_real = _mm256_permute2f128_ps(r, r, 0x00);
        __m256 b_real = _mm256_permute2f128_ps(r, r, 0x11);
        __m256 a_imag = _mm256_permute2f128_ps(i, i, 0x00);
        __m256 b_imag = _mm256_permute2f128_ps(i, i, 0x11);

        // Complex Multiply
        __m256 tr = _mm256_sub_ps(_mm256_mul_ps(b_real, w_real), _mm256_mul_ps(b_imag, w_imag));
        __m256 ti = _mm256_add_ps(_mm256_mul_ps(b_real, w_imag), _mm256_mul_ps(b_imag, w_real));

        real_vec[j] = _mm256_add_ps(a_real, tr);
        imag_vec[j] = _mm256_add_ps(a_imag, ti);
    }
}

// =============================================================
// 主函数: AVX Float Optimized
// =============================================================
void fft_AVX(float *real, float *imag, int N, FFTContext *ctx)
{
    // 1. Bit Reverse
    bit_reverse(real, imag, N, ctx);
    
    __m256 *real_vec = (__m256 *)real;
    __m256 *imag_vec = (__m256 *)imag;
    
    // 快速计算 log2
    int m = 0; 
    int tempN = N; while(tempN >>= 1) m++;
    int num_vecs = N / 8; // float 是 32bit, AVX256存8个

    // 2. Intra-register Stages (No branching)
    if (m >= 1) fft_stage_float_M2(real_vec, imag_vec, num_vecs);
    if (m >= 2) fft_stage_float_M4(real_vec, imag_vec, num_vecs);
    if (m >= 3) fft_stage_float_M8(real_vec, imag_vec, num_vecs);

    // 3. Inter-register Stages (M >= 16)
    // 统一循环处理，无内存 shuffle，直接加载预计算的表
    for (int s = 4; s <= m; s++)
    {
        int M = 1 << s;
        size_t offset_ptr = ctx->stage_offsets[s];
        
        int step_vecs = M / 8;     // stride in vectors
        int half_step = step_vecs / 2;

        for (int k = 0; k < num_vecs; k += step_vecs)
        {
            for (int j = 0; j < half_step; j++)
            {
                // 加载旋转因子
                // 每次处理一个 vector (8个点)，所以步进是 8
                __m256 w_real = _mm256_load_ps(&ctx->shuffled_cos_table[offset_ptr]);
                __m256 w_imag = _mm256_load_ps(&ctx->shuffled_sin_table[offset_ptr]);
                offset_ptr += 8; 

                int idx1 = k + j;
                int idx2 = idx1 + half_step; // 蝶形运算的另一半

                __m256 r1 = real_vec[idx1];
                __m256 i1 = imag_vec[idx1];
                __m256 r2 = real_vec[idx2];
                __m256 i2 = imag_vec[idx2];

                // 蝶形运算
                // (r2 + j*i2) * (wr + j*wi)
                __m256 term1 = _mm256_mul_ps(r2, w_real);
                __m256 term2 = _mm256_mul_ps(i2, w_imag);
                __m256 tr    = _mm256_sub_ps(term1, term2);

                __m256 term3 = _mm256_mul_ps(r2, w_imag);
                __m256 term4 = _mm256_mul_ps(i2, w_real);
                __m256 ti    = _mm256_add_ps(term3, term4);

                real_vec[idx1] = _mm256_add_ps(r1, tr);
                imag_vec[idx1] = _mm256_add_ps(i1, ti);
                real_vec[idx2] = _mm256_sub_ps(r1, tr);
                imag_vec[idx2] = _mm256_sub_ps(i1, ti);
            }
        }
    }
}