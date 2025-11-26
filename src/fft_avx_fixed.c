// 计算核：AVX 定点数 (Q15) 版本实现
#include "fft_internal.h"
#include <stdio.h>
#include <stdint.h>

static inline void bit_reverse_q15(int16_t real[], int16_t imag[], int N,FFTContext* ctx)
{

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
// =============================================================
// Stage 1: M = 2 (Butterfly Step = 1)
// 处理寄存器内部相邻元素: (0,1), (2,3)...
// =============================================================
static inline void fft_stage_M2(__m256i *real_vec, __m256i *imag_vec, int num_vecs)
{
    // Mask 用于提取偶数位 (0, 2, 4, 6...) 和 奇数位 (1, 3, 5, 7...)
    static const __m256i even_mask = {
        // _mm256_set_epi8 是逆序参数 (e31...e0)
        // 这里保留原代码的逻辑顺序
        0x0D0C090805040100ULL, 0x0D0C090805040100ULL,
        0x0D0C090805040100ULL, 0x0D0C090805040100ULL
    };
    static const __m256i odd_mask = {
        0x0F0E0B0A07060302ULL, 0x0F0E0B0A07060302ULL,
        0x0F0E0B0A07060302ULL, 0x0F0E0B0A07060302ULL
    };

    for (int j = 0; j < num_vecs; j++)
    {
        __m256i r = real_vec[j];
        __m256i i = imag_vec[j];

        __m256i er = _mm256_shuffle_epi8(r, even_mask);
        __m256i ei = _mm256_shuffle_epi8(i, even_mask);
        __m256i or_val = _mm256_shuffle_epi8(r, odd_mask);
        __m256i oi = _mm256_shuffle_epi8(i, odd_mask);

        // 蝶形运算: A+B, A-B
        __m256i sum_r = _mm256_add_epi16(er, or_val);
        __m256i sum_i = _mm256_add_epi16(ei, oi);
        __m256i sub_r = _mm256_sub_epi16(er, or_val);
        __m256i sub_i = _mm256_sub_epi16(ei, oi);

        // 交错存储: (Sum0, Sub0, Sum1, Sub1...)
        real_vec[j] = _mm256_unpacklo_epi16(sum_r, sub_r);
        imag_vec[j] = _mm256_unpacklo_epi16(sum_i, sub_i);
    }
}

// =============================================================
// Stage 2: M = 4 (Butterfly Step = 2)
// =============================================================
static inline void fft_stage_M4(__m256i *real_vec, __m256i *imag_vec, int num_vecs, FFTContext *ctx)
{
    int step = ctx->size / 4;
    __m256i w_real = _mm256_set_epi16(ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0]);
    __m256i w_imag = _mm256_set_epi16(ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0]);
    __m256i a_mask = _mm256_setr_epi8(
        0,1,2,3,0,1,2,3,8,9,10,11,8,9,10,11,
        0,1,2,3,0,1,2,3,8,9,10,11,8,9,10,11
    );
    // // 2 3 2 3 6 7 6 7 10 11 10 11 14 15 14 15
    __m256i b_mask = _mm256_setr_epi8(
        4,5,6,7,4,5,6,7,12,13,14,15,12,13,14,15,
        4,5,6,7,4,5,6,7,12,13,14,15,12,13,14,15
    );
    for(int j = 0; j < num_vecs; j++)
    {
        __m256i r = real_vec[j];
        __m256i i = imag_vec[j];

        __m256i ar = _mm256_shuffle_epi8(r, a_mask);
        __m256i ai = _mm256_shuffle_epi8(i, a_mask);
        __m256i br = _mm256_shuffle_epi8(r, b_mask);
        __m256i bi = _mm256_shuffle_epi8(i, b_mask);

        // Complex Multiply: (br + j*bi) * (wr + j*wi)
        // Real: br*wr - bi*wi, Imag: br*wi + bi*wr
        __m256i tr = _mm256_sub_epi16(_mm256_mulhrs_epi16(br, w_real), _mm256_mulhrs_epi16(bi, w_imag));
        __m256i ti = _mm256_add_epi16(_mm256_mulhrs_epi16(br, w_imag), _mm256_mulhrs_epi16(bi, w_real));

        real_vec[j] = _mm256_add_epi16(ar, tr);
        imag_vec[j] = _mm256_add_epi16(ai, ti);
    }
}

// =============================================================
// Stage 3: M = 8 (Butterfly Step = 4)
// =============================================================
static inline void fft_stage_M8(__m256i *real_vec, __m256i *imag_vec, int num_vecs, FFTContext *ctx)
{
    int step = ctx->size / 8;
    // 构造旋转因子 (建议预计算优化)
    __m256i w_real = _mm256_set_epi16(
        -ctx->cos_t[step*3], -ctx->cos_t[step*2], -ctx->cos_t[step*1], -ctx->cos_t[0],
         ctx->cos_t[step*3],  ctx->cos_t[step*2],  ctx->cos_t[step*1],  ctx->cos_t[0],
        -ctx->cos_t[step*3], -ctx->cos_t[step*2], -ctx->cos_t[step*1], -ctx->cos_t[0],
         ctx->cos_t[step*3],  ctx->cos_t[step*2],  ctx->cos_t[step*1],  ctx->cos_t[0]
    );
    __m256i w_imag = _mm256_set_epi16(
         ctx->sin_t[step*3],  ctx->sin_t[step*2],  ctx->sin_t[step*1],  ctx->sin_t[0],
        -ctx->sin_t[step*3], -ctx->sin_t[step*2], -ctx->sin_t[step*1], -ctx->sin_t[0],
         ctx->sin_t[step*3],  ctx->sin_t[step*2],  ctx->sin_t[step*1],  ctx->sin_t[0],
        -ctx->sin_t[step*3], -ctx->sin_t[step*2], -ctx->sin_t[step*1], -ctx->sin_t[0]
    );

    
    // 原代码 M=8 mask 修正直接引用:
    __m256i mask_A = _mm256_setr_epi8(
        0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7,
        0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7
    );
    __m256i mask_B = _mm256_setr_epi8(
        8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15,
        8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15
    );

    for(int j = 0; j < num_vecs; j++)
    {
        __m256i r = real_vec[j];
        __m256i i = imag_vec[j];

        __m256i ar = _mm256_shuffle_epi8(r, mask_A);
        __m256i ai = _mm256_shuffle_epi8(i, mask_A);
        __m256i br = _mm256_shuffle_epi8(r, mask_B);
        __m256i bi = _mm256_shuffle_epi8(i, mask_B);

        __m256i tr = _mm256_sub_epi16(_mm256_mulhrs_epi16(br, w_real), _mm256_mulhrs_epi16(bi, w_imag));
        __m256i ti = _mm256_add_epi16(_mm256_mulhrs_epi16(br, w_imag), _mm256_mulhrs_epi16(bi, w_real));

        real_vec[j] = _mm256_add_epi16(ar, tr);
        imag_vec[j] = _mm256_add_epi16(ai, ti);
    }
}

// =============================================================
// Stage 4: M = 16 (Butterfly Step = 8)
// 跨 128-bit Lane 操作
// =============================================================
static inline void fft_stage_M16(__m256i *real_vec, __m256i *imag_vec, int num_vecs, FFTContext *ctx)
{
    int step = ctx->size / 16;
    // 构造旋转因子
    // 简写示例，需按原逻辑填充 16 个值
    __m256i w_real = _mm256_set_epi16(-ctx->cos_t[step*7],-ctx->cos_t[step*6],-ctx->cos_t[step*5],-ctx->cos_t[step*4],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*7],ctx->cos_t[step*6],ctx->cos_t[step*5],ctx->cos_t[step*4],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[0]);
    __m256i w_imag = _mm256_set_epi16(ctx->sin_t[step*7],ctx->sin_t[step*6],ctx->sin_t[step*5],ctx->sin_t[step*4],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*7],-ctx->sin_t[step*6],-ctx->sin_t[step*5],-ctx->sin_t[step*4],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[0]);

    
    for(int j = 0; j < num_vecs; j++)
    {
        __m256i r = real_vec[j];
        __m256i i = imag_vec[j];

        // Lane 0 -> A, Lane 0 -> A (Lower 128)
        // Lane 1 -> B, Lane 1 -> B (Upper 128)
        // 0x00: Select Lane 0 for low, Lane 0 for high -> A
        // 0x11: Select Lane 1 for low, Lane 1 for high -> B
        __m256i ar = _mm256_permute2x128_si256(r, r, 0x00);
        __m256i ai = _mm256_permute2x128_si256(i, i, 0x00);
        __m256i br = _mm256_permute2x128_si256(r, r, 0x11);
        __m256i bi = _mm256_permute2x128_si256(i, i, 0x11);

        __m256i tr = _mm256_sub_epi16(_mm256_mulhrs_epi16(br, w_real), _mm256_mulhrs_epi16(bi, w_imag));
        __m256i ti = _mm256_add_epi16(_mm256_mulhrs_epi16(br, w_imag), _mm256_mulhrs_epi16(bi, w_real));

        real_vec[j] = _mm256_add_epi16(ar, tr);
        imag_vec[j] = _mm256_add_epi16(ai, ti);
    }
}

// =============================================================
// 主函数
// =============================================================
void fft_AVX_fixedP(int16_t *real, int16_t *imag, int N, FFTContext *ctx)
{
    bit_reverse_q15(real, imag, N, ctx);
    
    __m256i *real_vec = (__m256i *)real;
    __m256i *imag_vec = (__m256i *)imag;
    
    int num_vecs = N / 16;
    int m = 0; 
    // 快速计算 log2(N), 也可以直接从 ctx 中获取如果存在
    int tempN = N; while(tempN >>= 1) m++;

    // -----------------------------------------------------
    // 1. 寄存器内处理阶段 (Intra-register Stages)
    // 直接顺序执行，无循环分支判断
    // -----------------------------------------------------
    
    if (m >= 1) fft_stage_M2(real_vec, imag_vec, num_vecs);
    if (m >= 2) fft_stage_M4(real_vec, imag_vec, num_vecs, ctx);
    if (m >= 3) fft_stage_M8(real_vec, imag_vec, num_vecs, ctx);
    if (m >= 4) fft_stage_M16(real_vec, imag_vec, num_vecs, ctx);

    // -----------------------------------------------------
    // 2. 寄存器间处理阶段 (Inter-register Stages)
    // 统一循环逻辑，M >= 32
    // -----------------------------------------------------
    for (int s = 5; s <= m; s++)
    {
        int M = 1 << s;
        int step_vecs = M / 16;     // 当前Stage的跨度(以向量为单位)
        int half_step = step_vecs / 2; 
        size_t offset_ptr = ctx->stage_offsets[s];

        // 优化外层循环逻辑
        for (int k = 0; k < num_vecs; k += step_vecs)
        {
            // 每次蝶形处理 M/2 个元素，即 half_step 个向量
            for (int j = 0; j < half_step; j++)
            {
                // 加载预计算好的旋转因子
                // 注意：这里的 offset_ptr 逻辑需要确保 ctx->shuffled_xxx 是按照线性访问顺序存储的
                // 原代码中 offset_ptr 在 j 循环内自增，这意味着每一对向量使用一组W
                __m256i w_real = _mm256_load_si256((__m256i*)&ctx->shuffled_cos_t[offset_ptr]);
                __m256i w_imag = _mm256_load_si256((__m256i*)&ctx->shuffled_sin_t[offset_ptr]);
                offset_ptr += 16; 

                // 访问索引
                int idx1 = k + j;
                int idx2 = idx1 + half_step;

                __m256i r1 = real_vec[idx1];
                __m256i i1 = imag_vec[idx1];
                __m256i r2 = real_vec[idx2];
                __m256i i2 = imag_vec[idx2];

                // 蝶形运算
                __m256i tr = _mm256_sub_epi16(_mm256_mulhrs_epi16(r2, w_real), _mm256_mulhrs_epi16(i2, w_imag));
                __m256i ti = _mm256_add_epi16(_mm256_mulhrs_epi16(r2, w_imag), _mm256_mulhrs_epi16(i2, w_real));

                real_vec[idx1] = _mm256_add_epi16(r1, tr);
                imag_vec[idx1] = _mm256_add_epi16(i1, ti);
                real_vec[idx2] = _mm256_sub_epi16(r1, tr);
                imag_vec[idx2] = _mm256_sub_epi16(i1, ti);
            }
        }
    }
}