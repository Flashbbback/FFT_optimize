#include "fft_internal.h"
// 辅助宏：右移1位以进行缩放 (Divide by 2)
// 防止定点数加法溢出，实现 1/N 的归一化
#define SCALE_DOWN(v) _mm256_srai_epi16((v), 1)

// =============================================================
// IFFT Stage 1: M = 2
// 逻辑：Butterfly -> Scale Down
// =============================================================
static inline void ifft_stage_M2(__m256i *real_vec, __m256i *imag_vec, int num_vecs)
{
    static const __m256i even_mask = {
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

        // 蝶形运算
        __m256i sum_r = _mm256_adds_epi16(er, or_val);
        __m256i sum_i = _mm256_adds_epi16(ei, oi);
        __m256i sub_r = _mm256_subs_epi16(er, or_val);
        __m256i sub_i = _mm256_subs_epi16(ei, oi);

        // 【IFFT 关键点】：每一级结束后右移 1 位，防止溢出
        sum_r = SCALE_DOWN(sum_r);
        sum_i = SCALE_DOWN(sum_i);
        sub_r = SCALE_DOWN(sub_r);
        sub_i = SCALE_DOWN(sub_i);

        real_vec[j] = _mm256_unpacklo_epi16(sum_r, sub_r);
        imag_vec[j] = _mm256_unpacklo_epi16(sum_i, sub_i);
    }
}

// =============================================================
// IFFT Stage 2: M = 4
// 逻辑：IFFT Multiply (Conjugate) -> Butterfly -> Scale
// =============================================================
static inline void ifft_stage_M4(__m256i *real_vec, __m256i *imag_vec, int num_vecs)
{
    // 旋转因子与 FFT 保持一致，我们在计算逻辑中取共轭
    __m256i w_real = _mm256_set_epi16(
            0, INT16_MIN, 0, INT16_MAX, 0, INT16_MIN, 0, INT16_MAX,
            0, INT16_MIN, 0, INT16_MAX, 0, INT16_MIN, 0, INT16_MAX);
    __m256i w_imag = _mm256_set_epi16( 
            INT16_MAX, 0, INT16_MIN, 0, INT16_MAX, 0, INT16_MIN, 0,
            INT16_MAX, 0, INT16_MIN, 0, INT16_MAX, 0, INT16_MIN, 0);

    __m256i a_mask = _mm256_setr_epi8(
        0,1,2,3,0,1,2,3,8,9,10,11,8,9,10,11,
        0,1,2,3,0,1,2,3,8,9,10,11,8,9,10,11);
    __m256i b_mask = _mm256_setr_epi8(
        4,5,6,7,4,5,6,7,12,13,14,15,12,13,14,15,
        4,5,6,7,4,5,6,7,12,13,14,15,12,13,14,15);

    for(int j = 0; j < num_vecs; j++)
    {
        __m256i r = real_vec[j];
        __m256i i = imag_vec[j];

        __m256i ar = _mm256_shuffle_epi8(r, a_mask);
        __m256i ai = _mm256_shuffle_epi8(i, a_mask);
        __m256i br = _mm256_shuffle_epi8(r, b_mask);
        __m256i bi = _mm256_shuffle_epi8(i, b_mask);

        // 【IFFT 关键点】：复数乘法共轭
        // (br + j*bi) * (wr - j*wi)^-1 = (br + j*bi) * (wr + j*wi)
        // Real: br*wr - bi*wi  (FFT: sub, IFFT: sub? No.)
        // IFFT Math: (A+jB)(C+jD) = (AC-BD) + j(AD+BC). 
        // 但这里的 W 是 FFT 的 W (通常是 cos - j*sin)。
        // 所以 IFFT 需要乘 (cos + j*sin)。
        // Real: br*wr - bi*wi -> br*wr - bi*(-wi) -> br*wr + bi*wi
        // Imag: br*wi + bi*wr -> br*(-wi) + bi*wr -> bi*wr - br*wi
        
        __m256i mul_rr = _mm256_mulhrs_epi16(br, w_real);
        __m256i mul_ii = _mm256_mulhrs_epi16(bi, w_imag);
        __m256i mul_ri = _mm256_mulhrs_epi16(br, w_imag);
        __m256i mul_ir = _mm256_mulhrs_epi16(bi, w_real);

        // FFT: tr = rr - ii; ti = ri + ir;
        // IFFT: tr = rr + ii; ti = ir - ri;
        __m256i tr = _mm256_adds_epi16(mul_rr, mul_ii); 
        __m256i ti = _mm256_subs_epi16(mul_ir, mul_ri); // 注意顺序：bi*wr - br*wi

        // 蝶形 + 缩放
        real_vec[j] = SCALE_DOWN(_mm256_adds_epi16(ar, tr));
        imag_vec[j] = SCALE_DOWN(_mm256_adds_epi16(ai, ti));
    }
}

// =============================================================
// IFFT Stage 3: M = 8
// =============================================================
static inline void ifft_stage_M8(__m256i *real_vec, __m256i *imag_vec, int num_vecs)
{
    // 复用 FFT 的旋转因子表
    __m256i w_real = _mm256_set_epi16(
        Q15_SQRT2_2, 0, -Q15_SQRT2_2, INT16_MIN,
        -Q15_SQRT2_2, 0, Q15_SQRT2_2, INT16_MAX,
        Q15_SQRT2_2, 0, -Q15_SQRT2_2, INT16_MIN,
        -Q15_SQRT2_2, 0, Q15_SQRT2_2, INT16_MAX
    );
    __m256i w_imag = _mm256_set_epi16(
        Q15_SQRT2_2, INT16_MAX, Q15_SQRT2_2, 0,
        -Q15_SQRT2_2, INT16_MIN, -Q15_SQRT2_2, -0,
        Q15_SQRT2_2, INT16_MAX, Q15_SQRT2_2, 0,
        -Q15_SQRT2_2, INT16_MIN, -Q15_SQRT2_2, -0
    );

    __m256i mask_A = _mm256_setr_epi8(
        0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7);
    __m256i mask_B = _mm256_setr_epi8(
        8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15);

    for(int j = 0; j < num_vecs; j++)
    {
        __m256i r = real_vec[j];
        __m256i i = imag_vec[j];

        __m256i ar = _mm256_shuffle_epi8(r, mask_A);
        __m256i ai = _mm256_shuffle_epi8(i, mask_A);
        __m256i br = _mm256_shuffle_epi8(r, mask_B);
        __m256i bi = _mm256_shuffle_epi8(i, mask_B);

        // IFFT 乘法逻辑
        __m256i tr = _mm256_adds_epi16(_mm256_mulhrs_epi16(br, w_real), _mm256_mulhrs_epi16(bi, w_imag));
        __m256i ti = _mm256_subs_epi16(_mm256_mulhrs_epi16(bi, w_real), _mm256_mulhrs_epi16(br, w_imag));
        

        real_vec[j] = SCALE_DOWN(_mm256_adds_epi16(ar, tr));
        imag_vec[j] = SCALE_DOWN(_mm256_adds_epi16(ai, ti));
    }
}

// =============================================================
// IFFT Stage 4: M = 16
// =============================================================
static inline void ifft_stage_M16(__m256i *real_vec, __m256i *imag_vec, int num_vecs)
{
    // 这里省略具体的常量定义，请保持与 FFT 代码中的 w_real/w_imag 一致
    // 只要 W 是标准的 FFT 旋转因子，下面的 IFFT 乘法逻辑就是正确的
    
    // ... (复制 FFT 中的 w_real, w_imag 定义) ...
    // 为演示完整性，请确保这里填入 FFT 代码中的 Stage 4 常量
     __m256i w_real = _mm256_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/16*7)),FLOAT_TO_Q15(-cosf(2*M_PI/16*6)),FLOAT_TO_Q15(-cosf(2*M_PI/16*5)),FLOAT_TO_Q15(-cosf(2*M_PI/16*4)),FLOAT_TO_Q15(-cosf(2*M_PI/16*3)),FLOAT_TO_Q15(-cosf(2*M_PI/16*2)),FLOAT_TO_Q15(-cosf(2*M_PI/16*1)),FLOAT_TO_Q15(-cosf(2*M_PI/16*0)),FLOAT_TO_Q15(cosf(2*M_PI/16*7)),FLOAT_TO_Q15(cosf(2*M_PI/16*6)),FLOAT_TO_Q15(cosf(2*M_PI/16*5)),FLOAT_TO_Q15(cosf(2*M_PI/16*4)),FLOAT_TO_Q15(cosf(2*M_PI/16*3)),FLOAT_TO_Q15(cosf(2*M_PI/16*2)),FLOAT_TO_Q15(cosf(2*M_PI/16*1)),FLOAT_TO_Q15(cosf(2*M_PI/16*0)));
    __m256i w_imag = _mm256_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/16*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*0)),FLOAT_TO_Q15(sinf(-2*M_PI/16*7)),FLOAT_TO_Q15(sinf(-2*M_PI/16*6)),FLOAT_TO_Q15(sinf(-2*M_PI/16*5)),FLOAT_TO_Q15(sinf(-2*M_PI/16*4)),FLOAT_TO_Q15(sinf(-2*M_PI/16*3)),FLOAT_TO_Q15(sinf(-2*M_PI/16*2)),FLOAT_TO_Q15(sinf(-2*M_PI/16*1)),FLOAT_TO_Q15(sinf(-2*M_PI/16*0)));


    for(int j = 0; j < num_vecs; j++)
    {
        __m256i r = real_vec[j];
        __m256i i = imag_vec[j];

        __m256i ar = _mm256_permute2x128_si256(r, r, 0x00);
        __m256i ai = _mm256_permute2x128_si256(i, i, 0x00);
        __m256i br = _mm256_permute2x128_si256(r, r, 0x11);
        __m256i bi = _mm256_permute2x128_si256(i, i, 0x11);

        // IFFT Multiply
        __m256i tr = _mm256_adds_epi16(_mm256_mulhrs_epi16(br, w_real), _mm256_mulhrs_epi16(bi, w_imag));
        __m256i ti = _mm256_subs_epi16(_mm256_mulhrs_epi16(bi, w_real), _mm256_mulhrs_epi16(br, w_imag));

        real_vec[j] = SCALE_DOWN(_mm256_adds_epi16(ar, tr));
        imag_vec[j] = SCALE_DOWN(_mm256_adds_epi16(ai, ti));
    }
}

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
// IFFT 主函数
// =============================================================
void ifft_AVX_fixedP(int16_t *real, int16_t *imag, int N, FFTContext *ctx)
{
    // 1. 输入倒位序重排 (与 FFT 相同)
    bit_reverse_q15(real, imag, N, ctx);
    
    __m256i *real_vec = (__m256i *)real;
    __m256i *imag_vec = (__m256i *)imag;
    
    int num_vecs = N / 16;
    int m = 0; 
    int tempN = N; while(tempN >>= 1) m++;

    // 2. 级内处理 (Intra-register)
    if (m >= 1) ifft_stage_M2(real_vec, imag_vec, num_vecs);
    if (m >= 2) ifft_stage_M4(real_vec, imag_vec, num_vecs);
    if (m >= 3) ifft_stage_M8(real_vec, imag_vec, num_vecs);
    if (m >= 4) ifft_stage_M16(real_vec, imag_vec, num_vecs);

    // 3. 级间处理 (Inter-register)
    for (int s = 5; s <= m; s++)
    {
        int M = 1 << s;
        int step_vecs = M / 16;
        int half_step = step_vecs / 2; 
        size_t offset_ptr = ctx->stage_offsets[s];

        for (int k = 0; k < num_vecs; k += step_vecs)
        {
            for (int j = 0; j < half_step; j++)
            {
                // 加载 FFT 的旋转因子 (cos, -sin)
                __m256i w_real = _mm256_load_si256((__m256i*)&ctx->shuffled_cos_t[offset_ptr]);
                __m256i w_imag = _mm256_load_si256((__m256i*)&ctx->shuffled_sin_t[offset_ptr]);
                offset_ptr += 16; 

                int idx1 = k + j;
                int idx2 = idx1 + half_step;

                __m256i r1 = real_vec[idx1];
                __m256i i1 = imag_vec[idx1];
                __m256i r2 = real_vec[idx2];
                __m256i i2 = imag_vec[idx2];

                // IFFT 核心修改：复数乘法逻辑翻转 (假设使用 FFT 的 W)
                // Real = r2*wr + i2*wi
                // Imag = i2*wr - r2*wi
                __m256i tr = _mm256_adds_epi16(_mm256_mulhrs_epi16(r2, w_real), _mm256_mulhrs_epi16(i2, w_imag));
                __m256i ti = _mm256_subs_epi16(_mm256_mulhrs_epi16(i2, w_real), _mm256_mulhrs_epi16(r2, w_imag));

                // 蝶形运算 + 缩放 (>> 1)
                real_vec[idx1] = SCALE_DOWN(_mm256_adds_epi16(r1, tr));
                imag_vec[idx1] = SCALE_DOWN(_mm256_adds_epi16(i1, ti));
                
                real_vec[idx2] = SCALE_DOWN(_mm256_subs_epi16(r1, tr));
                imag_vec[idx2] = SCALE_DOWN(_mm256_subs_epi16(i1, ti));
            }
        }
    }
}


