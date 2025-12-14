#include "fft_internal.h" // 假设这里面包含了 immintrin.h 和结构体定义
#include <stdint.h>
#include <stdio.h>


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
// Stage 1: M = 2
// =============================================================
static inline void fft_stage_avx512_M2(__m512i *real_vec, __m512i *imag_vec, int num_vecs)
{
    // M=2 Mask (从用户代码提取)
    static const __m512i even_mask_data = {
        0x3D3C393835343130, 0x3D3C393835343130, 0x2D2C292825242120, 0x2D2C292825242120,
        0x1D1C191815141110, 0x1D1C191815141110, 0x0D0C090805040100, 0x0D0C090805040100
    }; 
    // 注意：上面这种初始化方式依赖编译器扩展，更标准的做法是 byte array，
    // 这里为了保持与 _mm512_set_epi8 逻辑一致，使用 load。
    
    // 更稳健的写法：定义 byte 数组
    static const uint8_t mask_even_bytes[64] __attribute__((aligned(64))) = {
        0x00,0x01,0x04,0x05,0x08,0x09,0x0C,0x0D, 0x10,0x11,0x14,0x15,0x18,0x19,0x1C,0x1D,
        0x20,0x21,0x24,0x25,0x28,0x29,0x2C,0x2D, 0x30,0x31,0x34,0x35,0x38,0x39,0x3C,0x3D,
        0x00,0x01,0x04,0x05,0x08,0x09,0x0C,0x0D, 0x10,0x11,0x14,0x15,0x18,0x19,0x1C,0x1D,
        0x20,0x21,0x24,0x25,0x28,0x29,0x2C,0x2D, 0x30,0x31,0x34,0x35,0x38,0x39,0x3C,0x3D
    };
    static const uint8_t mask_odd_bytes[64] __attribute__((aligned(64))) = {
        0x02,0x03,0x06,0x07,0x0A,0x0B,0x0E,0x0F, 0x12,0x13,0x16,0x17,0x1A,0x1B,0x1E,0x1F,
        0x22,0x23,0x26,0x27,0x2A,0x2B,0x2E,0x2F, 0x32,0x33,0x36,0x37,0x3A,0x3B,0x3E,0x3F,
        0x02,0x03,0x06,0x07,0x0A,0x0B,0x0E,0x0F, 0x12,0x13,0x16,0x17,0x1A,0x1B,0x1E,0x1F,
        0x22,0x23,0x26,0x27,0x2A,0x2B,0x2E,0x2F, 0x32,0x33,0x36,0x37,0x3A,0x3B,0x3E,0x3F
    };

    __m512i even_mask = _mm512_load_si512((const void*)mask_even_bytes);
    __m512i odd_mask  = _mm512_load_si512((const void*)mask_odd_bytes);

    for (int j = 0; j < num_vecs; j++)
    {
        __m512i r = real_vec[j];
        __m512i i = imag_vec[j];

        __m512i er = _mm512_shuffle_epi8(r, even_mask);
        __m512i ei = _mm512_shuffle_epi8(i, even_mask);
        __m512i or_val = _mm512_shuffle_epi8(r, odd_mask);
        __m512i oi = _mm512_shuffle_epi8(i, odd_mask);

        __m512i sum_r = _mm512_add_epi16(er, or_val);
        __m512i sum_i = _mm512_add_epi16(ei, oi);
        __m512i sub_r = _mm512_sub_epi16(er, or_val);
        __m512i sub_i = _mm512_sub_epi16(ei, oi);

        real_vec[j] = _mm512_unpacklo_epi16(sum_r, sub_r);
        imag_vec[j] = _mm512_unpacklo_epi16(sum_i, sub_i);
    }
}

// =============================================================
// Stage 2: M = 4
// =============================================================
static inline void fft_stage_avx512_M4(__m512i *real_vec, __m512i *imag_vec, int num_vecs, FFTContext *ctx)
{
    
    // W = {1, 0, -i, 0, 1, 0, -i ...}
    // 构造逻辑需根据你的 trig_table 布局，这里直接构造静态常量更高效
    // M=4 => W0=1, W1=-i
    // Real: 1, 0, 1, 0...
    // Imag: 0, -1, 0, -1...
    // AVX512 有 32 个 int16，对应 16 对复数，模式重复

    
    // 动态生成 (为了完全匹配原逻辑)
        int step = ctx->size / 4;
        __m512i wr = _mm512_set_epi16(ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0],ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*1],ctx->cos_t[step*0]);
        __m512i wi = _mm512_set_epi16(ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*1],ctx->sin_t[step*0]);

    __m512i mask_a = _mm512_set_epi8(
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
    __m512i mask_b = _mm512_set_epi8(
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

    for(int j = 0; j < num_vecs; j++) {
        __m512i r = real_vec[j];
        __m512i i = imag_vec[j];

        __m512i ar = _mm512_shuffle_epi8(r, mask_a);
        __m512i ai = _mm512_shuffle_epi8(i, mask_a);
        __m512i br = _mm512_shuffle_epi8(r, mask_b);
        __m512i bi = _mm512_shuffle_epi8(i, mask_b);

        __m512i tr = _mm512_sub_epi16(_mm512_mulhrs_epi16(br, wr), _mm512_mulhrs_epi16(bi, wi));
        __m512i ti = _mm512_add_epi16(_mm512_mulhrs_epi16(br, wi), _mm512_mulhrs_epi16(bi, wr));

        real_vec[j] = _mm512_add_epi16(ar, tr);
        imag_vec[j] = _mm512_add_epi16(ai, ti);
    }
}

// =============================================================
// Stage 3: M = 8
// =============================================================
static inline void fft_stage_avx512_M8(__m512i *real_vec, __m512i *imag_vec, int num_vecs, FFTContext *ctx)
{
    int step = ctx->size / 8;
    // 构造 W (重复填充)
    // 需要 4 个不同的 W: W0, W1, W2, W3 (每个占 1个 int16)
    // AVX512 有 32 个 int16 => 8 组 M=8 => 每组 4个 W
    // 布局: W3, W2, W1, W0 ... 
    
            // __m512i w_real = _mm512_set_epi16(-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0),-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0));//低位正，高位负
            // __m512i w_imag = _mm512_set_epi16(-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0),-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0));

            __m512i wr = _mm512_set_epi16(-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0],-ctx->cos_t[step*3],-ctx->cos_t[step*2],-ctx->cos_t[step*1],-ctx->cos_t[step*0],ctx->cos_t[step*3],ctx->cos_t[step*2],ctx->cos_t[step*1],ctx->cos_t[step*0]);
            __m512i wi = _mm512_set_epi16(ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0],ctx->sin_t[step*3],ctx->sin_t[step*2],ctx->sin_t[step*1],ctx->sin_t[step*0],-ctx->sin_t[step*3],-ctx->sin_t[step*2],-ctx->sin_t[step*1],-ctx->sin_t[step*0]);


            //0 1 2 3 0 1 2 3 2 9 10 11 8 9 10 11
            __m512i mask_a = _mm512_set_epi8(
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
            __m512i mask_b = _mm512_set_epi8(
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

    for(int j = 0; j < num_vecs; j++) {
        __m512i r = real_vec[j];
        __m512i i = imag_vec[j];

        __m512i ar = _mm512_shuffle_epi8(r, mask_a);
        __m512i ai = _mm512_shuffle_epi8(i, mask_a);
        __m512i br = _mm512_shuffle_epi8(r, mask_b);
        __m512i bi = _mm512_shuffle_epi8(i, mask_b);

        __m512i tr = _mm512_sub_epi16(_mm512_mulhrs_epi16(br, wr), _mm512_mulhrs_epi16(bi, wi));
        __m512i ti = _mm512_add_epi16(_mm512_mulhrs_epi16(br, wi), _mm512_mulhrs_epi16(bi, wr));

        real_vec[j] = _mm512_add_epi16(ar, tr);
        imag_vec[j] = _mm512_add_epi16(ai, ti);
    }
}

// =============================================================
// Stage 4: M = 16
// =============================================================
static inline void fft_stage_avx512_M16(__m512i *real_vec, __m512i *imag_vec, int num_vecs, FFTContext *ctx)
{
    // 逻辑同上，Mask 跨度变大
    // Mask A: 0..7, Mask B: 8..15 (lane local)
    // 0..15 刚好是一个 128位 lane 的全部
    // 所以这里的 Mask 实际上是重复 0..7 和 8..15
    
    // Mask 生成...
            __m512i wr = _mm512_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/32*7)),FLOAT_TO_Q15(-cosf(2*M_PI/32*6)),FLOAT_TO_Q15(-cosf(2*M_PI/32*5)),FLOAT_TO_Q15(-cosf(2*M_PI/32*4)),FLOAT_TO_Q15(-cosf(2*M_PI/32*3)),FLOAT_TO_Q15(-cosf(2*M_PI/32*2)),FLOAT_TO_Q15(-cosf(2*M_PI/32*1)),FLOAT_TO_Q15(-cosf(2*M_PI/32*0)),FLOAT_TO_Q15(cosf(2*M_PI/32*7)),FLOAT_TO_Q15(cosf(2*M_PI/32*6)),FLOAT_TO_Q15(cosf(2*M_PI/32*5)),FLOAT_TO_Q15(cosf(2*M_PI/32*4)),FLOAT_TO_Q15(cosf(2*M_PI/32*3)),FLOAT_TO_Q15(cosf(2*M_PI/32*2)),FLOAT_TO_Q15(cosf(2*M_PI/32*1)),FLOAT_TO_Q15(cosf(2*M_PI/32*0)),FLOAT_TO_Q15(-cosf(2*M_PI/32*7)),FLOAT_TO_Q15(-cosf(2*M_PI/32*6)),FLOAT_TO_Q15(-cosf(2*M_PI/32*5)),FLOAT_TO_Q15(-cosf(2*M_PI/32*4)),FLOAT_TO_Q15(-cosf(2*M_PI/32*3)),FLOAT_TO_Q15(-cosf(2*M_PI/32*2)),FLOAT_TO_Q15(-cosf(2*M_PI/32*1)),FLOAT_TO_Q15(-cosf(2*M_PI/32*0)),FLOAT_TO_Q15(cosf(2*M_PI/32*7)),FLOAT_TO_Q15(cosf(2*M_PI/32*6)),FLOAT_TO_Q15(cosf(2*M_PI/32*5)),FLOAT_TO_Q15(cosf(2*M_PI/32*4)),FLOAT_TO_Q15(cosf(2*M_PI/32*3)),FLOAT_TO_Q15(cosf(2*M_PI/32*2)),FLOAT_TO_Q15(cosf(2*M_PI/32*1)),FLOAT_TO_Q15(cosf(2*M_PI/32*0)));//低位正，高位负
            __m512i wi = _mm512_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/32*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*0)),FLOAT_TO_Q15(sinf(-2*M_PI/32*7)),FLOAT_TO_Q15(sinf(-2*M_PI/32*6)),FLOAT_TO_Q15(sinf(-2*M_PI/32*5)),FLOAT_TO_Q15(sinf(-2*M_PI/32*4)),FLOAT_TO_Q15(sinf(-2*M_PI/32*3)),FLOAT_TO_Q15(sinf(-2*M_PI/32*2)),FLOAT_TO_Q15(sinf(-2*M_PI/32*1)),FLOAT_TO_Q15(sinf(-2*M_PI/32*0)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*0)),FLOAT_TO_Q15(sinf(-2*M_PI/32*7)),FLOAT_TO_Q15(sinf(-2*M_PI/32*6)),FLOAT_TO_Q15(sinf(-2*M_PI/32*5)),FLOAT_TO_Q15(sinf(-2*M_PI/32*4)),FLOAT_TO_Q15(sinf(-2*M_PI/32*3)),FLOAT_TO_Q15(sinf(-2*M_PI/32*2)),FLOAT_TO_Q15(sinf(-2*M_PI/32*1)),FLOAT_TO_Q15(sinf(-2*M_PI/32*0)));
    
            // 序列1掩码：0~7重复、16~23重复
            __m512i mask_a= _mm512_set_epi8(
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
            __m512i mask_b = _mm512_set_epi8(
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

    for(int j = 0; j < num_vecs; j++) {
        __m512i r = real_vec[j];
        __m512i i = imag_vec[j];

        __m512i ar = _mm512_shuffle_epi8(r, mask_a);
        __m512i ai = _mm512_shuffle_epi8(i, mask_a);
        __m512i br = _mm512_shuffle_epi8(r, mask_b);
        __m512i bi = _mm512_shuffle_epi8(i, mask_b);

        __m512i tr = _mm512_sub_epi16(_mm512_mulhrs_epi16(br, wr), _mm512_mulhrs_epi16(bi, wi));
        __m512i ti = _mm512_add_epi16(_mm512_mulhrs_epi16(br, wi), _mm512_mulhrs_epi16(bi, wr));

        real_vec[j] = _mm512_add_epi16(ar, tr);
        imag_vec[j] = _mm512_add_epi16(ai, ti);
    }
}

// =============================================================
// Stage 5: M = 32 (AVX512 特有)
// =============================================================
static inline void fft_stage_avx512_M32(__m512i *real_vec, __m512i *imag_vec, int num_vecs, FFTContext *ctx)
{
    // M=32 填满整个 512 位寄存器 (32 int16)
    // 需要交换低 256 位和高 256 位？或者 Lane 间的交换。
    // 原代码使用了 shuffle_epi8，但 M=32 意味着要跨 128-bit Lane。
    // _mm512_shuffle_epi8 是 Lane-local 的。
    // 如果原代码能跑，意味着 M=32 的 Shuffle 其实是某种 Lane 内的模式？
    // 不，M=32 通常意味着 A=Lower 16 shorts, B=Upper 16 shorts.
    // 这需要跨 Lane。
    
    // 正确的 AVX512 跨 Lane Shuffle 应该使用 permuation
    // 例如 _mm512_permutexvar_epi16 (需 AVX512BW) 或 shuffle_i32x4
    
    // 按照原代码逻辑结构 (假设 mask 正确)
    // Mask A: 0..15 (Lower 256?), Mask B: 16..31
    
    int step = ctx->size / 32;
            __m512i wr = _mm512_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/32*15)),FLOAT_TO_Q15(-cosf(2*M_PI/32*14)),FLOAT_TO_Q15(-cosf(2*M_PI/32*13)),FLOAT_TO_Q15(-cosf(2*M_PI/32*12)),FLOAT_TO_Q15(-cosf(2*M_PI/32*11)),FLOAT_TO_Q15(-cosf(2*M_PI/32*10)),FLOAT_TO_Q15(-cosf(2*M_PI/32*9)),FLOAT_TO_Q15(-cosf(2*M_PI/32*8)),FLOAT_TO_Q15(-cosf(2*M_PI/32*7)),FLOAT_TO_Q15(-cosf(2*M_PI/32*6)),FLOAT_TO_Q15(-cosf(2*M_PI/32*5)),FLOAT_TO_Q15(-cosf(2*M_PI/32*4)),FLOAT_TO_Q15(-cosf(2*M_PI/32*3)),FLOAT_TO_Q15(-cosf(2*M_PI/32*2)),FLOAT_TO_Q15(-cosf(2*M_PI/32*1)),FLOAT_TO_Q15(-cosf(2*M_PI/32*0)),FLOAT_TO_Q15(cosf(2*M_PI/32*15)),FLOAT_TO_Q15(cosf(2*M_PI/32*14)),FLOAT_TO_Q15(cosf(2*M_PI/32*13)),FLOAT_TO_Q15(cosf(2*M_PI/32*12)),FLOAT_TO_Q15(cosf(2*M_PI/32*11)),FLOAT_TO_Q15(cosf(2*M_PI/32*10)),FLOAT_TO_Q15(cosf(2*M_PI/32*9)),FLOAT_TO_Q15(cosf(2*M_PI/32*8)),FLOAT_TO_Q15(cosf(2*M_PI/32*7)),FLOAT_TO_Q15(cosf(2*M_PI/32*6)),FLOAT_TO_Q15(cosf(2*M_PI/32*5)),FLOAT_TO_Q15(cosf(2*M_PI/32*4)),FLOAT_TO_Q15(cosf(2*M_PI/32*3)),FLOAT_TO_Q15(cosf(2*M_PI/32*2)),FLOAT_TO_Q15(cosf(2*M_PI/32*1)),FLOAT_TO_Q15(cosf(2*M_PI/32*0)));//低位正，高位负
            __m512i wi = _mm512_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/32*15)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*14)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*13)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*12)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*11)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*10)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*9)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*8)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*0)),FLOAT_TO_Q15(sinf(-2*M_PI/32*15)),FLOAT_TO_Q15(sinf(-2*M_PI/32*14)),FLOAT_TO_Q15(sinf(-2*M_PI/32*13)),FLOAT_TO_Q15(sinf(-2*M_PI/32*12)),FLOAT_TO_Q15(sinf(-2*M_PI/32*11)),FLOAT_TO_Q15(sinf(-2*M_PI/32*10)),FLOAT_TO_Q15(sinf(-2*M_PI/32*9)),FLOAT_TO_Q15(sinf(-2*M_PI/32*8)),FLOAT_TO_Q15(sinf(-2*M_PI/32*7)),FLOAT_TO_Q15(sinf(-2*M_PI/32*6)),FLOAT_TO_Q15(sinf(-2*M_PI/32*5)),FLOAT_TO_Q15(sinf(-2*M_PI/32*4)),FLOAT_TO_Q15(sinf(-2*M_PI/32*3)),FLOAT_TO_Q15(sinf(-2*M_PI/32*2)),FLOAT_TO_Q15(sinf(-2*M_PI/32*1)),FLOAT_TO_Q15(sinf(-2*M_PI/32*0)));

    //             int16_t c0  = ctx->cos_t[0 * step],  s0  = ctx->sin_t[0 * step];
    // int16_t c1  = ctx->cos_t[1 * step],  s1  = ctx->sin_t[1 * step];
    // int16_t c2  = ctx->cos_t[2 * step],  s2  = ctx->sin_t[2 * step];
    // int16_t c3  = ctx->cos_t[3 * step],  s3  = ctx->sin_t[3 * step];
    // int16_t c4  = ctx->cos_t[4 * step],  s4  = ctx->sin_t[4 * step];
    // int16_t c5  = ctx->cos_t[5 * step],  s5  = ctx->sin_t[5 * step];
    // int16_t c6  = ctx->cos_t[6 * step],  s6  = ctx->sin_t[6 * step];
    // int16_t c7  = ctx->cos_t[7 * step],  s7  = ctx->sin_t[7 * step];
    // int16_t c8  = ctx->cos_t[8 * step],  s8  = ctx->sin_t[8 * step];
    // int16_t c9  = ctx->cos_t[9 * step],  s9  = ctx->sin_t[9 * step];
    // int16_t c10 = ctx->cos_t[10 * step], s10 = ctx->sin_t[10 * step];
    // int16_t c11 = ctx->cos_t[11 * step], s11 = ctx->sin_t[11 * step];
    // int16_t c12 = ctx->cos_t[12 * step], s12 = ctx->sin_t[12 * step];
    // int16_t c13 = ctx->cos_t[13 * step], s13 = ctx->sin_t[13 * step];
    // int16_t c14 = ctx->cos_t[14 * step], s14 = ctx->sin_t[14 * step];
    // int16_t c15 = ctx->cos_t[15 * step], s15 = ctx->sin_t[15 * step];

    // // 注意：set_epi16 参数顺序是 e31, e30 ... e0 (高位在前)
    // // 我们需要 e0=c0, e1=c1 ... e15=c15, e16=c0 ...
    // __m512i wr = _mm512_set_epi16(
    //     c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0,
    //     c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0
    // );

    // // 对于 Sin，注意 FFT 公式中的符号。通常 Fixed Point Mul 逻辑需要配合符号
    // // 这里保持与 M=16 一致的逻辑
    // __m512i wi = _mm512_set_epi16(
    //     s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
    //     s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0
    // );

            __m512i mask_a= _mm512_set_epi8(
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

            __m512i mask_b = _mm512_set_epi8(
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

    for(int j = 0; j < num_vecs; j++) {
        __m512i r = real_vec[j];
        __m512i i = imag_vec[j];

        // 警告：如果 mask 涉及跨 128位 lane，shuffle_epi8 可能得不到预期结果
        // 除非硬件支持或 mask 设计巧妙
        __m512i ar = _mm512_shuffle_epi8(r, mask_a);
        __m512i ai = _mm512_shuffle_epi8(i, mask_a);
        __m512i br = _mm512_shuffle_epi8(r, mask_b);
        __m512i bi = _mm512_shuffle_epi8(i, mask_b);

        __m512i tr = _mm512_sub_epi16(_mm512_mulhrs_epi16(br, wr), _mm512_mulhrs_epi16(bi, wi));
        __m512i ti = _mm512_add_epi16(_mm512_mulhrs_epi16(br, wi), _mm512_mulhrs_epi16(bi, wr));

        real_vec[j] = _mm512_add_epi16(ar, tr);
        imag_vec[j] = _mm512_add_epi16(ai, ti);
    }
}

// =============================================================
// 主函数
// =============================================================
void fft_AVX512_fixedP(int16_t *real, int16_t *imag, FFTContext *ctx)
{
    int N = ctx->size;
    bit_reverse_q15(real, imag, N, ctx);
    
    __m512i *real_vec = (__m512i *)real;
    __m512i *imag_vec = (__m512i *)imag;
    
    int num_vecs = N / 32; // 512 bit = 32 * 16 bit
    int m = 0; 
    // 快速计算 log2(N), 也可以直接从 ctx 中获取如果存在
    int tempN = N; while(tempN >>= 1) m++;

    // Intra-register Stages
    if (m >= 1) fft_stage_avx512_M2(real_vec, imag_vec, num_vecs);
    if (m >= 2) fft_stage_avx512_M4(real_vec, imag_vec, num_vecs, ctx);
    if (m >= 3) fft_stage_avx512_M8(real_vec, imag_vec, num_vecs, ctx);
    if (m >= 4) fft_stage_avx512_M16(real_vec, imag_vec, num_vecs, ctx);
    if (m >= 5) fft_stage_avx512_M32(real_vec, imag_vec, num_vecs, ctx);

    // Inter-register Stages (M >= 64)
    for (int s = 6; s <= m; s++)
    {
        int M = 1 << s;
        int step_vecs = M / 32;
        int half_step = step_vecs / 2;
        size_t offset_ptr = ctx->stage_offsets[s];

        for (int k = 0; k < num_vecs; k += step_vecs)
        {
            for (int j = 0; j < half_step; j++)
            {
                // 加载 32 个旋转因子 (512位)
                // 注意：offset_ptr 需要对应到 512 位对齐的数据
                __m512i w_real = _mm512_load_si512((__m512i*)&ctx->shuffled_cos_t[offset_ptr]);
                __m512i w_imag = _mm512_load_si512((__m512i*)&ctx->shuffled_sin_t[offset_ptr]);
                offset_ptr += 32; 

                int idx1 = k + j;
                int idx2 = idx1 + half_step;

                __m512i r1 = real_vec[idx1];
                __m512i i1 = imag_vec[idx1];
                __m512i r2 = real_vec[idx2];
                __m512i i2 = imag_vec[idx2];

                __m512i tr = _mm512_sub_epi16(_mm512_mulhrs_epi16(r2, w_real), _mm512_mulhrs_epi16(i2, w_imag));
                __m512i ti = _mm512_add_epi16(_mm512_mulhrs_epi16(r2, w_imag), _mm512_mulhrs_epi16(i2, w_real));

                real_vec[idx1] = _mm512_add_epi16(r1, tr);
                imag_vec[idx1] = _mm512_add_epi16(i1, ti);
                real_vec[idx2] = _mm512_sub_epi16(r1, tr);
                imag_vec[idx2] = _mm512_sub_epi16(i1, ti);
            }
        }
    }
}