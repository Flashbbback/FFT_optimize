#include "fft_internal.h"
// 辅助宏：右移1位以进行缩放 (Divide by 2)
// 防止定点数加法溢出，实现 1/N 的归一化
#define SCALE_DOWN(v) _mm512_srai_epi16((v), 1)



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
static inline void ifft_stage_avx512_M2(__m512i *real_vec, __m512i *imag_vec, int num_vecs)
{
    // M=2 Mask (从用户代码提取)
    // 注意：上面这种初始化方式依赖编译器扩展，更标准的做法是 byte array，
    // 这里为了保持与 _mm512_set_epi8 逻辑一致，使用 load。
    
    // 更稳健的写法：定义 byte 数组
    //
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

        //0+1 2+3 4+5 6+7 ...
        __m512i sum_r = _mm512_adds_epi16(er, or_val);
        __m512i sum_i = _mm512_adds_epi16(ei, oi);
        __m512i sub_r = _mm512_subs_epi16(er, or_val);
        __m512i sub_i = _mm512_subs_epi16(ei, oi);

        // 【IFFT 关键点】：每一级结束后右移 1 位，防止溢出
        sum_r = SCALE_DOWN(sum_r);
        sum_i = SCALE_DOWN(sum_i);
        sub_r = SCALE_DOWN(sub_r);
        sub_i = SCALE_DOWN(sub_i);

        real_vec[j] = _mm512_unpacklo_epi16(sum_r, sub_r);
        imag_vec[j] = _mm512_unpacklo_epi16(sum_i, sub_i);
    }
}

// =============================================================
// Stage 2: M = 4
// =============================================================
static inline void ifft_stage_avx512_M4(__m512i *real_vec, __m512i *imag_vec, int num_vecs)
{
    
    // W = {1, 0, -i, 0, 1, 0, -i ...}
    // 构造逻辑需根据你的 trig_table 布局，这里直接构造静态常量更高效
    // M=4 => W0=1, W1=-i
    // Real: 1, 0, 1, 0...
    // Imag: 0, -1, 0, -1...
    // AVX512 有 32 个 int16，对应 16 对复数，模式重复


        // 构造 wr (旋转因子实部)：模式为 [0, -1, 0, 1] 重复16次（32个int16）
        __m512i wr = _mm512_set_epi16(
            0, INT16_MIN, 0, INT16_MAX,
            0, INT16_MIN, 0, INT16_MAX,
            0, INT16_MIN, 0, INT16_MAX,
            0, INT16_MIN, 0, INT16_MAX,
            0, INT16_MIN, 0, INT16_MAX,
            0, INT16_MIN, 0, INT16_MAX,
            0, INT16_MIN, 0, INT16_MAX,
            0, INT16_MIN, 0, INT16_MAX
        );

        // 构造 wi (旋转因子虚部)：模式为 [1, 0, -1, 0] 重复16次（32个int16）
        __m512i wi = _mm512_set_epi16(
            INT16_MAX, 0, INT16_MIN, 0,
            INT16_MAX, 0, INT16_MIN, 0,
            INT16_MAX, 0, INT16_MIN, 0,
            INT16_MAX, 0, INT16_MIN, 0,
            INT16_MAX, 0, INT16_MIN, 0,
            INT16_MAX, 0, INT16_MIN, 0,
            INT16_MAX, 0, INT16_MIN, 0,
            INT16_MAX, 0, INT16_MIN, 0
        );


//01014545
    __m512i mask_a = _mm512_set_epi8(
        // m63~m48（d24~d31的高/低字节）
        0x0B, 0x0A, 0x09, 0x08, 0x0B, 0x0A, 0x09, 0x08,
        0x03, 0x02, 0x01, 0x00, 0x03, 0x02, 0x01, 0x00,        
        0x0B, 0x0A, 0x09, 0x08, 0x0B, 0x0A, 0x09, 0x08,
        0x03, 0x02, 0x01, 0x00, 0x03, 0x02, 0x01, 0x00,        
        0x0B, 0x0A, 0x09, 0x08, 0x0B, 0x0A, 0x09, 0x08,
        0x03, 0x02, 0x01, 0x00, 0x03, 0x02, 0x01, 0x00,
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

        __m512i tr = _mm512_adds_epi16(_mm512_mulhrs_epi16(br, wr), _mm512_mulhrs_epi16(bi, wi));
        __m512i ti = _mm512_subs_epi16(_mm512_mulhrs_epi16(bi, wr), _mm512_mulhrs_epi16(br, wi));

        real_vec[j] = SCALE_DOWN(_mm512_add_epi16(ar, tr));
        imag_vec[j] = SCALE_DOWN(_mm512_add_epi16(ai, ti));
    }
}

// =============================================================
// Stage 3: M = 8
// =============================================================
static inline void fft_stage_avx512_M8(__m512i *real_vec, __m512i *imag_vec, int num_vecs)
{

  __m512i wr = _mm512_set_epi16(
        // 第1组：-cos3, -cos2, -cos1, -cos0
        Q15_SQRT2_2, 0, -Q15_SQRT2_2, -INT16_MIN,
        // 第2组：cos3, cos2, cos1, cos0
        -Q15_SQRT2_2, 0, Q15_SQRT2_2, INT16_MAX,
        // 第3组：-cos3, -cos2, -cos1, -cos0
        Q15_SQRT2_2, 0, -Q15_SQRT2_2, -INT16_MIN,
        // 第4组：cos3, cos2, cos1, cos0
        -Q15_SQRT2_2, 0, Q15_SQRT2_2, INT16_MAX,
        // 第5组：-cos3, -cos2, -cos1, -cos0
        Q15_SQRT2_2, 0, -Q15_SQRT2_2, -INT16_MIN,
        // 第6组：cos3, cos2, cos1, cos0
        -Q15_SQRT2_2, 0, Q15_SQRT2_2, INT16_MAX,
        // 第7组：-cos3, -cos2, -cos1, -cos0
        Q15_SQRT2_2, 0, -Q15_SQRT2_2, -INT16_MIN,
        // 第8组：cos3, cos2, cos1, cos0
        -Q15_SQRT2_2, 0, Q15_SQRT2_2, INT16_MAX
    );

    // wi构造规则：[sin3,sin2,sin1,sin0, -sin3,-sin2,-sin1,-sin0] 重复4次（共32个int16）
    __m512i wi = _mm512_set_epi16(
        // 第1组：sin3, sin2, sin1, sin0
        Q15_SQRT2_2, INT16_MAX, Q15_SQRT2_2, 0,
        // 第2组：-sin3, -sin2, -sin1, -sin0
        -Q15_SQRT2_2, INT16_MIN, -Q15_SQRT2_2, -0,
        // 第3组：sin3, sin2, sin1, sin0
        Q15_SQRT2_2, INT16_MAX, Q15_SQRT2_2, 0,
        // 第4组：-sin3, -sin2, -sin1, -sin0
        -Q15_SQRT2_2, INT16_MIN, -Q15_SQRT2_2, -0,
        // 第5组：sin3, sin2, sin1, sin0
        Q15_SQRT2_2, INT16_MAX, Q15_SQRT2_2, 0,
        // 第6组：-sin3, -sin2, -sin1, -sin0
        -Q15_SQRT2_2, INT16_MIN, -Q15_SQRT2_2, -0,
        // 第7组：sin3, sin2, sin1, sin0
        Q15_SQRT2_2, INT16_MAX, Q15_SQRT2_2, 0,
        // 第8组：-sin3, -sin2, -sin1, -sin0
        -Q15_SQRT2_2, INT16_MIN, -Q15_SQRT2_2, -0
    );

            //0 1 2 3 0 1 2 3 2 8 9 10 11 8 9 10 11
            __m512i mask_a = _mm512_set_epi8(
            // m63~m48（d24~d31的高/低字节）
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,
                // m47~m32（d16~d23的高/低字节）
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,
                // m31~m16（d8~d15的高/低字节）
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,
                0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,
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

        __m512i tr = _mm512_adds_epi16(_mm512_mulhrs_epi16(br, wr), _mm512_mulhrs_epi16(bi, wi));
        __m512i ti = _mm512_subs_epi16(_mm512_mulhrs_epi16(bi, wr), _mm512_mulhrs_epi16(br, wi));

        real_vec[j] = SCALE_DOWN(_mm512_add_epi16(ar, tr));
        imag_vec[j] = SCALE_DOWN(_mm512_add_epi16(ai, ti));
    }
}

// =============================================================
// Stage 4: M = 16
// =============================================================
static inline void ifft_stage_avx512_M16(__m512i *real_vec, __m512i *imag_vec, int num_vecs)
{
    // 逻辑同上，Mask 跨度变大
    // Mask A: 0..7, Mask B: 8..15 (lane local)
    // 0..15 刚好是一个 128位 lane 的全部
    // 所以这里的 Mask 实际上是重复 0..7 和 8..15
    
    // Mask 生成...
            __m512i wr = _mm512_set_epi16(
                FLOAT_TO_Q15(-cosf(2*M_PI/16*7)),FLOAT_TO_Q15(-cosf(2*M_PI/16*6)),FLOAT_TO_Q15(-cosf(2*M_PI/16*5)),FLOAT_TO_Q15(-cosf(2*M_PI/16*4)),FLOAT_TO_Q15(-cosf(2*M_PI/16*3)),FLOAT_TO_Q15(-cosf(2*M_PI/16*2)),FLOAT_TO_Q15(-cosf(2*M_PI/16*1)),FLOAT_TO_Q15(-cosf(2*M_PI/16*0)),FLOAT_TO_Q15(cosf(2*M_PI/16*7)),FLOAT_TO_Q15(cosf(2*M_PI/16*6)),FLOAT_TO_Q15(cosf(2*M_PI/16*5)),FLOAT_TO_Q15(cosf(2*M_PI/16*4)),FLOAT_TO_Q15(cosf(2*M_PI/16*3)),FLOAT_TO_Q15(cosf(2*M_PI/16*2)),FLOAT_TO_Q15(cosf(2*M_PI/16*1)),FLOAT_TO_Q15(cosf(2*M_PI/16*0)),FLOAT_TO_Q15(-cosf(2*M_PI/16*7)),FLOAT_TO_Q15(-cosf(2*M_PI/16*6)),FLOAT_TO_Q15(-cosf(2*M_PI/16*5)),FLOAT_TO_Q15(-cosf(2*M_PI/16*4)),FLOAT_TO_Q15(-cosf(2*M_PI/16*3)),FLOAT_TO_Q15(-cosf(2*M_PI/16*2)),FLOAT_TO_Q15(-cosf(2*M_PI/16*1)),FLOAT_TO_Q15(-cosf(2*M_PI/16*0)),FLOAT_TO_Q15(cosf(2*M_PI/16*7)),FLOAT_TO_Q15(cosf(2*M_PI/16*6)),FLOAT_TO_Q15(cosf(2*M_PI/16*5)),FLOAT_TO_Q15(cosf(2*M_PI/16*4)),FLOAT_TO_Q15(cosf(2*M_PI/16*3)),FLOAT_TO_Q15(cosf(2*M_PI/16*2)),FLOAT_TO_Q15(cosf(2*M_PI/16*1)),FLOAT_TO_Q15(cosf(2*M_PI/16*0)));//低位正，高位负
            
            __m512i wi = _mm512_set_epi16(
                FLOAT_TO_Q15(-sinf(-2*M_PI/16*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*0)),FLOAT_TO_Q15(sinf(-2*M_PI/16*7)),FLOAT_TO_Q15(sinf(-2*M_PI/16*6)),FLOAT_TO_Q15(sinf(-2*M_PI/16*5)),FLOAT_TO_Q15(sinf(-2*M_PI/16*4)),FLOAT_TO_Q15(sinf(-2*M_PI/16*3)),FLOAT_TO_Q15(sinf(-2*M_PI/16*2)),FLOAT_TO_Q15(sinf(-2*M_PI/16*1)),FLOAT_TO_Q15(sinf(-2*M_PI/16*0)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/16*0)),FLOAT_TO_Q15(sinf(-2*M_PI/16*7)),FLOAT_TO_Q15(sinf(-2*M_PI/16*6)),FLOAT_TO_Q15(sinf(-2*M_PI/16*5)),FLOAT_TO_Q15(sinf(-2*M_PI/16*4)),FLOAT_TO_Q15(sinf(-2*M_PI/16*3)),FLOAT_TO_Q15(sinf(-2*M_PI/16*2)),FLOAT_TO_Q15(sinf(-2*M_PI/16*1)),FLOAT_TO_Q15(sinf(-2*M_PI/16*0)));
    
            // 序列1掩码：0~7重复、16~23重复
            __m512i mask_a= _mm512_set_epi64(
                0x05,0x04,0x05,0x04,0x01,0x00,0x01,0x00
            );

            // 序列2掩码：8~15重复、24~31重复
            __m512i mask_b = _mm512_set_epi64(
                0x07,0x06,0x07,0x06,0x03,0x02,0x03,0x02
            );

    for(int j = 0; j < num_vecs; j++) {
        __m512i r = real_vec[j];
        __m512i i = imag_vec[j];

        __m512i ar = _mm512_permutexvar_epi64(mask_a, r);
        __m512i ai = _mm512_permutexvar_epi64(mask_a, i);
        __m512i br = _mm512_permutexvar_epi64(mask_b, r);
        __m512i bi = _mm512_permutexvar_epi64(mask_b, i);

        __m512i tr = _mm512_adds_epi16(_mm512_mulhrs_epi16(br, wr), _mm512_mulhrs_epi16(bi, wi));
        __m512i ti = _mm512_subs_epi16(_mm512_mulhrs_epi16(bi, wr), _mm512_mulhrs_epi16(br, wi));

        real_vec[j] = SCALE_DOWN(_mm512_add_epi16(ar, tr));
        imag_vec[j] = SCALE_DOWN(_mm512_add_epi16(ai, ti));
    }
}

// =============================================================
// Stage 5: M = 32 (AVX512 特有)
// =============================================================
static inline void ifft_stage_avx512_M32(__m512i *real_vec, __m512i *imag_vec, int num_vecs)
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
    
            __m512i wr = _mm512_set_epi16(FLOAT_TO_Q15(-cosf(2*M_PI/32*15)),FLOAT_TO_Q15(-cosf(2*M_PI/32*14)),FLOAT_TO_Q15(-cosf(2*M_PI/32*13)),FLOAT_TO_Q15(-cosf(2*M_PI/32*12)),FLOAT_TO_Q15(-cosf(2*M_PI/32*11)),FLOAT_TO_Q15(-cosf(2*M_PI/32*10)),FLOAT_TO_Q15(-cosf(2*M_PI/32*9)),FLOAT_TO_Q15(-cosf(2*M_PI/32*8)),FLOAT_TO_Q15(-cosf(2*M_PI/32*7)),FLOAT_TO_Q15(-cosf(2*M_PI/32*6)),FLOAT_TO_Q15(-cosf(2*M_PI/32*5)),FLOAT_TO_Q15(-cosf(2*M_PI/32*4)),FLOAT_TO_Q15(-cosf(2*M_PI/32*3)),FLOAT_TO_Q15(-cosf(2*M_PI/32*2)),FLOAT_TO_Q15(-cosf(2*M_PI/32*1)),FLOAT_TO_Q15(-cosf(2*M_PI/32*0)),FLOAT_TO_Q15(cosf(2*M_PI/32*15)),FLOAT_TO_Q15(cosf(2*M_PI/32*14)),FLOAT_TO_Q15(cosf(2*M_PI/32*13)),FLOAT_TO_Q15(cosf(2*M_PI/32*12)),FLOAT_TO_Q15(cosf(2*M_PI/32*11)),FLOAT_TO_Q15(cosf(2*M_PI/32*10)),FLOAT_TO_Q15(cosf(2*M_PI/32*9)),FLOAT_TO_Q15(cosf(2*M_PI/32*8)),FLOAT_TO_Q15(cosf(2*M_PI/32*7)),FLOAT_TO_Q15(cosf(2*M_PI/32*6)),FLOAT_TO_Q15(cosf(2*M_PI/32*5)),FLOAT_TO_Q15(cosf(2*M_PI/32*4)),FLOAT_TO_Q15(cosf(2*M_PI/32*3)),FLOAT_TO_Q15(cosf(2*M_PI/32*2)),FLOAT_TO_Q15(cosf(2*M_PI/32*1)),FLOAT_TO_Q15(cosf(2*M_PI/32*0)));//低位正，高位负
            __m512i wi = _mm512_set_epi16(FLOAT_TO_Q15(-sinf(-2*M_PI/32*15)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*14)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*13)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*12)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*11)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*10)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*9)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*8)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*7)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*6)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*5)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*4)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*3)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*2)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*1)),FLOAT_TO_Q15(-sinf(-2*M_PI/32*0)),FLOAT_TO_Q15(sinf(-2*M_PI/32*15)),FLOAT_TO_Q15(sinf(-2*M_PI/32*14)),FLOAT_TO_Q15(sinf(-2*M_PI/32*13)),FLOAT_TO_Q15(sinf(-2*M_PI/32*12)),FLOAT_TO_Q15(sinf(-2*M_PI/32*11)),FLOAT_TO_Q15(sinf(-2*M_PI/32*10)),FLOAT_TO_Q15(sinf(-2*M_PI/32*9)),FLOAT_TO_Q15(sinf(-2*M_PI/32*8)),FLOAT_TO_Q15(sinf(-2*M_PI/32*7)),FLOAT_TO_Q15(sinf(-2*M_PI/32*6)),FLOAT_TO_Q15(sinf(-2*M_PI/32*5)),FLOAT_TO_Q15(sinf(-2*M_PI/32*4)),FLOAT_TO_Q15(sinf(-2*M_PI/32*3)),FLOAT_TO_Q15(sinf(-2*M_PI/32*2)),FLOAT_TO_Q15(sinf(-2*M_PI/32*1)),FLOAT_TO_Q15(sinf(-2*M_PI/32*0)));

            __m512i mask_a= _mm512_set_epi64(
                3,2,1,0,3,2,1,0
            );

            __m512i mask_b = _mm512_set_epi64(
            7,6,5,4,7,6,5,4
            );

    for(int j = 0; j < num_vecs; j++) {
        __m512i r = real_vec[j];
        __m512i i = imag_vec[j];

        // 警告：如果 mask 涉及跨 128位 lane，shuffle_epi8 可能得不到预期结果
        // 除非硬件支持或 mask 设计巧妙
        __m512i ar = _mm512_permutexvar_epi64(mask_a, r);
        __m512i ai = _mm512_permutexvar_epi64(mask_a, i);
        __m512i br = _mm512_permutexvar_epi64(mask_b, r);
        __m512i bi = _mm512_permutexvar_epi64(mask_b, i);

        __m512i tr = _mm512_adds_epi16(_mm512_mulhrs_epi16(br, wr), _mm512_mulhrs_epi16(bi, wi));
        __m512i ti = _mm512_subs_epi16(_mm512_mulhrs_epi16(bi, wr), _mm512_mulhrs_epi16(br, wi));

        real_vec[j] = SCALE_DOWN(_mm512_add_epi16(ar, tr));
        imag_vec[j] = SCALE_DOWN(_mm512_add_epi16(ai, ti));
    }
}

// =============================================================
// 主函数
// =============================================================
void ifft_AVX512_fixedP(int16_t *real, int16_t *imag, FFTContext *ctx)
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
    if (m >= 1) ifft_stage_avx512_M2(real_vec, imag_vec, num_vecs);

    if (m >= 2) ifft_stage_avx512_M4(real_vec, imag_vec, num_vecs);

    if (m >= 3) ifft_stage_avx512_M8(real_vec, imag_vec, num_vecs);

    if (m >= 4) ifft_stage_avx512_M16(real_vec, imag_vec, num_vecs);

    if (m >= 5) ifft_stage_avx512_M32(real_vec, imag_vec, num_vecs);


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

                __m512i tr = _mm512_adds_epi16(_mm512_mulhrs_epi16(r2, w_real), _mm512_mulhrs_epi16(i2, w_imag));
                __m512i ti = _mm512_subs_epi16(_mm512_mulhrs_epi16(i2, w_real), _mm512_mulhrs_epi16(r2, w_imag));

                real_vec[idx1] = SCALE_DOWN(_mm512_add_epi16(r1, tr));
                imag_vec[idx1] = SCALE_DOWN(_mm512_add_epi16(i1, ti));
                real_vec[idx2] = SCALE_DOWN(_mm512_sub_epi16(r1, tr));
                imag_vec[idx2] = SCALE_DOWN(_mm512_sub_epi16(i1, ti));
            }
        }
    }
}