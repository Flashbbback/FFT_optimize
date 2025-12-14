#include "fft_internal.h"


// =============================================================
// IFFT Wrapper: 复用 FFT 实现
// 原理: IFFT(real, imag) = 1/N * conj(FFT(imag, real))
// =============================================================
void ifft_AVX_reuse(float *real, float *imag, int N, FFTContext *ctx)
{
    // 1. 交换实部和虚部传入 FFT
    // 注意：此时 fft_AVX 会把计算出的 IFFT 实部结果写在 imag 数组里，
    // 把 IFFT 虚部结果写在 real 数组里。
    fft_AVX(imag, real, N, ctx);

    // 2. 归一化 (除以 N) 并交换回正确的位置
    // 我们需要把 imag 的内容(真.实部) 搬回 real，把 real 的内容(真.虚部) 搬回 imag
    
    float scale = 1.0f / (float)N;
    __m256 v_scale = _mm256_set1_ps(scale);
    
    int num_vecs = N / 8;
    __m256 *ptr_real = (__m256 *)real;
    __m256 *ptr_imag = (__m256 *)imag;

    for (int i = 0; i < num_vecs; i++)
    {
        // 加载数据 (此时 ptr_real 里存的是虚部结果，ptr_imag 里存的是实部结果)
        __m256 v_res_imag = ptr_imag[i]; 
        __m256 v_res_real = ptr_real[i];

        // 缩放
        v_res_real = _mm256_mul_ps(v_res_real, v_scale);
        v_res_imag = _mm256_mul_ps(v_res_imag, v_scale);
    
        // 存回正确的位置
        ptr_real[i] = v_res_real;
        ptr_imag[i] = v_res_imag;
    }
}

