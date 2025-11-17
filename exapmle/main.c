#include "FFT.h"
#include <stdlib.h>
#include <stdio.h>

int main() {
    // 1. 配置参数（FFT尺寸为1024，2的幂）
    const int N = 1024;
    FFTContext* ctx = trig_table(N);
    if (!ctx) {
        printf("上下文初始化失败！\n");
        return -1;
    }

    // 2. 分配32字节对齐的输入/输出数组
    // 浮点版数组
    float* float_real = (float*)_aligned_malloc(N * sizeof(float), 32);
    float* float_imag = (float*)_aligned_malloc(N * sizeof(float), 32);
    // 定点版数组（Q15格式）
    int16_t* q15_real = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);
    int16_t* q15_imag = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);

    // 3. 填充输入信号（以单频正弦信号为例）
    for (int i = 0; i < N; i++) {
        // 浮点信号：sin(2π*10*i/N)，频率10Hz
        float sig_float = sinf(2 * M_PI * 10 * i / N);
        float_real[i] = sig_float;
        float_imag[i] = 0.0f;

        // 定点信号：转换为Q15格式
        q15_real[i] = FLOAT_TO_Q15(sig_float);
        q15_imag[i] = 0; // 虚部为0
    }

    // 4. 调用FFT计算
    fft_AVX(float_real, float_imag, N, ctx);         // 浮点版
    fft_AVX_fixedP(q15_real, q15_imag, N, ctx);      // 定点版

    // 5. 处理结果（示例：打印前10个点的实部/虚部）
    printf("=== 浮点版FFT结果（前10点）===\n");
    for (int i = 0; i < 10; i++) {
        printf("第%d点：实部=%.6f，虚部=%.6f\n", i, float_real[i], float_imag[i]);
    }

    printf("\n=== 定点版FFT结果（前10点，Q15转浮点）===\n");
    for (int i = 0; i < 10; i++) {
        printf("第%d点：实部=%.6f，虚部=%.6f\n", 
               i, Q15_TO_FLOAT(q15_real[i]), Q15_TO_FLOAT(q15_imag[i]));
    }

    // 6. 释放资源
    _aligned_free(float_real);
    _aligned_free(float_imag);
    _aligned_free(q15_real);
    _aligned_free(q15_imag);
    free_trig_table(ctx);

    return 0;
}