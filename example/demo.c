#include "../src/FFT.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define M_PI 3.14159265358979323846
#define Q 15                  // Q15格式:15位小数位
#define SCALE (1 << Q)        // 缩放因子:32768
#define INT16_MAX 32767       // int16_t最大值
#define INT16_MIN (-32768)    // int16_t最小值

// 修复后的宏:支持作为表达式赋值
#define FLOAT_TO_Q15(x) ({ \
    float _temp = (x) * SCALE; \
    _temp += (x >= 0 ? 0.5f : -0.5f); \
    if (_temp > INT16_MAX) _temp = INT16_MAX; \
    if (_temp < INT16_MIN) _temp = INT16_MIN; \
    (int16_t)_temp; \
})

#define Q15_TO_FLOAT(x) ((float)(x) / SCALE)

int main() {
    // 1. 配置参数(FFT尺寸为1024,2的幂)
    const int N = 1<<10;
    FFTContext* ctx = trig_table(N);
    if (!ctx) {
        printf("failed init FFTContext！\n");
        return -1;
    }

    // 2. 分配32字节对齐的输入/输出数组
    // 浮点版数组
    float* float_real = (float*)_aligned_malloc(N * sizeof(float), 32);
    float* float_imag = (float*)_aligned_malloc(N * sizeof(float), 32);
    // 定点版数组(Q15格式)
    int16_t* q15_real = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);
    int16_t* q15_imag = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);

    // 3. 填充输入信号(以单频正弦信号为例)
    for (int i = 0; i < N; i++) {
        // 浮点信号:sin(2π*10*i/N),频率10Hz
        float sig_float = sinf(2 * M_PI * 10 * i / N)/N;
        float_real[i] = sig_float;
        float_imag[i] = 0.0f;

        // 定点信号:转换为Q15格式
        q15_real[i] = FLOAT_TO_Q15(sig_float);
        q15_imag[i] = 0; // 虚部为0
    }

    // 4. 调用FFT计算
    fft_AVX(float_real, float_imag, N, ctx);         // 浮点版
    fft_AVX_fixedP(q15_real, q15_imag, N, ctx);      // 定点版

    // 5. 处理结果(示例:打印前10个点的实部/虚部)
    printf("=== Float FFT result(16 points)===\n");
    for (int i = 0; i < 16; i++) {
        printf("%d:real=%.6f,imag=%.6f\n", i, float_real[i], float_imag[i]);
    }

    printf("\n=== Fixed-point FFT result(16 points,Q15 format)===\n");
    for (int i = 0; i < 16; i++) {
        printf("%d:real=%.6f,imag=%.6f\n", 
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