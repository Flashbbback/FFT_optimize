#include <stdio.h>
#include <windows.h>
#include <stdint.h>
#include "FFT.h"

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



int main() {
    LARGE_INTEGER start[4], end[4], freq;
    QueryPerformanceFrequency(&freq);  // 先获取频率

    printf("---FFT---\n");
    int fft_size[] = {1<<16};  // 1048576
    int repeat = 100;
    for (int i = 0; i < sizeof(fft_size)/sizeof(fft_size[0]); i++) {
        int size = fft_size[i];
        // 堆上分配信号数组，避免栈溢出
        float* signal_real1 = (float*)_aligned_malloc(size * sizeof(float),32);
        float* signal_imag1 = (float*)_aligned_malloc(size * sizeof(float),32);
        float* signal_real2 = (float*)_aligned_malloc(size * sizeof(float),32);
        float* signal_imag2 = (float*)_aligned_malloc(size * sizeof(float),32);
        float* signal_real3 = (float*)_aligned_malloc(size * sizeof(float),32);
        float* signal_imag3 = (float*)_aligned_malloc(size * sizeof(float),32);
        int16_t* signal_real4 = (int16_t*)_aligned_malloc(size * sizeof(int16_t),32);
        int16_t* signal_imag4 = (int16_t*)_aligned_malloc(size * sizeof(int16_t),32);

        FFTContext* ctx = trig_table(size);
        signal_gen(signal_real1, signal_imag1 , size);
        // 分配内存后，复制内容（而非指针赋值）
        memcpy(signal_real2, signal_real1, size * sizeof(float));
        memcpy(signal_imag2, signal_imag1, size * sizeof(float));
        memcpy(signal_real3, signal_real1, size * sizeof(float));
        memcpy(signal_imag3, signal_imag1, size * sizeof(float));

        // fft_AVX(signal_real1, signal_imag1, size, ctx);
        
        // for(int i = 0;i<16;i++)
        // {
        // printf("real:%.f,imag:%.f\n",signal_real1[i],signal_imag1[i]);
        // }
        //递归版
        QueryPerformanceCounter(&start[0]);
        for (int j = 0; j < repeat; j++) {
            fft_digui(signal_real1, signal_imag1 , size, ctx);
        }
        QueryPerformanceCounter(&end[0]);
        float time1 = (end[0].QuadPart - start[0].QuadPart) / (float)freq.QuadPart /repeat ;
        printf("Size: %d, Repeat: %d, Time taken with digui: %.6f ms\n", size, repeat, time1 * 1000);


        QueryPerformanceCounter(&start[1]);
        for (int j = 0; j < repeat; j++) {
            fft_diedai(signal_real2, signal_imag2 , size, ctx);
        }
        QueryPerformanceCounter(&end[1]);
        float time2 = (end[1].QuadPart - start[1].QuadPart) / (float)freq.QuadPart /repeat;
        printf("Size: %d, Repeat: %d, Time taken with diedai: %.6f ms\n", size, repeat, time2 * 1000);


        QueryPerformanceCounter(&start[2]);
        for (int j = 0; j < repeat; j++) {

            fft_AVX(signal_real3, signal_imag3, size, ctx);
        }
        QueryPerformanceCounter(&end[2]);
        float time3 = (end[2].QuadPart - start[2].QuadPart) / (float)freq.QuadPart /repeat ;
        printf("Size: %d, Repeat: %d, Time taken with AVX: %.6f ms\n", size, repeat, time3 * 1000);

        signal_gen_q15(signal_real4, signal_imag4 , size);

        QueryPerformanceCounter(&start[3]);
        for (int j = 0; j < repeat; j++) {

            fft_AVX_fixedP(signal_real4, signal_imag4, size, ctx);
        }
        QueryPerformanceCounter(&end[3]);


        // printf("--------------------\n");
        // for(int i = 0;i<16;i++)
        // {
        // printf("real:%.6f,imag:%.6f\n",signal_real1[i],signal_imag1[i]);
        // }
        // printf("--------------------\n");
        // for(int i = 0;i<16;i++)
        // {
        // printf("real:%.6f,imag:%.6f\n",signal_real2[i],signal_imag2[i]);
        // }

        // printf("--------------------\n");
        // for(int i = 0;i<16;i++)
        // {
        // printf("real:%.6f,imag:%.6f\n",signal_real3[i],signal_imag3[i]);
        // }
        // printf("--------------------\n");
        // for(int i = 0;i<16;i++)
        // {
        // printf("real:%.6f,imag:%.6f\n",Q15_TO_FLOAT(signal_real4[i]),Q15_TO_FLOAT(signal_imag4[i]));
        // }
        // printf("--------------------\n");
        // for(int i = 0;i<16;i++)
        // {
        // printf("real:%d,imag:%d\n",signal_real4[i],signal_imag4[i]);
        // }
        
        // 计算并输出当前size的耗时（建议在此处输出，避免循环外变量问题）
        float time4 = (end[3].QuadPart - start[3].QuadPart) / (float)freq.QuadPart /repeat ;
        // printf("Size: %d, Repeat: %d, Time taken with digui: %.6f ms\n", size, repeat, time1 * 1000);
        // printf("Size: %d, Repeat: %d, Time taken with diedai: %.6f ms\n", size, repeat, time2 * 1000);
        // printf("Size: %d, Repeat: %d, Time taken with AVX: %.6f ms\n", size, repeat, time3 * 1000);
        printf("Size: %d, Repeat: %d, Time taken with fied point: %.6f ms\n", size, repeat, time4 * 1000);



        // 释放信号数组
        _aligned_free(signal_real1);
        _aligned_free(signal_imag1);
        _aligned_free(signal_real2);
        _aligned_free(signal_imag2);
        _aligned_free(signal_real3);
        _aligned_free(signal_imag3);
        _aligned_free(signal_real4);
        _aligned_free(signal_imag4);
        free_trig_table(ctx);
    }

    printf("FFT completed.\n");  // 现在会执行到这里
    return 0;
}