#include <stdio.h>
#include <windows.h>
#include "FFT.h"

int main() {
    LARGE_INTEGER start[3], end[3], freq;
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
        QueryPerformanceCounter(&start[0]);
        for (int j = 0; j < repeat; j++) {
            fft_digui(signal_real2, signal_imag2 , size, ctx);
        }
        QueryPerformanceCounter(&end[0]);

        QueryPerformanceCounter(&start[1]);
        for (int j = 0; j < repeat; j++) {
            fft_diedai(signal_real2, signal_imag2 , size, ctx);
        }
        QueryPerformanceCounter(&end[1]);
        


        QueryPerformanceCounter(&start[2]);
        for (int j = 0; j < repeat; j++) {

            fft_AVX(signal_real1, signal_imag1, size, ctx);
        }
        QueryPerformanceCounter(&end[2]);


        // printf("--------------------\n");
        // for(int i = 0;i<16;i++)
        // {
        // printf("real:%.f,imag:%.f\n",signal_real2[i],signal_imag2[i]);
        // }
        // 计算并输出当前size的耗时（建议在此处输出，避免循环外变量问题）
        float time1 = (end[0].QuadPart - start[0].QuadPart) / (float)freq.QuadPart / repeat;
        float time2 = (end[1].QuadPart - start[1].QuadPart) / (float)freq.QuadPart / repeat;
        float time3 = (end[2].QuadPart - start[2].QuadPart) / (float)freq.QuadPart / repeat;
        printf("Size: %d, Repeat: %d, Time taken with digui: %.6f ms\n", size, repeat, time1 * 1000);
        printf("Size: %d, Repeat: %d, Time taken with diedai: %.6f ms\n", size, repeat, time2 * 1000);
        printf("Size: %d, Repeat: %d, Time taken with AVX: %.6f ms\n", size, repeat, time3 * 1000);



        // 释放信号数组
        free(signal_real1);
        free(signal_imag1);
        free(signal_real2);
        free(signal_imag2);
        free_trig_table(ctx);
    }

    printf("FFT completed.\n");  // 现在会执行到这里
    return 0;
}