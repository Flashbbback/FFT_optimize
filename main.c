#include <stdio.h>
#include <windows.h>
#include "FFT.h"

int main() {
    LARGE_INTEGER start[2], end[2], freq;
    QueryPerformanceFrequency(&freq);  // 先获取频率

    printf("---FFT---\n");
    int fft_size[] = {1<<16};  // 1048576
    int repeat = 100;
    for (int i = 0; i < sizeof(fft_size)/sizeof(fft_size[0]); i++) {
        int size = fft_size[i];
        // 堆上分配信号数组，避免栈溢出
        float* signal_real = (float*)malloc(size * sizeof(float));
        float* signal_imag = (float*)malloc(size * sizeof(float));
        
        signal_gen(signal_real, signal_imag, size);

        QueryPerformanceCounter(&start[0]);
        for (int j = 0; j < repeat; j++) {
            fft_digui(signal_real, signal_imag, size);
        }
        QueryPerformanceCounter(&end[0]);

        QueryPerformanceCounter(&start[1]);
        for (int j = 0; j < repeat; j++) {
            fft_diedai(signal_real, signal_imag, size);
        }
        QueryPerformanceCounter(&end[1]);
        // 计算并输出当前size的耗时（建议在此处输出，避免循环外变量问题）
        float time = (end[0].QuadPart - start[0].QuadPart) / (float)freq.QuadPart / repeat;
        float time2 = (end[1].QuadPart - start[1].QuadPart) / (float)freq.QuadPart / repeat;
        printf("Size: %d, Repeat: %d, Time taken with digui: %.6f ms\n", size, repeat, time * 1000);
        printf("Size: %d, Repeat: %d, Time taken with diedai: %.6f ms\n", size, repeat, time2 * 1000);

        // 释放信号数组
        free(signal_real);
        free(signal_imag);
    }

    printf("FFT completed.\n");  // 现在会执行到这里
    return 0;
}