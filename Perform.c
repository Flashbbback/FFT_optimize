#include <stdio.h>
#include <windows.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "include/fft_lib.h"

#define M_PI 3.14159265358979323846
#define Q 15                  // Q15格式：15位小数位
#define SCALE (1 << Q)        // 缩放因子：32768
#define INT16_MAX 32767       // int16_t最大值
#define INT16_MIN (-32768)    // int16_t最小值
#define MAG_THRESHOLD 1e-6f   // 幅度阈值，避免除零

// 修复后的宏：支持作为表达式赋值
#define FLOAT_TO_Q15(x) ({ \
    float _temp = (x) * SCALE; \
    _temp += (x >= 0 ? 0.5f : -0.5f); \
    if (_temp > INT16_MAX) _temp = INT16_MAX; \
    if (_temp < INT16_MIN) _temp = INT16_MIN; \
    (int16_t)_temp; \
})

#define Q15_TO_FLOAT(x) ((float)(x) / SCALE)  

// 误差统计结构体
typedef struct {
    float max_real_err;    
    float avg_real_err;    
    float rmse_real_err;   
    float max_imag_err;    
    float avg_imag_err;    
    float rmse_imag_err;   
    float max_mag_rel_err; 
    float avg_mag_rel_err; 
    float max_phase_err;   
    float avg_phase_err;   
} FFTError;

// ---------------------------------------------------------
// 新增功能：导出数据到CSV文件
// ---------------------------------------------------------
void save_fft_data_to_csv(const char* filename, 
                          const float* in_real, const float* in_imag,
                          const float* out_real, const float* out_imag, 
                          int size) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("无法打开文件 %s 进行写入。\n", filename);
        return;
    }

    // 写入表头：索引，输入实部，输入虚部，输出实部，输出虚部，输出幅度
    fprintf(fp, "Index,Input_Real,Input_Imag,Output_Real,Output_Imag,Output_Magnitude\n");

    for (int i = 0; i < size; i++) {
        // 计算幅度 (Magnitude) 方便查看频谱
        float mag = sqrtf(out_real[i] * out_real[i] + out_imag[i] * out_imag[i]);
        
        fprintf(fp, "%d,%.15f,%.15f,%.15f,%.15f,%.15f\n", 
                i, in_real[i], in_imag[i], out_real[i], out_imag[i], mag);
    }

    fclose(fp);
    printf(">> 数据已成功导出至文件: %s\n", filename);
}

// ---------------------------------------------------------
// 原有计算函数
// ---------------------------------------------------------
void calc_fft_error(const float* target_real, const float* target_imag,
                    const float* base_real, const float* base_imag,
                    int size, FFTError* err) {
    memset(err, 0, sizeof(FFTError));
    float sum_real_err = 0.0f, sum_imag_err = 0.0f;
    float sum_real_err_sq = 0.0f, sum_imag_err_sq = 0.0f;
    float sum_mag_rel_err = 0.0f, sum_phase_err = 0.0f;
    int valid_mag_cnt = 0;

    for (int k = 0; k < size; k++) {
        float real_err = fabsf(target_real[k] - base_real[k]);
        sum_real_err += real_err;
        sum_real_err_sq += real_err * real_err;
        if (real_err > err->max_real_err) err->max_real_err = real_err;

        float imag_err = fabsf(target_imag[k] - base_imag[k]);
        sum_imag_err += imag_err;
        sum_imag_err_sq += imag_err * imag_err;
        if (imag_err > err->max_imag_err) err->max_imag_err = imag_err;

        float base_mag = sqrtf(base_real[k] * base_real[k] + base_imag[k] * base_imag[k]);
        float target_mag = sqrtf(target_real[k] * target_real[k] + target_imag[k] * target_imag[k]);
        if (base_mag > MAG_THRESHOLD) {
            float mag_rel_err = (fabsf(target_mag - base_mag) / base_mag) * 100.0f;
            sum_mag_rel_err += mag_rel_err;
            if (mag_rel_err > err->max_mag_rel_err) err->max_mag_rel_err = mag_rel_err;
            valid_mag_cnt++;
        }

        float base_phase = atan2f(base_imag[k], base_real[k]);
        float target_phase = atan2f(target_imag[k], target_real[k]);
        float phase_err = fabsf(target_phase - base_phase);
        phase_err = (phase_err > M_PI) ? (2 * M_PI - phase_err) : phase_err;
        sum_phase_err += phase_err;
        if (phase_err > err->max_phase_err) err->max_phase_err = phase_err;
    }

    if (size > 0) {
        err->avg_real_err = sum_real_err / size;
        err->rmse_real_err = sqrtf(sum_real_err_sq / size);
        err->avg_imag_err = sum_imag_err / size;
        err->rmse_imag_err = sqrtf(sum_imag_err_sq / size);
        err->avg_phase_err = sum_phase_err / size;
    }
    if (valid_mag_cnt > 0) {
        err->avg_mag_rel_err = sum_mag_rel_err / valid_mag_cnt;
    }
}

void print_fft_error(const char* method_name, const FFTError* err) {
    printf("\n=== %s 与迭代版FFT的误差对比 ===\n", method_name);
    printf("实部误差 - 最大：%.6e | 平均：%.6e | RMSE：%.6e\n",
           err->max_real_err, err->avg_real_err, err->rmse_real_err);
    printf("虚部误差 - 最大：%.6e | 平均：%.6e | RMSE：%.6e\n",
           err->max_imag_err, err->avg_imag_err, err->rmse_imag_err);
    printf("幅度相对误差 - 最大：%.4f%% | 平均：%.4f%%\n",
           err->max_mag_rel_err, err->avg_mag_rel_err);
}

int main() {
    SetConsoleOutputCP(65001);
    LARGE_INTEGER start[4], end[4], freq;
    QueryPerformanceFrequency(&freq);

    printf("---FFT 性能与误差对比测试---\n");
    // 添加多种FFT尺寸进行测试
    int fft_size[] = {1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14}; // 256, 512, 1024, 2048, 4096, 8192, 16384点
    
    // 根据尺寸动态调整重复次数，小尺寸多重复几次
    int repeat[] = {1000000, 1000000, 1000000, 500000, 100000, 10000, 1000};
    int sizes_count = sizeof(fft_size)/sizeof(fft_size[0]);

    printf("性能测试结果汇总:\n");
    printf("| FFT尺寸 | 迭代版性能(ms) | 浮点版性能(ms) | 浮点速度倍数 | 定点版性能(ms) | 定点速度倍数 |\n");
    printf("|---------|--------------|---------------|------------|---------------|------------|\n");

    for (int i = 0; i < sizes_count; i++) {
        int size = fft_size[i];
        int rep = repeat[i];
        
        // 分配内存
        float* signal_orig_real = (float*)_aligned_malloc(size * sizeof(float), 32);
        float* signal_orig_imag = (float*)_aligned_malloc(size * sizeof(float), 32);
        float* signal_real = (float*)_aligned_malloc(size * sizeof(float), 32);
        float* signal_imag = (float*)_aligned_malloc(size * sizeof(float), 32);
        int16_t* signal_q15_real = (int16_t*)_aligned_malloc(size * sizeof(int16_t), 32);
        int16_t* signal_q15_imag = (int16_t*)_aligned_malloc(size * sizeof(int16_t), 32);
        int16_t* signal_q15_temp_real = (int16_t*)_aligned_malloc(size * sizeof(int16_t), 32);
        int16_t* signal_q15_temp_imag = (int16_t*)_aligned_malloc(size * sizeof(int16_t), 32);

        FFTContext* ctx = trig_table(size);
        
        // 填充输入信号
        for (int j = 0; j < size; j++) {
            float sig_float = sinf(2 * M_PI * 10 * j / size)/size;
            signal_orig_real[j] = sig_float;
            signal_orig_imag[j] = 0.0f;
            signal_q15_real[j] = FLOAT_TO_Q15(sig_float);
            signal_q15_imag[j] = 0;
        }

        // 获取迭代版基准结果
        float* base_real = (float*)_aligned_malloc(size * sizeof(float), 32);
        float* base_imag = (float*)_aligned_malloc(size * sizeof(float), 32);
        memcpy(base_real, signal_orig_real, size * sizeof(float));
        memcpy(base_imag, signal_orig_imag, size * sizeof(float));
        fft_diedai(base_real, base_imag, size, ctx);

        // 性能测试
        float time_diedai = 0.0f, time_avx_float = 0.0f, time_avx_fixed = 0.0f;

        // 迭代版测试
        QueryPerformanceCounter(&start[1]);
        for (int j = 0; j < rep; j++) {
            memcpy(signal_real, signal_orig_real, size * sizeof(float));
            memcpy(signal_imag, signal_orig_imag, size * sizeof(float));
            fft_diedai(signal_real, signal_imag, size, ctx);
        }
        QueryPerformanceCounter(&end[1]);
        time_diedai = (end[1].QuadPart - start[1].QuadPart) / (float)freq.QuadPart / rep * 1000;

        // AVX浮点版测试
        QueryPerformanceCounter(&start[2]);
        for (int j = 0; j < rep; j++) {
            memcpy(signal_real, signal_orig_real, size * sizeof(float));
            memcpy(signal_imag, signal_orig_imag, size * sizeof(float));
            fft_AVX(signal_real, signal_imag, size, ctx);
        }
        QueryPerformanceCounter(&end[2]);
        time_avx_float = (end[2].QuadPart - start[2].QuadPart) / (float)freq.QuadPart / rep * 1000;

        // AVX定点版测试
        QueryPerformanceCounter(&start[3]);
        for (int j = 0; j < rep; j++) {
            memcpy(signal_q15_temp_real, signal_q15_real, size * sizeof(int16_t));
            memcpy(signal_q15_temp_imag, signal_q15_imag, size * sizeof(int16_t));
            fft_AVX_fixedP(signal_q15_temp_real, signal_q15_temp_imag, size, ctx);
        }
        QueryPerformanceCounter(&end[3]);
        time_avx_fixed = (end[3].QuadPart - start[3].QuadPart) / (float)freq.QuadPart / rep * 1000;

        // 计算速度提升倍数（将比值转换为倍数）
        float float_speedup = time_diedai / time_avx_float;
        float fixed_speedup = time_diedai / time_avx_fixed;
        
        // 打印当前尺寸的性能结果，格式与README.md保持一致
        printf("| %-9d | %-14.6f | %-15.6f | %-11.2fx | %-15.6f | %-11.2fx |\n", 
               size, time_diedai, time_avx_float, float_speedup, time_avx_fixed, fixed_speedup);

        // 释放内存
        _aligned_free(signal_orig_real);
        _aligned_free(signal_orig_imag);
        _aligned_free(signal_real);
        _aligned_free(signal_imag);
        _aligned_free(signal_q15_real);
        _aligned_free(signal_q15_imag);
        _aligned_free(signal_q15_temp_real);
        _aligned_free(signal_q15_temp_imag);
        _aligned_free(base_real);
        _aligned_free(base_imag);
        free_trig_table(ctx);
    }

    // 循环结束后，单独进行4096点的误差分析
    printf("\n\n--- 4096点 FFT 误差分析 ---");
    
    int test_size = 4096;
    // 分配内存用于误差分析
    float* signal_orig_real = (float*)_aligned_malloc(test_size * sizeof(float), 32);
    float* signal_orig_imag = (float*)_aligned_malloc(test_size * sizeof(float), 32);
    float* signal_real = (float*)_aligned_malloc(test_size * sizeof(float), 32);
    float* signal_imag = (float*)_aligned_malloc(test_size * sizeof(float), 32);
    int16_t* signal_q15_real = (int16_t*)_aligned_malloc(test_size * sizeof(int16_t), 32);
    int16_t* signal_q15_imag = (int16_t*)_aligned_malloc(test_size * sizeof(int16_t), 32);
    int16_t* signal_q15_temp_real = (int16_t*)_aligned_malloc(test_size * sizeof(int16_t), 32);
    int16_t* signal_q15_temp_imag = (int16_t*)_aligned_malloc(test_size * sizeof(int16_t), 32);
    float* base_real = (float*)_aligned_malloc(test_size * sizeof(float), 32);
    float* base_imag = (float*)_aligned_malloc(test_size * sizeof(float), 32);
    
    FFTContext* ctx = trig_table(test_size);
    
    // 填充输入信号
    for (int j = 0; j < test_size; j++) {
        float sig_float = sinf(2 * M_PI * 10 * j / test_size)/test_size;
        signal_orig_real[j] = sig_float;
        signal_orig_imag[j] = 0.0f;
        signal_q15_real[j] = FLOAT_TO_Q15(sig_float);
        signal_q15_imag[j] = 0;
    }
    
    // 获取迭代版基准结果
    memcpy(base_real, signal_orig_real, test_size * sizeof(float));
    memcpy(base_imag, signal_orig_imag, test_size * sizeof(float));
    fft_diedai(base_real, base_imag, test_size, ctx);
    
    // AVX浮点版误差
    memcpy(signal_real, signal_orig_real, test_size * sizeof(float));
    memcpy(signal_imag, signal_orig_imag, test_size * sizeof(float));
    fft_AVX(signal_real, signal_imag, test_size, ctx);
    
    FFTError avx_err;
    calc_fft_error(signal_real, signal_imag, base_real, base_imag, test_size, &avx_err);
    print_fft_error("AVX浮点版", &avx_err);
    
    // AVX定点版误差
    memcpy(signal_q15_temp_real, signal_q15_real, test_size * sizeof(int16_t));
    memcpy(signal_q15_temp_imag, signal_q15_imag, test_size * sizeof(int16_t));
    fft_AVX_fixedP(signal_q15_temp_real, signal_q15_temp_imag, test_size, ctx);
    
    float* fixed_float_real = (float*)_aligned_malloc(test_size * sizeof(float), 32);
    float* fixed_float_imag = (float*)_aligned_malloc(test_size * sizeof(float), 32);
    for (int k = 0; k < test_size; k++) {
        fixed_float_real[k] = Q15_TO_FLOAT(signal_q15_temp_real[k]);
        fixed_float_imag[k] = Q15_TO_FLOAT(signal_q15_temp_imag[k]);
    }
    
    FFTError fixed_err;
    calc_fft_error(fixed_float_real, fixed_float_imag, base_real, base_imag, test_size, &fixed_err);
    print_fft_error("AVX定点版", &fixed_err);
    
    // 释放内存
    _aligned_free(signal_orig_real);
    _aligned_free(signal_orig_imag);
    _aligned_free(signal_real);
    _aligned_free(signal_imag);
    _aligned_free(signal_q15_real);
    _aligned_free(signal_q15_imag);
    _aligned_free(signal_q15_temp_real);
    _aligned_free(signal_q15_temp_imag);
    _aligned_free(base_real);
    _aligned_free(base_imag);
    _aligned_free(fixed_float_real);
    _aligned_free(fixed_float_imag);
    free_trig_table(ctx);
    
    printf("\nFFT 性能测试完成!\n");
    printf("注：性能数据基于当前硬件平台，不同平台可能有差异。\n");
    return 0;
}