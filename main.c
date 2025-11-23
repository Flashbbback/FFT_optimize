#include <stdio.h>
#include <windows.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "FFT.h"

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

// 误差统计结构体，存储各类误差指标
typedef struct {
    float max_real_err;    // 实部最大绝对误差
    float avg_real_err;    // 实部平均绝对误差
    float rmse_real_err;   // 实部均方根误差
    float max_imag_err;    // 虚部最大绝对误差
    float avg_imag_err;    // 虚部平均绝对误差
    float rmse_imag_err;   // 虚部均方根误差
    float max_mag_rel_err; // 幅度最大相对误差（%）
    float avg_mag_rel_err; // 幅度平均相对误差（%）
    float max_phase_err;   // 相位最大绝对误差（弧度）
    float avg_phase_err;   // 相位平均绝对误差（弧度）
} FFTError;

/**
 * 核心误差计算函数：对比目标方法与迭代版（基准）的误差
 * @param target_real: 目标方法的实部结果（float）
 * @param target_imag: 目标方法的虚部结果（float）
 * @param base_real: 迭代版（基准）的实部结果
 * @param base_imag: 迭代版（基准）的虚部结果
 * @param size: FFT长度
 * @param err: 输出参数，存储计算后的误差指标
 */
void calc_fft_error(const float* target_real, const float* target_imag,
                    const float* base_real, const float* base_imag,
                    int size, FFTError* err) {
    // 初始化误差结构体为0
    memset(err, 0, sizeof(FFTError));

    float sum_real_err = 0.0f;    // 实部误差总和
    float sum_imag_err = 0.0f;    // 虚部误差总和
    float sum_real_err_sq = 0.0f; // 实部误差平方和
    float sum_imag_err_sq = 0.0f; // 虚部误差平方和
    float sum_mag_rel_err = 0.0f; // 幅度相对误差总和
    float sum_phase_err = 0.0f;   // 相位误差总和
    int valid_mag_cnt = 0;        // 有效幅度点数（避免除零）

    for (int k = 0; k < size; k++) {
        // 1. 计算实部绝对误差
        float real_err = fabsf(target_real[k] - base_real[k]);
        sum_real_err += real_err;
        sum_real_err_sq += real_err * real_err;
        if (real_err > err->max_real_err) {
            err->max_real_err = real_err;
        }

        // 2. 计算虚部绝对误差
        float imag_err = fabsf(target_imag[k] - base_imag[k]);
        sum_imag_err += imag_err;
        sum_imag_err_sq += imag_err * imag_err;
        if (imag_err > err->max_imag_err) {
            err->max_imag_err = imag_err;
        }

        // 3. 计算幅度相对误差（跳过幅度接近0的点）
        float base_mag = sqrtf(base_real[k] * base_real[k] + base_imag[k] * base_imag[k]);
        float target_mag = sqrtf(target_real[k] * target_real[k] + target_imag[k] * target_imag[k]);
        if (base_mag > MAG_THRESHOLD) {
            float mag_rel_err = (fabsf(target_mag - base_mag) / base_mag) * 100.0f;
            sum_mag_rel_err += mag_rel_err;
            if (mag_rel_err > err->max_mag_rel_err) {
                err->max_mag_rel_err = mag_rel_err;
            }
            valid_mag_cnt++;
        }

        // 4. 计算相位绝对误差（处理-π~π的角度环绕）
        float base_phase = atan2f(base_imag[k], base_real[k]);
        float target_phase = atan2f(target_imag[k], target_real[k]);
        float phase_err = fabsf(target_phase - base_phase);
        // 取最小角度差（避免因π/-π导致的误差误判）
        phase_err = (phase_err > M_PI) ? (2 * M_PI - phase_err) : phase_err;
        sum_phase_err += phase_err;
        if (phase_err > err->max_phase_err) {
            err->max_phase_err = phase_err;
        }
    }

    // 计算平均误差和RMSE
    if (size > 0) {
        err->avg_real_err = sum_real_err / size;
        err->rmse_real_err = sqrtf(sum_real_err_sq / size);
        err->avg_imag_err = sum_imag_err / size;
        err->rmse_imag_err = sqrtf(sum_imag_err_sq / size);
        err->avg_phase_err = sum_phase_err / size;
    }
    // 计算幅度平均相对误差
    if (valid_mag_cnt > 0) {
        err->avg_mag_rel_err = sum_mag_rel_err / valid_mag_cnt;
    }
}

/**
 * 打印误差结果的辅助函数
 * @param method_name: 待对比的方法名称
 * @param err: 误差指标结构体
 */
void print_fft_error(const char* method_name, const FFTError* err) {
    printf("\n=== %s 与迭代版FFT的误差对比 ===\n", method_name);
    printf("实部误差 - 最大：%.6e | 平均：%.6e | RMSE：%.6e\n",
           err->max_real_err, err->avg_real_err, err->rmse_real_err);
    printf("虚部误差 - 最大：%.6e | 平均：%.6e | RMSE：%.6e\n",
           err->max_imag_err, err->avg_imag_err, err->rmse_imag_err);
    printf("幅度相对误差 - 最大：%.4f%% | 平均：%.4f%%\n",
           err->max_mag_rel_err, err->avg_mag_rel_err);
    // printf("相位误差（弧度） - 最大：%.6e | 平均：%.6e\n",
    //        err->max_phase_err, err->avg_phase_err);
}

int main() {
    SetConsoleOutputCP(65001);
    LARGE_INTEGER start[4], end[4], freq;
    QueryPerformanceFrequency(&freq);

    printf("---FFT 性能与误差对比测试---\n");
    int fft_size[] = {1 << 12};  // 4096点FFT
    int repeat = 100000;

    for (int i = 0; i < sizeof(fft_size)/sizeof(fft_size[0]); i++) {
        int size = fft_size[i];
        // 分配内存（32字节对齐，适配AVX）
        float* signal_orig_real = (float*)_aligned_malloc(size * sizeof(float), 32);
        float* signal_orig_imag = (float*)_aligned_malloc(size * sizeof(float), 32);
        float* signal_real = (float*)_aligned_malloc(size * sizeof(float), 32);
        float* signal_imag = (float*)_aligned_malloc(size * sizeof(float), 32);
        int16_t* signal_q15_real = (int16_t*)_aligned_malloc(size * sizeof(int16_t), 32);
        int16_t* signal_q15_imag = (int16_t*)_aligned_malloc(size * sizeof(int16_t), 32);
        int16_t* signal_q15_temp_real = (int16_t*)_aligned_malloc(size * sizeof(int16_t), 32);
        int16_t* signal_q15_temp_imag = (int16_t*)_aligned_malloc(size * sizeof(int16_t), 32);

        // 生成原始信号并初始化FFT上下文
        FFTContext* ctx = trig_table(size);
        signal_gen(signal_orig_real, signal_orig_imag, size);
        // 生成与原始float信号一致的Q15定点信号
        for (int k = 0; k < size; k++) {
            signal_q15_real[k] = FLOAT_TO_Q15(signal_orig_real[k]);
            signal_q15_imag[k] = FLOAT_TO_Q15(signal_orig_imag[k]);
        }

        // ==========================
        // 步骤1：获取迭代版基准结果
        // ==========================
        float* base_real = (float*)_aligned_malloc(size * sizeof(float), 32);
        float* base_imag = (float*)_aligned_malloc(size * sizeof(float), 32);
        memcpy(base_real, signal_orig_real, size * sizeof(float));
        memcpy(base_imag, signal_orig_imag, size * sizeof(float));
        fft_diedai(base_real, base_imag, size, ctx);

        // ==========================
        // 步骤2：性能测试（原有逻辑保留）
        // ==========================
        float time1 = 0.0f, time2 = 0.0f, time3 = 0.0f, time4 = 0.0f;

        // 递归版（注释保留，如需测试可取消注释）
        QueryPerformanceCounter(&start[0]);
        QueryPerformanceCounter(&end[0]);
        time1 = (end[0].QuadPart - start[0].QuadPart) / (float)freq.QuadPart / repeat * 1000;
        printf("Size: %d, Repeat: %d, Time taken with digui: %.6f ms\n", size, repeat, time1 * 1000);

        // 迭代版
        QueryPerformanceCounter(&start[1]);
        for (int j = 0; j < repeat; j++) {
            memcpy(signal_real, signal_orig_real, size * sizeof(float));
            memcpy(signal_imag, signal_orig_imag, size * sizeof(float));
            fft_diedai(signal_real, signal_imag, size, ctx);
        }
        QueryPerformanceCounter(&end[1]);
        time2 = (end[1].QuadPart - start[1].QuadPart) / (float)freq.QuadPart / repeat * 1000;
        printf("Size: %d, Repeat: %d, Time taken with diedai: %.6f ms\n", size, repeat, time2);

        // AVX浮点版
        QueryPerformanceCounter(&start[2]);
        for (int j = 0; j < repeat; j++) {
            memcpy(signal_real, signal_orig_real, size * sizeof(float));
            memcpy(signal_imag, signal_orig_imag, size * sizeof(float));
            fft_AVX(signal_real, signal_imag, size, ctx);
        }
        QueryPerformanceCounter(&end[2]);
        time3 = (end[2].QuadPart - start[2].QuadPart) / (float)freq.QuadPart / repeat * 1000;
        printf("Size: %d, Repeat: %d, Time taken with AVX: %.6f ms\n", size, repeat, time3);

        // AVX定点版
        QueryPerformanceCounter(&start[3]);
        for (int j = 0; j < repeat; j++) {
            memcpy(signal_q15_temp_real, signal_q15_real, size * sizeof(int16_t));
            memcpy(signal_q15_temp_imag, signal_q15_imag, size * sizeof(int16_t));
            fft_AVX_fixedP(signal_q15_temp_real, signal_q15_temp_imag, size, ctx);
        }
        QueryPerformanceCounter(&end[3]);
        time4 = (end[3].QuadPart - start[3].QuadPart) / (float)freq.QuadPart / repeat * 1000;
        printf("Size: %d, Repeat: %d, Time taken with fixed point: %.6f ms\n", size, repeat, time4);

        // ==========================
        // 步骤3：误差对比
        // ==========================
        FFTError avx_err, fixed_err;

        // 3.1 AVX浮点版 与 迭代版误差对比
        memcpy(signal_real, signal_orig_real, size * sizeof(float));
        memcpy(signal_imag, signal_orig_imag, size * sizeof(float));
        fft_AVX(signal_real, signal_imag, size, ctx);
        calc_fft_error(signal_real, signal_imag, base_real, base_imag, size, &avx_err);
        print_fft_error("AVX浮点版", &avx_err);

        // 3.2 AVX定点版 与 迭代版误差对比（先转float）
        memcpy(signal_q15_temp_real, signal_q15_real, size * sizeof(int16_t));
        memcpy(signal_q15_temp_imag, signal_q15_imag, size * sizeof(int16_t));
        fft_AVX_fixedP(signal_q15_temp_real, signal_q15_temp_imag, size, ctx);
        // 定点结果转为float
        float* fixed_float_real = (float*)_aligned_malloc(size * sizeof(float), 32);
        float* fixed_float_imag = (float*)_aligned_malloc(size * sizeof(float), 32);
        for (int k = 0; k < size; k++) {
            fixed_float_real[k] = Q15_TO_FLOAT(signal_q15_temp_real[k]);
            fixed_float_imag[k] = Q15_TO_FLOAT(signal_q15_temp_imag[k]);
        }
        calc_fft_error(fixed_float_real, fixed_float_imag, base_real, base_imag, size, &fixed_err);
        print_fft_error("AVX定点版", &fixed_err);

        // ==========================
        // 释放内存
        // ==========================
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
    }

    printf("\nFFT completed.\n");
    return 0;
}