#include "include/fft_lib.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

// ================= 配置参数 =================
#define TEST_N 1024           // 测试点数
#define TEST_AMP 0.01f         // 输入幅度 (0.3是安全区，既不大到溢出，也不小到被淹没)
#define Q 15
#define SCALE (1 << Q)
#define INT16_MAX 32767
#define INT16_MIN (-32768)

// ================= 辅助宏 =================
#define FLOAT_TO_Q15(x) ({ \
    float _temp = (x) * SCALE; \
    _temp += (_temp >= 0 ? 0.5f : -0.5f); \
    if (_temp > INT16_MAX) _temp = INT16_MAX; \
    else if (_temp < INT16_MIN) _temp = INT16_MIN; \
    (int16_t)_temp; \
})

#define Q15_TO_FLOAT(x) ((float)(x) / SCALE)

// ================= 误差分析工具 =================
void analyze_error(const char* test_name, 
                   const float* ref_real, const float* ref_imag, 
                   const int16_t* fix_real, const int16_t* fix_imag, 
                   int N) {
    double error_energy = 0.0;
    double signal_energy = 0.0;
    double max_error = 0.0;
    
    // 1. 计算增益失配 (Gain Mismatch)
    double sum_ref_energy = 0;
    double sum_fix_energy = 0;
    
    for (int i = 0; i < N; i++) {
        float fr = Q15_TO_FLOAT(fix_real[i]);
        float fi = Q15_TO_FLOAT(fix_imag[i]);
        sum_fix_energy += (double)fr*fr + (double)fi*fi;
        sum_ref_energy += (double)ref_real[i]*ref_real[i] + (double)ref_imag[i]*ref_imag[i];
    }
    
    // 计算幅度缩放比 (sqrt of energy ratio)
    double gain_comp = 1.0;
    if (sum_fix_energy > 1e-9) {
        gain_comp = sqrt(sum_ref_energy / sum_fix_energy);
    }

    // 2. 逐点计算误差 (应用增益补偿后)
    for (int i = 0; i < N; i++) {
        float r_ref = ref_real[i];
        float i_ref = ref_imag[i];
        
        // 补偿定点结果的缩放问题
        float r_fix = Q15_TO_FLOAT(fix_real[i]) * (float)gain_comp;
        float i_fix = Q15_TO_FLOAT(fix_imag[i]) * (float)gain_comp;
        
        float diff_r = r_ref - r_fix;
        float diff_i = i_ref - i_fix;
        
        double current_err_sq = diff_r*diff_r + diff_i*diff_i;
        error_energy += current_err_sq;
        signal_energy += r_ref*r_ref + i_ref*i_ref;
        
        if (sqrt(current_err_sq) > max_error) max_error = sqrt(current_err_sq);
    }

    double snr_db = (error_energy > 1e-12) ? 10.0 * log10(signal_energy / error_energy) : 999.0;

    printf("\n--- Test: %s ---\n", test_name);
    printf("  Gain Correction  : x%.4f (Ref/Fix)\n", gain_comp);
    printf("  SNR              : %.2f dB\n", snr_db);
    printf("  Max Abs Error    : %.6f\n", max_error);
    
    if (snr_db > 50) printf("  Result           : [PASS] Excellent\n");
    else if (snr_db > 35) printf("  Result           : [PASS] Good\n");
    else printf("  Result           : [FAIL] Too much distortion\n");
}

// ================= 测试用例 1: 仅正变换 (FFT) =================
void test_forward_fft(FFTContext* ctx, int N) {
    float* f_real = (float*)_aligned_malloc(N * sizeof(float), 32);
    float* f_imag = (float*)_aligned_malloc(N * sizeof(float), 32);
    int16_t* q_real = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);
    int16_t* q_imag = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);

    // 准备时域信号
    for (int i = 0; i < N; i++) {
        float val = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * TEST_AMP;
        f_real[i] = val;
        f_imag[i] = 0.0f;
        q_real[i] = FLOAT_TO_Q15(val);
        q_imag[i] = 0;
        
        // 保持输入完全一致
        f_real[i] = Q15_TO_FLOAT(q_real[i]);
    }

    fft_AVX(f_real, f_imag, N, ctx);
    fft_AVX_fixedP(q_real, q_imag, N, ctx);

    analyze_error("Forward FFT Only", f_real, f_imag, q_real, q_imag, N);

    _aligned_free(f_real); _aligned_free(f_imag);
    _aligned_free(q_real); _aligned_free(q_imag);
}

// ================= 测试用例 2: 仅逆变换 (IFFT) =================
void test_inverse_ifft(FFTContext* ctx, int N) {
    float* f_real = (float*)_aligned_malloc(N * sizeof(float), 32);
    float* f_imag = (float*)_aligned_malloc(N * sizeof(float), 32);
    int16_t* q_real = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);
    int16_t* q_imag = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);

    // 准备频域信号
    for (int i = 0; i < N; i++) {
        float vr = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * 0.5;
        float vi = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * 0.5;
        
        q_real[i] = FLOAT_TO_Q15(vr);
        q_imag[i] = FLOAT_TO_Q15(vi);
        f_real[i] = Q15_TO_FLOAT(q_real[i]);
        f_imag[i] = Q15_TO_FLOAT(q_imag[i]);
    }

    ifft_AVX_reuse(f_real, f_imag, N, ctx);
    ifft_AVX_fixedP(q_real, q_imag, N, ctx);

    analyze_error("Inverse IFFT Only", f_real, f_imag, q_real, q_imag, N);

    _aligned_free(f_real); _aligned_free(f_imag);
    _aligned_free(q_real); _aligned_free(q_imag);
}

// ================= 测试用例 3: 闭环测试 (FFT + IFFT) =================
void test_round_trip(FFTContext* ctx, int N) {
    float* f_real = (float*)_aligned_malloc(N * sizeof(float), 32);
    float* f_imag = (float*)_aligned_malloc(N * sizeof(float), 32);
    int16_t* q_real = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);
    int16_t* q_imag = (int16_t*)_aligned_malloc(N * sizeof(int16_t), 32);

    // 准备初始信号
    for (int i = 0; i < N; i++) {
        float val = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * TEST_AMP;
        q_real[i] = FLOAT_TO_Q15(val);
        q_imag[i] = 0;
        f_real[i] = Q15_TO_FLOAT(q_real[i]); // 浮点基准使用相同的量化后输入
        f_imag[i] = 0;
    }

    // --- Float 路径 ---
    fft_AVX(f_real, f_imag, N, ctx);
    ifft_AVX_reuse(f_real, f_imag, N, ctx);

    // --- Fixed 路径 ---
    fft_AVX_fixedP(q_real, q_imag, N, ctx);
    ifft_AVX_fixedP(q_real, q_imag, N, ctx);

    // 对比最终还原的时域信号
    analyze_error("Round Trip (FFT->IFFT)", f_real, f_imag, q_real, q_imag, N);

    _aligned_free(f_real); _aligned_free(f_imag);
    _aligned_free(q_real); _aligned_free(q_imag);
}

int main() {
    srand(time(NULL));
    FFTContext* ctx = trig_table(TEST_N);
    if (!ctx) {
        printf("Init Failed\n");
        return -1;
    }

    printf("========================================\n");
    printf("Fixed-Point Accuracy Test Suite (N=%d)\n", TEST_N);
    printf("Input Amplitude: %.2f (Safe Range)\n", TEST_AMP);
    printf("========================================\n");

    test_forward_fft(ctx, TEST_N);
    test_inverse_ifft(ctx, TEST_N);
    test_round_trip(ctx, TEST_N);

    printf("\nDone.\n");
    free_trig_table(ctx);
    return 0;
}