// 数据层：初始化、三角函数表生成、重排逻辑
#include "fft_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>


FFTContext* trig_table(int max_size)
{
    FFTContext* ctx = (FFTContext*)malloc(sizeof(FFTContext));
    ctx->size = max_size;
    ctx->cos_table = (float*)_aligned_malloc((max_size/2) * sizeof(float),32);
    ctx->sin_table = (float*)_aligned_malloc((max_size/2) * sizeof(float),32);
    ctx->cos_t = (int16_t*)_aligned_malloc((max_size/2) * sizeof(int16_t),32);
    ctx->sin_t = (int16_t*)_aligned_malloc((max_size/2) * sizeof(int16_t),32);
    ctx->pos = (int*)_aligned_malloc(max_size * sizeof(int),32);
    ctx->pos[0] = 0;
    for (int i = 1; i < max_size; i++) {
        ctx->pos[i] = ctx->pos[i / 2] / 2 + (i % 2) * max_size / 2;
    }
    
    for (int i = 0; i < max_size/2; i++) {
        ctx->cos_table[i] = cosf(2 * M_PI * i / max_size);
        ctx->sin_table[i] = sinf(2 * M_PI * i / max_size);
        ctx->cos_t[i] = FLOAT_TO_Q15(ctx->cos_table[i]);
        ctx->sin_t[i] = FLOAT_TO_Q15(ctx->sin_table[i]);

    }
    //重排旋转因子 用于FFT中顺序读取
    int m = log2(max_size);
    ctx->stage_offsets = (int*)malloc((m + 1) * sizeof(int));
    memset(ctx->stage_offsets, -1, (m + 1) * sizeof(int));
    // 计算重排表所需的总大小
    size_t total_twiddles = 0;
    for (int s = 4; s <= m; s++) { // 从 M=16 (s=4) 开始需要通用加载
        // int M = 1 << s;
        // 在这一阶段，内层循环(j)会执行 N/M * (M/2/8) = N/16 次加载
        total_twiddles += (max_size / 16); 
    }
    
    ctx->shuffled_cos_table = (float*)_aligned_malloc(total_twiddles * 8 * sizeof(float), 32);
    ctx->shuffled_sin_table = (float*)_aligned_malloc(total_twiddles * 8 * sizeof(float), 32);
    ctx->shuffled_cos_t = (int16_t*)_aligned_malloc(total_twiddles * 8 * sizeof(int16_t), 32);
    ctx->shuffled_sin_t = (int16_t*)_aligned_malloc(total_twiddles * 8 * sizeof(int16_t), 32);
    // 开始填充重排表
    size_t current_offset = 0;
    ctx->stage_offsets[0] = 0;
    for (int s = 1; s <= m; s++) {
        ctx->stage_offsets[s] = current_offset;
        
        // 我们只为需要跨步访问的通用阶段进行重排
        // M=2, 4, 8 可以硬编码，不需要查表
        if (s <= 3) continue; // M <= 8

        int M = 1 << s;
        int step = max_size / M;

        // 模拟fft_AVX的循环来填充数据
        for (int k = 0; k < max_size / 8; k += M / 8) {
            for (int j = 0; j < M / 2 / 8; j++) {
                int idx = j * 8 * step;

                // 把将来要访问的8个跨步值，现在就连续存起来
                for (int i = 0; i < 8; i++) {
                    ctx->shuffled_cos_table[current_offset + i] = ctx->cos_table[idx + i * step];
                    // 存 -sin 的值，这样在fft中就不用再取反了
                    ctx->shuffled_sin_table[current_offset + i] = -ctx->sin_table[idx + i * step];
                    ctx->shuffled_cos_t[current_offset + i] = FLOAT_TO_Q15(ctx->cos_table[idx + i * step]);
                    // 存 -sin 的值，这样在fft中就不用再取反了
                    ctx->shuffled_sin_t[current_offset + i] = FLOAT_TO_Q15(-ctx->sin_table[idx + i * step]);
                }
                current_offset += 8; // 移动到下一个存储位置
            }
        }
    }
    //




    return ctx;
}

void free_trig_table(FFTContext* ctx) {
    if (ctx) {
        _aligned_free(ctx->cos_table);
        _aligned_free(ctx->sin_table);
        _aligned_free(ctx->cos_t);
        _aligned_free(ctx->sin_t);
        _aligned_free(ctx->pos);
        free(ctx);
    }
}