#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <stdlib.h>  // 新增：用于malloc和free
#include <immintrin.h>
#include "FFT.h"

#define M_PI 3.14159265358979323846



FFTContext* trig_table(int max_size)
{
    FFTContext* ctx = (FFTContext*)malloc(sizeof(FFTContext));
    ctx->size = max_size;
    ctx->cos_table = (float*)malloc((max_size/2) * sizeof(float));
    ctx->sin_table = (float*)malloc((max_size/2) * sizeof(float));
    for (int i = 0; i < max_size/2; i++) {
        ctx->cos_table[i] = cosf(2 * M_PI * i / max_size);
        ctx->sin_table[i] = sinf(2 * M_PI * i / max_size);
    }
    return ctx;
}

void free_trig_table(FFTContext* ctx) {
    if (ctx) {
        free(ctx->cos_table);
        free(ctx->sin_table);
        free(ctx);
    }
}

void signal_gen(float real[], float imag[], int N) {
    for (int n = 0; n < N; n++) {
        real[n] = sinf(2 * M_PI * n * 10 / N);
        imag[n] = 0.0f;
    }
}


void fft_AVX(float real[], float imag[], int N, FFTContext *ctx)
{
    bit_reverse(real, imag, N);
     __m256i *real_vec = (__m256i *)real;
     __m256i *imag_vec = (__m256i *)imag;
    int m = log2(N);
    for (int s = 1;s<=m;s++)
    {
        int M = 1<<s;
        //M = 2时 cos = 1,sin = 0;
        if (M == 2)
        {
            for (int j = 0; j < N/16; j++)
            {
                __m256i real_val = real_vec[j];
                __m256i imag_val = imag_vec[j];

                
                
                // // 提取偶数和奇数元素
                __m256i even_mask = _mm256_set_epi8(
                        13,12,9,8,5,4,1,0,13,12,9,8,5,4,1,0,
                        13,12,9,8,5,4,1,0,13,12,9,8,5,4,1,0
                );


                __m256i odd_mask = _mm256_set_epi8(
                        15,14,11,10,7,6,3,2,15,14,11,10,7,6,3,2,
                        15,14,11,10,7,6,3,2,15,14,11,10,7,6,3,2
                );

                // 0 2 4 6 0 2 4 6/8 10 12 14 8 10 12 14
                // 1 3 5 7 1 3 5 7/9 11 13 15 9 11 13 15 
                __m256i even_real = _mm256_shuffle_epi8(real_val,even_mask);
                __m256i even_imag = _mm256_shuffle_epi8(imag_val,even_mask);
                __m256i odd_real = _mm256_shuffle_epi8(real_val,odd_mask);
                __m256i odd_imag = _mm256_shuffle_epi8(imag_val,odd_mask);


                // __m256i even_real = _mm256_shuffle_epi32(_mm256_shufflelo_epi16(_mm256_shufflehi_epi16(real_val, _MM_SHUFFLE(2,0,2,0)),_MM_SHUFFLE(2,0,2,0)),_MM_SHUFFLE(2,0,2,0));
                // __m256i even_imag = _mm256_shuffle_epi32(_mm256_shufflelo_epi16(_mm256_shufflehi_epi16(imag_val, _MM_SHUFFLE(2,0,2,0)),_MM_SHUFFLE(2,0,2,0)),_MM_SHUFFLE(2,0,2,0));
                // __m256i odd_real = _mm256_shuffle_epi32(_mm256_shufflelo_epi16(_mm256_shufflehi_epi16(real_val, _MM_SHUFFLE(3,1,3,1)),_MM_SHUFFLE(3,1,3,1)),_MM_SHUFFLE(3,1,3,1));
                // __m256i odd_imag = _mm256_shuffle_epi32(_mm256_shufflelo_epi16(_mm256_shufflehi_epi16(imag_val, _MM_SHUFFLE(3,1,3,1)),_MM_SHUFFLE(3,1,3,1)),_MM_SHUFFLE(3,1,3,1));


                // __m256 even_real = _mm256_shuffle_ps(real_val, real_val, _MM_SHUFFLE(2,0,2,0));
                // __m256 odd_real = _mm256_shuffle_ps(real_val, real_val, _MM_SHUFFLE(3,1,3,1));
                // __m256 even_imag = _mm256_shuffle_ps(imag_val, imag_val, _MM_SHUFFLE(2,0,2,0));
                // __m256 odd_imag = _mm256_shuffle_ps(imag_val, imag_val, _MM_SHUFFLE(3,1,3,1));
                
                // 蝶形运算
                //0+1 2+3 4+5 6+7 0+1 2+3 4+5 6+7 8+9 10+11 12+13 14+15 8+9 10+11 12+13 14+15
                //0-1 2-3 4-5 6-7 0-1 2-3 4-5 6-7 8-9 10-11 12-13 14-15 8-9 10-11 12-13 14-15
                __m256i result_real = _mm256_add_epi16(even_real, odd_real);
                __m256i result_imag = _mm256_add_epi16(even_imag, odd_imag);
                __m256i result_real2 = _mm256_sub_epi16(even_real, odd_real);
                __m256i result_imag2 = _mm256_sub_epi16(even_imag, odd_imag);
                
                // 交错存储结果
                real_vec[j] = _mm256_unpacklo_epi16(result_real, result_real2);
                imag_vec[j] = _mm256_unpacklo_epi16(result_imag, result_imag2);
            }
        }

        else if(M == 4)
        {
            __m256i w_real = _mm256_setr_epi16(1,0,-1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0);
            __m256i w_imag = _mm256_setr_epi16(0,-1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0,1);


            // 提取对应元素
            //0 1 0 1  4 5 4 5 8 9 8 9  12 13 12 13
            __m256i a_mask = _mm256_setr_epi8(
                0,1,2,3,0,1,2,3,8,9,10,11,8,9,10,11,
                0,1,2,3,0,1,2,3,8,9,10,11,8,9,10,11
            );

            // 2 3 2 3 6 7 6 7 10 11 10 11 14 15 14 15
            __m256i b_mask = _mm256_setr_epi8(
                4,5,6,7,4,5,6,7,12,13,14,15,12,13,14,15,
                4,5,6,7,4,5,6,7,12,13,14,15,12,13,14,15
            );
            
            for(int j = 0;j<N/16;j++)
            {
                __m256i real_val = real_vec[j];
                __m256i imag_val = imag_vec[j];

                __m256i a_real = _mm256_shuffle_epi8(real_val,a_mask);
                __m256i a_imag = _mm256_shuffle_epi8(imag_val,a_mask);
                __m256i b_real = _mm256_shuffle_epi8(real_val,b_mask);
                __m256i b_imag = _mm256_shuffle_epi8(imag_val,b_mask);

                __m256i r_real = _mm256_sub_epi16(_mm256_mulhrs_epi16(b_real,w_real),_mm256_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m256i i_imag = _mm256_add_epi16(_mm256_mulhrs_epi16(b_real,w_imag),_mm256_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm256_add_epi16(a_real,r_real);
                imag_vec[j] = _mm256_add_epi16(a_imag,i_imag);
            }



        }

        else if(M == 8)
        {
            __m256i w_real = _mm256_set_epi16(-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0),-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0));//低位正，高位负
            __m256i w_imag = _mm256_set_epi16(-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0),-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0));

            //0 1 2 3 0 1 2 3 2 9 10 11 8 9 10 11
            __m256i a_mask = _mm256_setr_epi8(
                0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,
                0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7
            );

            //4 5 6 7 4 5 6 7 12 13 14 15 12 13 14 15
            __m256i b_mask = _mm256_setr_epi8(
                8,9,10,11,12,13,14,15,8,9,10,11,12,13,14,15,
                8,9,10,11,12,13,14,15,8,9,10,11,12,13,14,15
            );
            for(int j = 0;j<N/16;j++)
            {
                __m256i real_val = real_vec[j];
                __m256i imag_val = imag_vec[j];

                __m256i a_real = _mm256_shuffle_epi8(real_val,a_mask);
                __m256i a_imag = _mm256_shuffle_epi8(imag_val,a_mask);
                __m256i b_real = _mm256_shuffle_epi8(real_val,b_mask);
                __m256i b_imag = _mm256_shuffle_epi8(imag_val,b_mask);

                __m256i r_real = _mm256_sub_epi16(_mm256_mulhrs_epi16(b_real,w_real),_mm256_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m256i i_imag = _mm256_add_epi16(_mm256_mulhrs_epi16(b_real,w_imag),_mm256_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm256_add_epi16(a_real,r_real);
                imag_vec[j] = _mm256_add_epi16(a_imag,i_imag);
            }



        }

        else if(m == 16)
        {

            __m256i w_real = _mm256_set_epi16(-cosf(2*M_PI/M*7),-cosf(2*M_PI/M*6),-cosf(2*M_PI/M*5),-cosf(2*M_PI/M*4),-cosf(2*M_PI/M*3),-cosf(2*M_PI/M*2),-cosf(2*M_PI/M*1),-cosf(2*M_PI/M*0),cosf(2*M_PI/M*7),cosf(2*M_PI/M*6),cosf(2*M_PI/M*5),cosf(2*M_PI/M*4),cosf(2*M_PI/M*3),cosf(2*M_PI/M*2),cosf(2*M_PI/M*1),cosf(2*M_PI/M*0));//低位正，高位负
            __m256i w_imag = _mm256_set_epi16(-sinf(-2*M_PI/M*7),-sinf(-2*M_PI/M*6),-sinf(-2*M_PI/M*5),-sinf(-2*M_PI/M*4),-sinf(-2*M_PI/M*3),-sinf(-2*M_PI/M*2),-sinf(-2*M_PI/M*1),-sinf(-2*M_PI/M*0),sinf(-2*M_PI/M*7),sinf(-2*M_PI/M*6),sinf(-2*M_PI/M*5),sinf(-2*M_PI/M*4),sinf(-2*M_PI/M*3),sinf(-2*M_PI/M*2),sinf(-2*M_PI/M*1),sinf(-2*M_PI/M*0));

            for(int j = 0;j<N/16;j++)
            {
                __m256i real_val = real_vec[j];
                __m256i imag_val = imag_vec[j];
                //0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
                __m256i a_real = _mm256_permute2x128_si256(real_val,real_val,0x00);
                __m256i a_imag = _mm256_permute2x128_si256(imag_val,imag_val,0x00);
                //8 9 10 11 12 13 14 15 8 9 10 11 12 13 14 15
                __m256i b_real = _mm256_permute2x128_si256(real_val,real_val,0x11);
                __m256i b_imag = _mm256_permute2x128_si256(imag_val,imag_val,0x11);

                __m256i r_real = _mm256_sub_epi16(_mm256_mulhrs_epi16(b_real,w_real),_mm256_mulhrs_epi16(b_imag,w_imag));//可用FMA优化 
                __m256i i_imag = _mm256_add_epi16(_mm256_mulhrs_epi16(b_real,w_imag),_mm256_mulhrs_epi16(b_imag,w_real));//可用FMA优化

                real_vec[j] = _mm256_add_epi16(a_real,r_real);
                imag_vec[j] = _mm256_add_epi16(a_imag,i_imag);
            }





        }


        else{

            for(int k = 0;k<N/16;k+=M/16)
            {

                for(int j = 0;j<M/2/16;j++)
                {
                    int idx = j * (ctx->size / M);

                    __m256i w_real = _mm256_set_epi16(ctx->cos_table[idx+15],ctx->cos_table[idx+14],ctx->cos_table[idx+13],ctx->cos_table[idx+12],ctx->cos_table[idx+11],ctx->cos_table[idx+10],ctx->cos_table[idx+9],ctx->cos_table[idx+8],ctx->cos_table[idx+7],ctx->cos_table[idx+6],ctx->cos_table[idx+5],ctx->cos_table[idx+4],ctx->cos_table[idx+3],ctx->cos_table[idx+2],ctx->cos_table[idx+1],ctx->cos_table[idx]);
                    __m256i w_imag = _mm256_set_epi16(-ctx->sin_table[idx+15],-ctx->sin_table[idx+14],-ctx->sin_table[idx+13],-ctx->sin_table[idx+12],-ctx->sin_table[idx+11],-ctx->sin_table[idx+10],-ctx->sin_table[idx+9],-ctx->sin_table[idx+8],-ctx->sin_table[idx+7],-ctx->sin_table[idx+6],-ctx->sin_table[idx+5],-ctx->sin_table[idx+4],-ctx->sin_table[idx+3],-ctx->sin_table[idx+2],-ctx->sin_table[idx+1],-ctx->sin_table[idx]);


                    __m256i real_val1 = real_vec[k+j];
                    __m256i imag_val1 = imag_vec[k+j];
                    __m256i real_val2 = real_vec[k+j+M/2/8];
                    __m256i imag_val2 = imag_vec[k+j+M/2/8];


                    __m256i r_real = _mm256_sub_epi16(_mm256_mulhrs_epi16(real_val2,w_real),_mm256_mulhrs_epi16(imag_val2,w_imag));//可用FMA优化 
                    __m256i i_imag = _mm256_add_epi16(_mm256_mulhrs_epi16(real_val2,w_imag),_mm256_mulhrs_epi16(imag_val2,w_real));//可用FMA优化

                    real_vec[k+j] = _mm256_add_epi16(real_val1,r_real);
                    imag_vec[k+j] = _mm256_add_epi16(imag_val1,i_imag);
                    real_vec[k+j+M/2/16] = _mm256_sub_epi16(real_val1,r_real);
                    imag_vec[k+j+M/2/16] = _mm256_sub_epi16(imag_val1,i_imag);




                }
            }






        }



    }
}



