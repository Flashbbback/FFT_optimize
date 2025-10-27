#include <stdio.h>
#include <immintrin.h>
#include <windows.h>
#include <math.h>

#define M_PI 3.14159265358979323846

void add_vectors(__m256 *a,__m256 *b,int n)
{
    __m256 result;
    for (int i = 0;i<n;i++)
    {
        result = _mm256_add_ps(_mm256_mul_ps(a[i], a[i]), _mm256_mul_ps(b[i], b[i]));
        // result = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(a[i], a[i]), _mm256_mul_ps(b[i], b[i])));
        // float d[8];
        // _mm256_store_ps(d,result);
        // printf("Result %d: %f\n", i, d[0]);
    }
    return;

}


void add(float *a, float *b, int n)
{
    float c;
    for (int i = 0; i < n; i++)
    {
        c = a[i]*a[i] + b[i]*b[i];
        // c = sqrt(a[i]*a[i] + b[i]*b[i]);
    }
}

void signal_gen(float real[], float imag[], int N) {
    for (int n = 0; n < N; n++) {
        real[n] = sinf(2 * M_PI * n * 10 / N);
        imag[n] = cosf(2 * M_PI * n * 10 / N);
    }
}

int main()
{


    int repeat = 1000;
    int N = 1 << 20;
    LARGE_INTEGER start1, end1, frequency;
    QueryPerformanceFrequency(&frequency);

    // 替换原来的malloc
    float *real = (float *)_aligned_malloc(N * sizeof(float), 32);  // 32字节对齐
    float *imag = (float *)_aligned_malloc(N * sizeof(float), 32);
    signal_gen(real, imag, N);
    QueryPerformanceCounter(&start1);
    __m256 *real_vec = (__m256 *)real;
    __m256 *imag_vec = (__m256 *)imag;
    for (int i = 0; i < repeat; i++)
    {
        add_vectors(real_vec, imag_vec, N/8);
    }
    QueryPerformanceCounter(&end1);
    float cpu_time_used = ((float) (end1.QuadPart - start1.QuadPart)) / frequency.QuadPart;
    printf("Time taken: %f seconds with AVX\n", cpu_time_used);

    LARGE_INTEGER start2, end2;
    signal_gen(real, imag, N);
    QueryPerformanceCounter(&start2);
    for (int i = 0; i < repeat; i++)
    {
        add(real, imag, N);
    }
    QueryPerformanceCounter(&end2);
    float cpu_time_used1 = ((float) (end2.QuadPart - start2.QuadPart)) / frequency.QuadPart;
    printf("Time taken: %f seconds without AVX\n", cpu_time_used1);
    _aligned_free(real);
    _aligned_free(imag);

    return 0;

}