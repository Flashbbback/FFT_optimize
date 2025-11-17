avx_fft_demo: example/demo.c src/FFT.c
	gcc -o avx_fft_demo example/demo.c src/FFT.c -mavx2 -O2 -lm

clean:
	del *.o *.s *.i avx_fft_demo.exe