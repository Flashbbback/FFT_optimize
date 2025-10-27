FFT:main.o fft.o
	gcc -o FFT main.o fft.o -lm

main.o:main.s
	gcc -c main.s 

fft.o:fft.s
	gcc -c fft.s

main.s:main.i
	gcc -S main.i -O3 -march=native

fft.s:fft.i
	gcc -S fft.i -O3 -march=native

main.i:main.c
	gcc -E main.c -o main.i 

fft.i:FFT.c
	gcc -E FFT.c -o fft.i 

clean:
	del *.o *.s *.i FFT.exe