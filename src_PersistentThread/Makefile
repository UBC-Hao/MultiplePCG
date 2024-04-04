NVCC      = /cs/local/lib/pkg/cudatoolkit-11.6.2/bin/nvcc


hw4_lib.o: hw4_lib.cu hw4_lib.h
	$(NVCC) -c -g hw4_lib.cu

jacobi: jacobi_p.cu hw4_lib.o 
	$(NVCC) -o jacobi jacobi_p.cu  hw4_lib.o


clean:
	rm -f *.o 
