NVCC  = /cs/local/lib/pkg/cudatoolkit-11.6.2/bin/nvcc

%.beam: %.erl
	erlc -o . $<

test_gpu: jacobi_gpu
	./jacobi_gpu
jacobi_gpu: jacobi_gpu.cu
	$(NVCC) -o $@ $< 

all : libjacobi.so tools.o poisson.o jacobi_cpu.o tools_nif.o mat_nif.so main.beam mat.beam workers.beam wtree.beam

mat_nif.so: tools_nif.c tools.c tools.o poisson.o
	gcc -o mat_nif.so -fpic -shared tools_nif.c jacobi_cpu.o tools.o poisson.o -llapack -lcudart -lcudadevrt -L. -ljacobi

tools_nif.o: tools_nif.c
	gcc -c -fpic tools_nif.c

jacobi_cpu.o: jacobi_cpu.c
	gcc -c -fpic jacobi_cpu.c

poisson.o: poisson.c
	gcc -c -fpic poisson.c

tools.o: tools.c tools.h 
	gcc -c -fpic tools.c

libjacobi.so: jacobi_gpu.cu
	$(NVCC) -shared -o libjacobi.so jacobi_gpu.cu --compiler-options '-fPIC' -lcublas

test_cpu: all
	erl -boot start_clean -s main mpcg

clean:
	rm -f *.o *.so *.beam *.log *.txt