build:
	mpicxx -fopenmp -c main.c -o main.o 
	mpicxx -fopenmp -c mpi_functions.c -o mpi_functions.o 
	/usr/local/cuda/bin/nvcc -c -Xcompiler -fopenmp cuda_functions.cu -o cuda_functions.o
	mpicxx -fopenmp -o mpiCudaOpemMP main.o mpi_functions.o cuda_functions.o /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt -lgomp

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP $(arg1) $(arg2)
