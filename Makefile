ldb: libnp_helper.c
	mpicc -Wall -fopenmp -shared libnp_helper.c -fPIC -lblas -o liblnp_helper.so

lsb: libnp_helper.c libblas.a
	mpicc -Wall -fopenmp -shared libnp_helper.c -fPIC -l:libblas.a -o liblnp_helper.so

mkl: libnp_helper.c
	mpiicc -Wall -fopenmp -shared libnp_helper.c -fPIC /usr/lib64/atlas/libf77blas.so.3 -o libnp_helper.so
