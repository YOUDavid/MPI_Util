mpih = mpi_helper.h 
omph = omp_helper.h
sharedh = shared_library.h shared_types.h
objfile =  mpi_lNPdgemm.o mpi_lpresidue.o mpi_print.o blas_util.o mpi_util.o omp_util.o #mpi_lOMPdgemm.o   mpi_ldgemm.o

TARGET = mpi_util.so

SOURCEDIR = source
HEADERDIR = header
OBJDIR = objective
SOBJDIR = shared_objective
CC = mpiicc
INC = -I./header
OFLAG = $(INC) -Wall -mt_mpi -fopenmp -pthread -fPIC -O3 -DMACRO=_GNU_SOURCE -mkl  -c  -o
TFLAG = $(INC) -Wall -mt_mpi  -fopenmp -pthread -O3 -DMACRO=_GNU_SOURCE -mkl -o
SOFLAG = $(INC) -Wall -mt_mpi -fopenmp -pthread -fPIC -O3 -DMACRO=_GNU_SOURCE -shared -o 
OBJS = $(foreach o, $(objfile), $(OBJDIR)/$o)
	
$(OBJDIR)/%.o: $(SOURCEDIR)/%.c $(HEADERDIR)/*.h
	$(CC) $(OFLAG) $@ $<
    
$(SOBJDIR)/$(TARGET): $(OBJS)
	$(CC) $(SOFLAG) $@ $(OBJS)
	
.PRECIOUS: $(OBJDIR)/*.o 
	
.PHONY: clean remake test
debug:
	make $(SOBJDIR)/$(TARGET)
	cp $(SOBJDIR)/mpi_util.so ./mpi_util.so
	cp $(SOBJDIR)/mpi_util.so ../OSV-BOMD-SMP-pool-new/mpi_util.so

	
test: $(OBJDIR)/blas_util.o $(OBJDIR)/omp_util.o 
	icc -g $(OFLAG) $(OBJDIR)/test_wrapper.o  $(SOURCEDIR)/test_wrapper.c
	icc -g $(TFLAG) test.out $(OBJDIR)/test_wrapper.o $(OBJDIR)/blas_util.o $(OBJDIR)/omp_util.o 
	rm -f $(OBJDIR)/test_wrapper.o
	
	
remake:
	make clean
	make debug

clean:
	rm -f ./*.so
	rm -f $(SOBJDIR)/*.so
	rm -f $(OBJDIR)/*.o
