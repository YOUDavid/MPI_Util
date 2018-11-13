mpih = mpi_helper.h 
omph = omp_helper.h
sharedh = shared_library.h shared_types.h
objfile =  mpi_lNPdgemm.o mpi_print.o mpi_util.o omp_util.o #mpi_lOMPdgemm.o  test_wrapper.o mpi_ldgemm.o

TARGET = mpi_util.so

SOURCEDIR = source
HEADERDIR = header
OBJDIR = objective
SOBJDIR = shared_objective
CC = mpiicc
INC = -I./header
OFLAG = $(INC) -Wall -fopenmp -pthread -fPIC -O3 -DMACRO=_GNU_SOURCE -lmkl -c -o
SOFLAG = $(INC) -Wall -fopenmp -pthread -fPIC -O3 -DMACRO=_GNU_SOURCE -shared -o 
OBJS = $(foreach o, $(objfile), $(OBJDIR)/$o)
	
$(OBJDIR)/%.o: $(SOURCEDIR)/%.c $(HEADERDIR)/*.h
	$(CC) $(OFLAG) $@ $<
    
$(SOBJDIR)/$(TARGET): $(OBJS)
	$(CC) $(SOFLAG) $@ $(OBJS)
	
.PRECIOUS: $(OBJDIR)/*.o 
	
.PHONY: clean remake
debug:
	make $(SOBJDIR)/$(TARGET)
	cp $(SOBJDIR)/mpi_util.so ./mpi_util.so

remake:
	make clean
	make debug

clean:
	rm -f ./*.so
	rm -f $(SOBJDIR)/*.so
	rm -f $(OBJDIR)/*.o
