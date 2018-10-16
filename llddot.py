#!/usr/bin/python
import numpy
import ctypes
import os
import sys
from time import time
from time import sleep


#from pyscf.lib.numpy_helper import ddot #testing purpose
np = numpy
_np_helper = ctypes.CDLL(r'./libnp_helper.so')
np.random.seed(0)

#Compact version of ddot from pyscf
def cddot(a, b, alpha=1, c=None, beta=0, ):
    '''Matrix-matrix multiplication for double precision arrays
    '''
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if a.flags.c_contiguous:
        trans_a = 'N'
    elif a.flags.f_contiguous:
        trans_a = 'T'
        a = a.T
    else:
        a = numpy.asarray(a, order='C')
        trans_a = 'N'
        #raise ValueError('a.flags: %s' % str(a.flags))

    assert(k == b.shape[0])
    if b.flags.c_contiguous:
        trans_b = 'N'
    elif b.flags.f_contiguous:
        trans_b = 'T'
        b = b.T
    else:
        b = numpy.asarray(b, order='C')
        trans_b = 'N'
        #raise ValueError('b.flags: %s' % str(b.flags))

    if c is None:
        c = numpy.empty((m,n))
        beta = 0
    else:
        assert(c.shape == (m,n))

    offseta=0
    offsetb=0
    offsetc=0
    
    if a.size == 0 or b.size == 0:
        if beta == 0:
            c[:] = 0
        else:
            c[:] *= beta
        return c

    assert(a.flags.c_contiguous)
    assert(b.flags.c_contiguous)
    assert(c.flags.c_contiguous)

    _np_helper.NPdgemm(ctypes.c_char(trans_b.encode('ascii')),
                       ctypes.c_char(trans_a.encode('ascii')),
                       ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
                       ctypes.c_int(b.shape[1]), ctypes.c_int(a.shape[1]),
                       ctypes.c_int(c.shape[1]),
                       ctypes.c_int(offsetb), ctypes.c_int(offseta),
                       ctypes.c_int(offsetc),
                       b.ctypes.data_as(ctypes.c_void_p),
                       a.ctypes.data_as(ctypes.c_void_p),
                       c.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_double(alpha), ctypes.c_double(beta))
    return c


def mpi_lddot(a, b):
    assert(len(a) == len(b))
    matrix_num = len(a)
    c = []   
    ptr2a = np.empty(matrix_num, dtype=ctypes.c_void_p)
    ptr2b = np.empty(matrix_num, dtype=ctypes.c_void_p)
    ptr2c = np.empty(matrix_num, dtype=ctypes.c_void_p)  
    m_arr = np.empty(matrix_num, dtype=np.int32)  
    k_arr = np.empty(matrix_num, dtype=np.int32)  
    n_arr = np.empty(matrix_num, dtype=np.int32)
    tra_a = np.empty(matrix_num, dtype=bytes)
    tra_b = np.empty(matrix_num, dtype=bytes)
    offfseta = np.zeros(matrix_num, dtype=np.int32)    
    offfsetb = np.zeros(matrix_num, dtype=np.int32)    
    offfsetc = np.zeros(matrix_num, dtype=np.int32)    
    alpha = np.ones(matrix_num, dtype=np.float64)  
    beta = np.zeros(matrix_num, dtype=np.float64)  
    for i in range(matrix_num):
        m_arr[i] = a[i].shape[0]
        k_arr[i] = a[i].shape[1]
        n_arr[i] = b[i].shape[1]
        if a[i].flags.c_contiguous:
            tra_a[i] = 'N'.encode('ascii')
        elif a[i].flags.f_contiguous:
            tra_a[i] = 'T'.encode('ascii')
            a[i] = a[i].T
        else:
            a[i] = np.asarray(a[i], order='C')
            tra_a[i] = 'N'.encode('ascii')
            #raise ValueError('a.flags: %s' % str(a.flags))   
        assert(k_arr[i] == b[i].shape[0])
        
        if b[i].flags.c_contiguous:
            tra_b[i] = 'N'.encode('ascii')
        elif b.flags.f_contiguous:
            tra_b[i] = 'T'.encode('ascii')
            b[i] = b[i].T
        else:
            b[i] = np.asarray(b[i], order='C')
            tra_b[i] = b'N'
        #raise ValueError('b.flags: %s' % str(b.flags))   
        c.append(np.empty((m_arr[i], n_arr[i]), dtype=np.float64))
        assert(a[i].flags.c_contiguous)
        assert(b[i].flags.c_contiguous)
    
        ptr2a[i] = a[i].ctypes.data
        ptr2b[i] = b[i].ctypes.data               
        ptr2c[i] = c[i].ctypes.data
         
    #END FOR                   
                        
    _np_helper.mpi_lNPdgemm(tra_b.ctypes.data_as(ctypes.c_void_p),
                        tra_a.ctypes.data_as(ctypes.c_void_p),
                        n_arr.ctypes.data_as(ctypes.c_void_p),
                        m_arr.ctypes.data_as(ctypes.c_void_p),
                        k_arr.ctypes.data_as(ctypes.c_void_p),
                        n_arr.ctypes.data_as(ctypes.c_void_p),
                        k_arr.ctypes.data_as(ctypes.c_void_p),
                        n_arr.ctypes.data_as(ctypes.c_void_p),
                        offfsetb.ctypes.data_as(ctypes.c_void_p),
                        offfseta.ctypes.data_as(ctypes.c_void_p),                        
                        offfsetc.ctypes.data_as(ctypes.c_void_p),  
                        ptr2b.ctypes.data_as(ctypes.c_void_p),     
                        ptr2a.ctypes.data_as(ctypes.c_void_p),
                        ptr2c.ctypes.data_as(ctypes.c_void_p),
                        alpha.ctypes.data_as(ctypes.c_void_p),
                        beta.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(matrix_num))
    return c

def mpi_print(l, a):
    _np_helper.mpi_print(ctypes.c_int(l), a.ctypes.data_as(ctypes.c_void_p))
    
def mpi_init():
    _np_helper.mpi_init()
    if int(os.environ['PMI_RANK']) != 0:
        exit(0)
        
def mpi_final():
    _np_helper.mpi_final()

if __name__ == '__main__':
    mpi_init()
    
    l = 1000
    m = 0
    k = 0
    n = 0
    low=100
    hig=101
    interval = 1
    repeats = 7
    a = []
    b = []
    print "Start to generate ramdom square matrix list"
    T_gen = time()
    for i in range(l):
        m = np.random.randint(low, high=hig)
        k = np.random.randint(low, high=hig)        
        n = np.random.randint(low, high=hig)        
        a.append(np.random.rand(m,k) * 2)
        b.append(np.random.rand(k,n) * 2)
        #a.append(np.array(np.array(np.random.rand(m, k), dtype = np.int32), dtype = np.float64))
        #b.append(np.array(np.array(np.random.rand(k, n), dtype = np.int32), dtype = np.float64))
        #a.append(np.array(np.arange(m * k), dtype=np.float64).reshape(m, k) + 2 * i)
        #b.append(np.array(np.arange(n * k), dtype=np.float64).reshape(k, n) + 2 * i + 1)        
    print "Finish generating the random matrix of length:", l, "and size:",m, k, n
    print "Time used:", time() - T_gen
    #for i in a:
    #    print i
    #    print "---"
    print "Running test cases for", repeats, "cycles"
    
    print "Sleeping for", interval, " seconds to settle the CPU usage"
    sleep(interval)
    print "Start cddot for list"
    T_cddots = time()
    for _ in range(repeats):
        r_cddot = None
        r_cddot = list(map(lambda x: cddot(x[0], x[1]), zip(a, b)))
    T_cddote = time()  
     
    print "Sleeping for", interval, " seconds to settle the CPU usage"
    sleep(interval)
    print "Start lddot for list"
    T_lddots = time()
    for _ in range(repeats):
        r_lddot = None
        r_lddot = mpi_lddot(a, b)
    T_lddote = time()
    
    
    print "Using cddot time: ", T_cddote - T_cddots, ", average:", (T_cddote - T_cddots) / repeats
    print "Using lddot time:", T_lddote - T_lddots, ", average", (T_lddote - T_lddots) / repeats
    
    for i in range(l):
        #print "ddot: "
        #print r_ddot[i]
        #print "lldot: "
        #print r_lddot[i]
        if not (r_cddot[i] == r_lddot[i]).all():
            print i, "NOT EQUAL"
            print "INPUT A:"
            print a[i]
            print "----------"
            print "INPUT B:"
            print b[i]
            print "----------"
            print "DDOT:"
            print r_cddot[i]
            print "----------"
            print "LDDOT:"
            print r_lddot[i]
            print "END OF NOT EQUAL"
            
            
    #for i in a:
    #    print i
    #    print "---"    
    mpi_final()

