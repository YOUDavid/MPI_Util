#!/usr/bin/python
import numpy
import ctypes
import os
import sys
from time import time
from time import sleep
from multiprocessing import Pool

#from pyscf.lib.numpy_helper import ddot #testing purpose
nprocs = 16
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

def cddot2(matrices):
    return cddot(matrices[0], matrices[1]);

def pmmul(matrix1,matrix2):
    P = Pool(processes=nprocs,maxtasksperchild=len(matrix1)/nprocs)
    result = P.map(cddot2, zip(matrix1,matrix2))
    P.close()
    P.join()
    return result


if __name__ == '__main__':
    
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
    print "Start pmmul for list"
    T_pmmuls = time()
    for _ in range(repeats):
        r_pmmul = None
        r_pmmul = pmmul(a, b)
    T_pmmule = time()  
    
    
    print "Using pmmul time: ", T_pmmule - T_pmmuls, ", average:", (T_pmmule - T_pmmuls) / repeats
    

