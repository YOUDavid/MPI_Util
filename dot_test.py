#!/usr/bin/python
if __name__ == '__main__':
    ###A.T @ B
    #$OMP_DYNAMIC      $OMP_NESTED              $OMP_NUM_THREADS 
    #$MKL_BLAS         $MKL_DOMAIN_NUM_THREADS  $MKL_ENABLE_INSTRUCTIONS  
    #$MKL_NUM_THREADS  $MKL_THREADING_LAYER      
    #$MKL_CBWR         $MKL_DYNAMIC             $MKL_NUM_STRIPES          
    import ctypes
    import os
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
#    print mkl_rt.MKL_Get_Max_Threads()
    mkl_rt.MKL_Set_Dynamic(0)
    mkl_rt.MKL_Set_Num_Threads(16)
#    print mkl_rt.MKL_Get_Max_Threads()

    #os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'SSE4_1'
    #os.environ['MKL_BLAS'] = '4'
    os.environ['MKL_CBWR'] = 'AUTO'#os.environ['MKL_ENABLE_INSTRUCTIONS']
    #os.environ['MKL_NUM_STRIPES'] = '-1'
    os.environ['MKL_THREADING_LAYER'] = 'INTEL'
    os.environ["MKL_VERBOSE"] = '1'
    
    import numpy as np
    from scipy.linalg import blas as bl
    
    
    import sys
    from time import time, sleep, clock
    from functools import reduce, partial, update_wrapper #wraps
    from pyscf.lib.numpy_helper import ddot
    from pyscf.lib import logger
    from multiprocessing import Pool
    
    def mydot(a,b):
        if a.flags.c_contiguous:
            a_T = True
            a = a.T
        elif a.flags.f_contiguous:
            a_T = False
        else:
            a_T = False
            a = np.asfortranarray(a)
    
        if b.flags.c_contiguous:
            b_T = True
            b = b.T
        elif b.flags.f_contiguous:
            b_T = False
        else:
            b_T = False
            b = np.asfortranarray(b)
        return bl.dgemm(alpha=1.0, a=a, b=b, trans_a=a_T, trans_b=b_T)   
    
    def mytdot(a,b):
        if a.flags.c_contiguous:
            a_T = False
            a = a.T
        elif a.flags.f_contiguous:
            a_T = True
        else:
            a_T = True
            a = np.asfortranarray(a)
    
        if b.flags.c_contiguous:
            b_T = False
            b = b.T
        elif b.flags.f_contiguous:
            b_T = True
        else:
            b_T = True
            b = np.asfortranarray(b)
        return bl.dgemm(alpha=1.0, a=b, b=a, trans_a=b_T, trans_b=a_T).T   
    
    def naivedot(a,b):
        return bl.dgemm(alpha=1.0, a=a, b=b)
    
    
    log = logger.Logger(verbose=5)
    np.random.seed(1)
    a = np.random.rand(4000, 4000) * 2
    alist = [a] * 8
    #b = np.random.rand(2000,2000)
    
    
    
    
    
    t0 = clock(), time()
    c = reduce(np.dot, alist)
    log.timer("np.dot", *t0)
    
    
    
    result = c    
    c = None
    t0 = clock(), time()
    c = reduce(ddot, alist)
    log.timer("ddot", *t0)
    #print abs(c - result).sum()
    
#    c = None
#    t0 = clock(), time()
#    c = reduce(naivedot, alist)#bl.dgemm(alpha=1.0, a=a, b=a)
#    log.timer("naive dgemm", *t0)
#    print abs(c - result).sum()
#    
#    c = None
#    t0 = clock(), time()
#    c = reduce(mytdot, alist)#bl.dgemm(alpha=1.0, a=a.T, b=b.T, trans_b=True)
#    log.timer("wrapped dgemm", *t0)
#    print  abs(c - result).sum()
#    
#    c = None
#    t0 = clock(), time()
#    c = reduce(mydot, alist)#bl.dgemm(alpha=1.0, a=b.T, b=a.T, trans_b=True).T
#    log.timer("c=(b.T@a).T dgemm", *t0)
#    print  abs(c - result).sum()
