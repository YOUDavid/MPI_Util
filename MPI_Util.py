#!/usr/bin/python
import numpy as np
import ctypes
import os
import sys
from functools import reduce, partial, update_wrapper #wraps
mpi_util = ctypes.CDLL(r'mpi_util.so')

def mpi_init():
    mpi_util.mpi_init()
    if int(os.environ['PMI_RANK']) != 0:
        exit(0)
        
def mpi_final():
    mpi_util.mpi_final()
    
def mpi_setONT(t):
    try:
        t = int(t)
    except ValueError:
        t = 1
        print "Warning: Invalid value for mpi_setONT(), using 1 as input"
    mpi_util.mpi_setONT(ctypes.c_int(t))

def _imple_protocol(protocol, real_func):
    return_func = partial(protocol, func=real_func)
    return_func.__name__ = real_func.__name__
    return return_func

def _mpi_dot_protocol(a, b, a_T=0, b_T=0, func=mpi_util.mpi_lNPdgemm):
    assert(len(a) == len(b))
    matrix_num = len(a)
    c = []   
    ptr2a = np.empty(matrix_num, dtype=ctypes.c_void_p)
    ptr2b = np.empty(matrix_num, dtype=ctypes.c_void_p)
    ptr2c = np.empty(matrix_num, dtype=ctypes.c_void_p)  
    m_arr = np.empty(matrix_num, dtype=np.int32)  
    k_arr = np.empty(matrix_num, dtype=np.int32)  
    n_arr = np.empty(matrix_num, dtype=np.int32)
    lda_arr = np.empty(matrix_num, dtype=np.int32)  
    ldb_arr = np.empty(matrix_num, dtype=np.int32)  
    ldc_arr = np.empty(matrix_num, dtype=np.int32)
    tra_a = np.empty(matrix_num, dtype=bytes)
    tra_b = np.empty(matrix_num, dtype=bytes)
    completed = np.zeros(matrix_num, dtype=np.int32)
    offfseta = np.zeros(matrix_num, dtype=np.int32)    
    offfsetb = np.zeros(matrix_num, dtype=np.int32)    
    offfsetc = np.zeros(matrix_num, dtype=np.int32)    
    alpha = np.ones(matrix_num, dtype=np.float64)  
    beta = np.zeros(matrix_num, dtype=np.float64)  
    for i in range(matrix_num):
        if a_T == 0:
            m_arr[i] = a[i].shape[0]
            k_arr[i] = a[i].shape[1]
            if a[i].flags.c_contiguous:
                tra_a[i] = 'N'.encode('ascii')
                lda_arr[i] = a[i].shape[1]
            elif a[i].flags.f_contiguous:
                tra_a[i] = 'T'.encode('ascii')
                lda_arr[i] = a[i].shape[0]
            else:
                a[i] = np.asarray(a[i], order='C')
                tra_a[i] = 'N'.encode('ascii')
                lda_arr[i] = a[i].shape[1]
        else:
            m_arr[i] = a[i].shape[1]
            k_arr[i] = a[i].shape[0]
            if a[i].flags.c_contiguous:
                tra_a[i] = 'T'.encode('ascii')
                lda_arr[i] = a[i].shape[1]
            elif a[i].flags.f_contiguous:
                tra_a[i] = 'N'.encode('ascii')
                lda_arr[i] = a[i].shape[0]
            else:
                a[i] = np.asarray(a[i], order='C')
                tra_a[i] = 'T'.encode('ascii')
                lda_arr[i] = a[i].shape[1]

        if b_T == 0:
            assert(k_arr[i] == b[i].shape[0])
            n_arr[i] = b[i].shape[1]
            if b[i].flags.c_contiguous:
                tra_b[i] = 'N'.encode('ascii')
                ldb_arr[i] = b[i].shape[1]
            elif b[i].flags.f_contiguous:
                tra_b[i] = 'T'.encode('ascii')
                ldb_arr[i] = b[i].shape[0]
            else:
                b[i] = np.asarray(b[i], order='C')
                tra_b[i] = 'N'.encode('ascii')
                ldb_arr[i] = b[i].shape[1]
        else:
            assert(k_arr[i] == b[i].shape[1])
            n_arr[i] = b[i].shape[0]
            if b[i].flags.c_contiguous:
                tra_b[i] = 'T'.encode('ascii')
                ldb_arr[i] = b[i].shape[1]
            elif b[i].flags.f_contiguous:
                tra_b[i] = 'N'.encode('ascii')
                ldb_arr[i] = b[i].shape[0]
            else:
                b[i] = np.asarray(b[i], order='C')
                tra_b[i] = 'T'.encode('ascii')
                ldb_arr[i] = b[i].shape[1]
                       
        c.append(np.empty((m_arr[i], n_arr[i]), dtype=np.float64, order='C'))
        if a[i].size == 0 or b[i].size == 0:
            if beta[i] == 0:
                c[i][:] = 0
            else:
                c[i][:] *= beta
            completed[i] = 1
    
        ptr2a[i] = a[i].ctypes.data
        ptr2b[i] = b[i].ctypes.data               
        ptr2c[i] = c[i].ctypes.data
        ldc_arr[i] = c[i].shape[1]        
    #END FOR                   
    
    func(tra_b.ctypes.data_as(ctypes.c_void_p),
         tra_a.ctypes.data_as(ctypes.c_void_p),
         n_arr.ctypes.data_as(ctypes.c_void_p),
         m_arr.ctypes.data_as(ctypes.c_void_p),
         k_arr.ctypes.data_as(ctypes.c_void_p),
         ldb_arr.ctypes.data_as(ctypes.c_void_p),
         lda_arr.ctypes.data_as(ctypes.c_void_p),
         ldc_arr.ctypes.data_as(ctypes.c_void_p),
         offfsetb.ctypes.data_as(ctypes.c_void_p),
         offfseta.ctypes.data_as(ctypes.c_void_p),                        
         offfsetc.ctypes.data_as(ctypes.c_void_p),  
         ptr2b.ctypes.data_as(ctypes.c_void_p),     
         ptr2a.ctypes.data_as(ctypes.c_void_p),
         ptr2c.ctypes.data_as(ctypes.c_void_p),
         alpha.ctypes.data_as(ctypes.c_void_p),
         beta.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(matrix_num),
         completed.ctypes.data_as(ctypes.c_void_p), )
    return c
    
pth_lNPdgemm = _imple_protocol(_mpi_dot_protocol, mpi_util.pth_lNPdgemm)
def pmmul(matrix1,matrix2,trans_a=0,trans_b=0):
    return pth_lNPdgemm(matrix1, matrix2, trans_a, trans_b)

def pmmmul(matrix1,matrix2,matrix3,trans_a=0,trans_b=0):
    return pth_lNPdgemm(pth_lNPdgemm(matrix1, matrix2, trans_a, 0), matrix3, 0, trans_b)

if __name__ == '__main__':
    mpi_init()
    mpi_final()

