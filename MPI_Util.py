#!/usr/bin/python
import numpy as np
import ctypes
import os
import sys
from functools import reduce, partial, update_wrapper #wraps
mpi_util = ctypes.CDLL(r'./mpi_util.so')
    

class Type(ctypes.Structure):
    _fields_ = [("currentType", ctypes.c_int),
                ("currentLength", ctypes.c_int),
                ("currentPointer", ctypes.c_void_p)]

class Datapack(ctypes.Structure):
    _fields_ = [("allEntryNumber", ctypes.c_int),
                ("typeList", ctypes.POINTER(Type)),
                ("sharedEntryNumber", ctypes.c_int),
                ("sharedEntryIndices", ctypes.POINTER(ctypes.c_int)),
                ("dependentEntryNumber", ctypes.c_int),
                ("dependentEntryIndices", ctypes.POINTER(ctypes.c_int)),
                ("resultEntryNumber", ctypes.c_int),
                ("resultEntryIndices", ctypes.POINTER(ctypes.c_int)),      
                ("totalTaskNumber", ctypes.c_int),
                ("isCompleted", ctypes.POINTER(ctypes.c_char)),                      
                ("actualFunction", ctypes.c_ulong)]



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

def mpi_lNPdgemm(a, b, a_T=0, b_T=0):
    assert(len(a) == len(b))
    matrix_num = len(a)

    tra_a = np.empty(matrix_num, dtype=ctypes.c_char)
    tra_b = np.empty(matrix_num, dtype=ctypes.c_char)
    m_arr = np.empty(matrix_num, dtype=ctypes.c_int) 
    n_arr = np.empty(matrix_num, dtype=ctypes.c_int)
    k_arr = np.empty(matrix_num, dtype=ctypes.c_int)  
    ptr2a = np.empty(matrix_num, dtype=Type)
    ptr2b = np.empty(matrix_num, dtype=Type)
    ptr2c = np.empty(matrix_num, dtype=Type) 
    
    completed = np.empty(matrix_num, dtype=ctypes.c_char)
    c = []

    
    for i in range(matrix_num):
        if a_T == 0:
            m_arr[i] = a[i].shape[0]
            k_arr[i] = a[i].shape[1]
            if a[i].flags.c_contiguous:
                tra_a[i] = 'N'.encode('ascii')
            elif a[i].flags.f_contiguous:
                tra_a[i] = 'T'.encode('ascii')
            else:
                a[i] = np.asarray(a[i], order='C')
                tra_a[i] = 'N'.encode('ascii')
        else:
            m_arr[i] = a[i].shape[1]
            k_arr[i] = a[i].shape[0]
            if a[i].flags.c_contiguous:
                tra_a[i] = 'T'.encode('ascii')
            elif a[i].flags.f_contiguous:
                tra_a[i] = 'N'.encode('ascii')
            else:
                a[i] = np.asarray(a[i], order='C')
                tra_a[i] = 'T'.encode('ascii')

        if b_T == 0:
            assert(k_arr[i] == b[i].shape[0])
            n_arr[i] = b[i].shape[1]
            if b[i].flags.c_contiguous:
                tra_b[i] = 'N'.encode('ascii')
            elif b[i].flags.f_contiguous:
                tra_b[i] = 'T'.encode('ascii')
            else:
                b[i] = np.asarray(b[i], order='C')
                tra_b[i] = 'N'.encode('ascii')
        else:
            assert(k_arr[i] == b[i].shape[1])
            n_arr[i] = b[i].shape[0]
            if b[i].flags.c_contiguous:
                tra_b[i] = 'T'.encode('ascii')
            elif b[i].flags.f_contiguous:
                tra_b[i] = 'N'.encode('ascii')
            else:
                b[i] = np.asarray(b[i], order='C')
                tra_b[i] = 'T'.encode('ascii')
                       
        c.append(np.empty((m_arr[i], n_arr[i]), dtype=np.float64, order='C'))
        if a[i].size == 0 or b[i].size == 0:
            completed[i] = 'Y'
        else:
            completed[i] = 'N'
        
        
        ptr2a[i] = 2, a[i].size, a[i].ctypes.data
        ptr2b[i] = 2, b[i].size, b[i].ctypes.data
        ptr2c[i] = 2, c[i].size, c[i].ctypes.data
    #END FOR         

    
    allEntryNumber = 8
    sharedEntryNumber = 0
    sharedEntryIndices= np.array([-1], dtype=ctypes.c_int)
    dependentEntryNumber = 7
    dependentEntryIndices = np.array([0, 1, 2, 3, 4, 5, 6], dtype=ctypes.c_int)
    resultEntryNumber = 1
    resultEntryIndices = np.array([7], dtype=ctypes.c_int)
    totalTaskNumber = matrix_num
    isCompleted = completed.ctypes.data_as(ctypes.POINTER(ctypes.c_char))
    actualFunction = 2
    typeList = np.empty(8, dtype=Type) 
    
    typeList[0] = 0, matrix_num, tra_a.ctypes.data
    typeList[1] = 0, matrix_num, tra_b.ctypes.data
    typeList[2] = 1, matrix_num, m_arr.ctypes.data
    typeList[3] = 1, matrix_num, n_arr.ctypes.data
    typeList[4] = 1, matrix_num, k_arr.ctypes.data
    typeList[5] = 3, matrix_num, ptr2a.ctypes.data
    typeList[6] = 3, matrix_num, ptr2b.ctypes.data
    typeList[7] = 3, matrix_num, ptr2c.ctypes.data

    
    datapack = Datapack(allEntryNumber, 
                        typeList.ctypes.data_as(ctypes.POINTER(Type)),
                        sharedEntryNumber,
                        sharedEntryIndices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        dependentEntryNumber,
                        dependentEntryIndices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        resultEntryNumber,
                        resultEntryIndices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        totalTaskNumber,
                        isCompleted, 
                        actualFunction)
    
    mpi_util.mpi_template_local(ctypes.byref(datapack))
    return c

if __name__ == '__main__':
    mpi_init()
    mpi_final()

