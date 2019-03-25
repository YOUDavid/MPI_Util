#!/usr/bin/python
import numpy as np
import ctypes
import os
import sys
from time import time, sleep, clock
from functools import reduce, partial, update_wrapper
mpi_util = ctypes.CDLL(r'./test.so')
    
raw_partial = partial
def named_partial(raw_func, *args, **kwargs):
    part_func = raw_partial(raw_func, *args, **kwargs)
    update_wrapper(part_func, raw_func)
    return part_func
partial = named_partial


def test_timer(func, r=1, i=1):
    ct_l = np.zeros(r, dtype=np.float64)
    wt_l = np.zeros(r, dtype=np.float64)    
    result = None
    ct, wt = clock(), time()
    
    for i in range(r):
        ct_l[i], wt_l[i] = clock(), time()
        if result == None:
            result = func()
        else:
            func()
        ct_l[i], wt_l[i] = clock() - ct_l[i], time() - wt_l[i]
    ct, wt = clock() - ct, time() - wt
    print "Using %s with ont: %2s, CPU time: %10.6f, wall time: %10.6f" % (func.__name__, os.environ['OMP_NUM_THREADS'], ct, wt)
    print "CPU time average: %10.6f, wall time average: %10.6f" % (np.mean(ct_l), np.mean(wt_l))
    print "CPU time stddevi: %10.6f, wall time stddevi: %10.6f" % (np.std(ct_l), np.std(wt_l)) 
    print ""   
    sleep(i)
    return result, ct_l, wt_l

    
    
def test_wrapper(func, matrix1=None, matrix2=None, a_T=0, b_T=0, rep=1, inter=1, ont=[1]):
    resultlist = list()
    for t in ont:
        mpi_util.mpi_setONT(ctypes.c_int(t))
        os.environ['OMP_NUM_THREADS'] = str(t)
        result = test_timer(partial(func, matrix1, matrix2, a_T, b_T), r=repeats)
        resultlist.append({'result': result[0], 'ctlist': result[1], 'wtlist': result[2], 'name': func.__name__, 'ont': t})
        sleep(inter)
    return resultlist
    

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
                ("actualFunction", ctypes.c_int)]

def mpi_init():
    mpi_util.mpi_init()
    if int(os.environ['PMI_RANK']) != 0:
        exit(0)
        
def mpi_final():
    mpi_util.mpi_final()


def lddot(a, b, a_T=0, b_T=0):
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
        #ptr2a[i]['currentPointer'] = a[i].ctypes.data
        #ptr2a[i]['currentLength'] = a[i].size
        #ptr2a[i]['currentType'] = 2
        #ptr2b[i]['currentPointer'] = b[i].ctypes.data               
        #ptr2b[i]['currentLength'] = b[i].size
        #ptr2b[i]['currentType'] = 2        
        #ptr2c[i]['currentPointer'] = c[i].ctypes.data
        #ptr2c[i]['currentLength'] = c[i].size
        #ptr2c[i]['currentType'] =  2  
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
#define TYPE_CHAR 0
#define TYPE_INT 1
#define TYPE_DOUBLE 2
#define TYPE_POINTER 3
#define TYPE_NP_MATRIX 4                        
               
               

    
    mpi_util.mpi_template_local(ctypes.byref(datapack))
    return c

def set_type(typeList, index, currentType, currentLength, currentPointer):    
    typeList[index]['currentType'] = currentType
    typeList[index]['currentLength'] = currentLength
    typeList[index]['currentPointer'] = currentPointer
    
def map_dot_protocol(a, b, a_T=0, b_T=0, func=np.dot):
    if a_T == 0:
        if b_T == 0:
            return list(map(lambda x: func(x[0], x[1]), zip(a, b)))
        else:
            return list(map(lambda x: func(x[0], x[1].T), zip(a, b)))
    else:
        if b_T == 0:
            return list(map(lambda x: func(x[0].T, x[1]), zip(a, b)))
        else:
            return list(map(lambda x: func(x[0].T, x[1].T), zip(a, b)))    
    
def imple_protocol(protocol, real_func):
    return_func = partial(protocol, func=real_func)
    return_func.__name__ = real_func.__name__
    return return_func

def test_checker(resultlist, threshold=1e-3):
    if len(resultlist) < 2:
        print "Only one result, skip checking!"
    else:
        base = resultlist[0]
        print "Start checking!"
        print "Using function", base['name'], "with OMP_NUM_THREADS =", base['ont'], "as base"
        for i in resultlist:
            if not i is base:
                print "Checking", i['name'], "against base:"
                for index in range(len(base['result'])):
                    
                    if not (np.abs(i['result'][index] - base['result'][index]) < threshold).all():
                        print "Not equal at index", index#, "Summed abs difference:"
                        print base['result'][index]
                        print '@@@@@@@@@@@@@@@@@@@@@@@'
                        print i['result'][index]
                        #print np.abs(i['result'][index] - base['result'][index]).sum()
                        print "------------------------------------"
                print "Finish checking", i['name']
        print "End of checking!" 

if __name__ == '__main__':
    np.random.seed(0)
    mpi_init()
    #np.set_printoptions(threshold=np.nan)
    #if os.environ['length'] != '':
    #    l = int(os.environ['length'])
    #else:
    l = 1
        
    #if os.environ['size'] != '':
    #    low = int(os.environ['size'])
    #else:
    low = 100
    
    
    
    m = 0
    k = 0
    n = 0
    #p = 0
    hig = low + 1
    interval = 0
    repeats = 1
    a = []
    b = []
    c = []
    T_cddot = np.zeros(repeats, dtype=np.float64)
    T_lddot = np.zeros(repeats, dtype=np.float64)
    m = np.random.randint(low, high=hig)
    k = np.random.randint(low, high=hig)        
    n = np.random.randint(low, high=hig) 
    print "Start to generate ramdom square matrix list"
    T_gen = time()
    for i in range(l):
     
        temp = np.random.rand(m,m) * 2        
        a.append(temp)
        b.append(temp)

    print "Finish generating the random matrix of length:", l, "and size:",m, k, n#, p
    print "Time used:", time() - T_gen
    print "Running test cases for", repeats, "cycles"
    wrapper = partial(test_wrapper, matrix1=a, matrix2=b, a_T=1, rep=repeats, inter=interval)
    funclist = [imple_protocol(map_dot_protocol, np.dot),
                lddot
               ]
                
    resultlist = list(map(wrapper, funclist))
    result = []

    for i in resultlist:
        result +=i
        
    test_checker(result)
    
    
    
    print 'time'
    for i in xrange(len(result)):
        sys.stdout.write(str(np.mean(result[i]['wtlist'])) + ' ')
        if (i + 1) % 5 == 0:
            print ''
    print ''
    print 'stdev'
    for i in xrange(len(result)):
        sys.stdout.write(str(np.std(result[i]['wtlist'])) + ' ')
        if (i + 1) % 5 == 0:
            print ''
    
    mpi_final()

