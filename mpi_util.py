#!/usr/bin/python
import numpy as np
import ctypes
import os
import sys
from time import time, sleep, clock
from functools import reduce, partial, update_wrapper #wraps
from pyscf.lib.numpy_helper import ddot
from multiprocessing import Pool

# 1: map numpy dot,  omp=cpu_per_task, 1
# 2: map pyscf ddot, omp=cpu_per_task, 1
# 3: pmmul use ddot, omp=cpu_per_task, 1
# 4: lOMPdgemm,      omp=cpu_per_task, 1
# 5: ldgemm,         omp=cpu_per_task, 1
# 6: lNPdgemm,       omp=cpu_per_task, 1
# 7: map cdgemm,     omp=cpu_per_task, 1
# 8: map cddot,      omp=cpu_per_task, 1

class NP_MATRIX(ctypes.Structure):
    _fields_ = [('m'      , ctypes.c_int   ),
                ('n'      , ctypes.c_int   ),
                ('mstride', ctypes.c_int   ),
                ('nstride', ctypes.c_int   ),
                ('matrix' , ctypes.c_void_p)]
                
                

mpi_util = ctypes.CDLL(r'/home/zhyou/mpi/mpi_util.so')
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

def test_wrapper(func, matrix1=None, matrix2=None, rep=1, inter=1, ont=[1,2,4,8,16]):
    resultlist = list()
    for t in ont:
        mpi_util.mpi_setONT(ctypes.c_int(t))
        os.environ['OMP_NUM_THREADS'] = str(t)
        result = test_timer(partial(func, matrix1, matrix2), r=repeats)
        resultlist.append({'result': result[0], 'ctlist': result[1], 'wtlist': result[2], 'name': func.__name__, 'ont': t})
        sleep(inter)
    return resultlist


def mpi_init():
    mpi_util.mpi_init()
    if int(os.environ['PMI_RANK']) != 0:
        exit(0)
        
def mpi_final():
    mpi_util.mpi_final()

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
        a = np.asarray(a, order='C')
        trans_a = 'N'
        #raise ValueError('a.flags: %s' % str(a.flags))

    assert(k == b.shape[0])
    if b.flags.c_contiguous:
        trans_b = 'N'
    elif b.flags.f_contiguous:
        trans_b = 'T'
        b = b.T
    else:
        b = np.asarray(b, order='C')
        trans_b = 'N'
        #raise ValueError('b.flags: %s' % str(b.flags))

    if c is None:
        c = np.empty((m,n))
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

    mpi_util.NPdgemm(ctypes.c_char(trans_b.encode('ascii')),
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


def cdgemm(a, b, alpha=1, c=None, beta=0, ):
    
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
        a = np.asarray(a, order='C')
        trans_a = 'N'
        #raise ValueError('a.flags: %s' % str(a.flags))

    assert(k == b.shape[0])
    if b.flags.c_contiguous:
        trans_b = 'N'
    elif b.flags.f_contiguous:
        trans_b = 'T'
        b = b.T
    else:
        b = np.asarray(b, order='C')
        trans_b = 'N'
        #raise ValueError('b.flags: %s' % str(b.flags))

    if c is None:
        c = np.empty((m,n))
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

    mpi_util.dgemm(ctypes.c_char(trans_b.encode('ascii')),
                       ctypes.c_char(trans_a.encode('ascii')),
                       ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
                       ctypes.c_int(b.shape[1]), ctypes.c_int(a.shape[1]),
                       ctypes.c_int(c.shape[1]),
                       b.ctypes.data_as(ctypes.c_void_p),
                       a.ctypes.data_as(ctypes.c_void_p),
                       c.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_double(alpha), ctypes.c_double(beta))
    return c



def imple_protocol(protocol, real_func):
    return_func = partial(protocol, func=real_func)
    return_func.__name__ = real_func.__name__
    return return_func

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

def mpi_dot_protocol(a, b, a_T=0, b_T=0, func=mpi_util.mpi_lNPdgemm):
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
                #a[i] = a[i].T
                #a_transposed.append(i)
            else:
                a[i] = np.asarray(a[i], order='C')
                tra_a[i] = 'N'.encode('ascii')
                lda_arr[i] = a[i].shape[1]
                #raise ValueError('a.flags: %s' % str(a.flags))  
        else:
            m_arr[i] = a[i].shape[1]
            k_arr[i] = a[i].shape[0]
            if a[i].flags.c_contiguous:
                tra_a[i] = 'T'.encode('ascii')
                lda_arr[i] = a[i].shape[1]
            elif a[i].flags.f_contiguous:
                tra_a[i] = 'N'.encode('ascii')
                lda_arr[i] = a[i].shape[0]
                #a[i] = a[i].T
                #a_transposed.append(i)
            else:
                a[i] = np.asarray(a[i], order='C')
                tra_a[i] = 'T'.encode('ascii')
                lda_arr[i] = a[i].shape[1]
                #raise ValueError('a.flags: %s' % str(a.flags)) 

        if b_T == 0:
            assert(k_arr[i] == b[i].shape[0])
            n_arr[i] = b[i].shape[1]
            if b[i].flags.c_contiguous:
                tra_b[i] = 'N'.encode('ascii')
                ldb_arr[i] = b[i].shape[1]
            elif b[i].flags.f_contiguous:
                tra_b[i] = 'T'.encode('ascii')
                ldb_arr[i] = b[i].shape[0]
                #b[i] = b[i].T
                #b_transposed.append(i)
            else:
                b[i] = np.asarray(b[i], order='C')
                tra_b[i] = 'N'.encode('ascii')
                ldb_arr[i] = b[i].shape[1]
            #raise ValueError('b.flags: %s' % str(b.flags))
        else:
            assert(k_arr[i] == b[i].shape[1])
            n_arr[i] = b[i].shape[0]
            if b[i].flags.c_contiguous:
                tra_b[i] = 'T'.encode('ascii')
                ldb_arr[i] = b[i].shape[1]
            elif b[i].flags.f_contiguous:
                tra_b[i] = 'N'.encode('ascii')
                ldb_arr[i] = b[i].shape[0]
                #b[i] = b[i].T
                #b_transposed.append(i)
            else:
                b[i] = np.asarray(b[i], order='C')
                tra_b[i] = 'T'.encode('ascii')
                ldb_arr[i] = b[i].shape[1]
            #raise ValueError('b.flags: %s' % str(b.flags))
            
                       
        c.append(np.empty((m_arr[i], n_arr[i]), dtype=np.float64, order='C'))
        if a[i].size == 0 or b[i].size == 0:
            if beta[i] == 0:
                c[i][:] = 0
            else:
                c[i][:] *= beta
            completed[i] = 1
            #print "ERROR!!"
        
        #assert(a[i].flags.c_contiguous)
        #assert(b[i].flags.c_contiguous)
    
        ptr2a[i] = a[i].ctypes.data
        ptr2b[i] = b[i].ctypes.data               
        ptr2c[i] = c[i].ctypes.data
        #lda_arr[i] = a[i].shape[1]
        #ldb_arr[i] = b[i].shape[1]
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
         completed.ctypes.data_as(ctypes.c_void_p))
    #for i in a_transposed:
    #    a[i] = a[i].T
    #for i in b_transposed:
    #    b[i] = b[i].T
    return c
    
    

def mpi_print(l, a):
    mpi_util.mpi_print(ctypes.c_int(l), a.ctypes.data_as(ctypes.c_void_p))

def omp_print(l, a):
    mpi_util.omp_print(ctypes.c_int(l), a.ctypes.data_as(ctypes.c_void_p))

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
    mpi_init()
#    l = 0;
#    a = np.empty((2,2), dtype=np.float64, order='C')
#    mpi_print(l, a)
#    mpi_final()


    np.random.seed(0)
    mpi_init()
    #np.set_printoptions(threshold=np.nan)
    if os.environ['length'] != '':
        l = int(os.environ['length'])
    else:
        l = 100
        
    if os.environ['size'] != '':
        low = int(os.environ['size'])
    else:
        low = 100    
    
    low *= 2
    
    m = 0
    k = 0
    n = 0
    #p = 0
    hig = low + 1
    interval = 0
    repeats = 5
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

        #p = np.random.randint(low, high=hig)        
        a.append(np.random.rand(m,k) * 2)
        b.append(np.random.rand(k,n) * 2)
        a[i] = a[i][:low/2, :low/2]
        b[i] = b[i][:low/2, :low/2]
        
#        c.append(np.random.rand(n,p) * 20)
        #b[i] = b[i].T
        #a[i] = a[i].T
#        a.append(np.array(np.array(np.random.rand(m, k), dtype = np.int32), dtype = np.float64))
#        b.append(np.array(np.array(np.random.rand(k, n), dtype = np.int32), dtype = np.float64))
#        a.append(np.array(np.arange(m * k), dtype=np.float64).reshape(m, k) + 2 * i)
#        b.append(np.array(np.arange(n * k), dtype=np.float64).reshape(k, n) + 2 * i + 1) 
#    print a[50]
#    print "============"
#    print b[50]
    print "Finish generating the random matrix of length:", l, "and size:",m, k, n#, p
    print "Time used:", time() - T_gen
    print "Running test cases for", repeats, "cycles"
    wrapper = partial(test_wrapper, matrix1=a, matrix2=b, rep=repeats, inter=interval)
    funclist = [#imple_protocol(map_dot_protocol, np.dot),
                imple_protocol(map_dot_protocol, ddot),
                #pmmul,
                #imple_protocol(mpi_dot_protocol, mpi_util.mpi_ldgemm),
                #imple_protocol(mpi_dot_protocol, mpi_util.mpi_lOMPdgemm),
                imple_protocol(mpi_dot_protocol, mpi_util.mpi_lNPdgemm),
                #imple_protocol(map_dot_protocol, cdgemm),
                #imple_protocol(map_dot_protocol, cddot),
                ]
                
    resultlist = list(map(wrapper, funclist))
    result = []
    #print len(resultlist[0])
    for i in resultlist:
        result +=i
    #print len(result)
    
    #test_checker(result)
# 1: map numpy dot,  omp=cpu_per_task, 1
# 2: map pyscf ddot, omp=cpu_per_task, 1
# 3: pmmul use ddot, omp=cpu_per_task, 1
# 4: lOMPdgemm,      omp=cpu_per_task, 1
# 5: ldgemm,         omp=cpu_per_task, 1
# 6: lNPdgemm,       omp=cpu_per_task, 1
# 7: map cdgemm,     omp=cpu_per_task, 1
# 8: map cddot,      omp=cpu_per_task, 1
    print 'time'
    for i in xrange(len(result)):
        sys.stdout.write(str(np.mean(result[i]['wtlist'])) + ' ')
        if (i + 1) % 5 == 0:
            print ''

    print 'stdev'
    for i in xrange(len(result)):
        sys.stdout.write(str(np.std(result[i]['wtlist'])) + ' ')
        if (i + 1) % 5 == 0:
            print ''
#        
    mpi_final()

