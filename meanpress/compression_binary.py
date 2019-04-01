import sys

import numpy as np

import numba
from numba import jit

from collections import deque

from .compression_general import *

arraytype = numba.typeof(np.zeros(2,dtype=np.int32))
repetitiontype = numba.typeof([np.zeros(2,dtype=np.int32)])
repetitiontype.reflected = False
# NOTE: I don't know if this ^ is the right way to do it, but I didn't find another one.
# May have problems with different versions of numba or different systems.


@jit(repetitiontype(arraytype),nopython=True)
def binary_repetition(binm):
    """
    Returns a list of (0|1,repetitions)
    """
    lens = []
    # assert(np.all(binm==0 or binm==1))
    len = 1
    last = binm[0]
    for i in range(1,binm.size):
        if binm[i]==last:
            len += 1
        else:
            lens.append(np.array((last,len),dtype=np.int32))
            last = 1-last
            len = 1
    lens.append(np.array((last,len),dtype=np.int32))
    lens = lens[::-1]
    assert([x[1]>0 for x in lens])
    return lens

@jit(numba.none(repetitiontype,numba.int32),nopython=True)
def repetition_del_first_values(reps,n):
    """
    Removes first n values on a list of repetions
    """
    while n>0:
        if len(reps)==0: break
        v,p = reps[-1]
        if p>n:
            reps[-1][1] = p-n
            n = 0
        else:
            reps.pop()
            n -= p

@jit(nopython=True)
def repetition_pop_next(reps):
    if len(reps) == 0: return 0
    v,p = reps[-1]
    assert(p>0)
    if p>1:
        reps[-1][1] = p-1
    else:
        reps.pop()
    return v

"""
Compress some of the binary values (stored as repetitions) in one word,
last_bit must be 0 or 1 depending on the last bit sequence, -1 if none.
-1 forces explicit mode, and should be used for the firsts of the sequence.
retuns grabs: how many values were added
retuns final: the final word

Binary compression has 2 modes (3 depending on how you count):
mode 0b0:  31 explicit binary digits.
mode 0b10: 6 repetitions, up to 31 times each (5 bits), starting with 0
mode 0b11: 6 repetitions, up to 31 times each (5 bits), starting with 1
"""

@jit(numba.uint32(repetitiontype),nopython=True)
def compress_binary_nexts(reps):
    MAXR = (1<<6)-1
    MASK = 0b111111
    MASKLEN = 6
    NUMBERS = 5
    rs = np.zeros(NUMBERS,dtype=np.int32)
    #
    assert(len(reps)>0)
    v,p = reps[-1]
    first_v = v
    j = 0
    i = 0
    while i<NUMBERS:
        if p>MAXR:
            rs[i] = -MAXR
            i += 1
            p -= MAXR
        else:
            rs[i] = p
            i += 1
            # Advance in reps
            j += 1
            if j<len(reps):
                _,p = reps[-j-1]
            else:
                _,p = 0,1
    # Compute binary representation
    represented = np.sum(np.abs(rs))
    bin = np.uint32(0)
    #
    assert([x[1]>0 for x in reps])
    if represented>31:
        # Perform repetition compression
        bin |= (0b10 if first_v ==0 else 0b11)<<30
        for i in range(NUMBERS):
            if rs[i]==-MAXR:
                bin |= MASK<<(30-MASKLEN*(i+1))
            else:
                bin |= (rs[i]-1)<<(30-MASKLEN*(i+1))
        repetition_del_first_values(reps,represented)
    else:
        # Don't compress
        # bin |= 0b0<<31
        for i in range(31):
            bin |= repetition_pop_next(reps)<<(31-(i+1))
    return bin

@jit(nopython=True)
def compress_binary(bins):
    """
    Compresses a matrix of binary values,
    """
    # Flatten deltas and means
    bins = bins.flatten().astype(np.int32)
    if bins.size==0:
        reps = [np.zeros(2,dtype=np.int32)][:0]
    else:
        reps = binary_repetition(bins)
    # Until there is no more to compress
    numvalues = []
    while len(reps)>0:
        numvalues.append(compress_binary_nexts(reps))
    numvalues2 = np.array(numvalues,dtype=np.uint32)
    return numvalues2

@jit(nopython=True)
def decompress_word_bin(w):
    vals = []
    if (w>>31) == 0b0:
        return np.array(divide_word(w)[1:],dtype=np.int32)
        # return divide_word(w)[1:]
    else:
        MAXR = (1<<6)-1
        MASK = 0b111111
        MASKLEN = 6
        NUMBERS = 5
        next = (w>>30) & 1
        for i in range(NUMBERS):
            wres = (w >> (30-MASKLEN*(i+1))) & MASK
            change = True
            if wres==MAXR:
                change = False
            else:
                wres += 1
            for i in range(wres):
                vals.append(next)
            if change:
                next = 1-next
        return np.array(vals,dtype=np.int32)

@jit(nopython=True)
def decompress_binary(shape,bytesec=np.array([1,2,3],dtype=np.uint32)):
    """
    Reads a matrix of binaries of the given shape,
    bytesec is the sequence of uint32 to read.
    Returns the matrix of binaries and the number of uint32 read.
    """
    w_read = 0
    # The array of bins:
    su = np.int64(shape[0])
    sv = np.int64(shape[1])
    flatsize = shape[0]*shape[1]
    bins = np.zeros(flatsize,dtype=np.int32)
    #
    w_read = 0
    nmbs = np.array([0],dtype=np.int32)[:0]
    i = 0
    #
    for n in range(flatsize):
        if i==len(nmbs):
            nmbs = decompress_word_bin(bytesec[w_read])
            w_read += 1
            i = 0
        bins[n] = nmbs[i]
        i += 1
    #
    binsx = bins.reshape((su,sv))
    return binsx,w_read


if __name__ == '__main__':
    assert(len(sys.argv)==2)
    SEED = int(sys.argv[1])
    VERBOSE = 0
    np.random.seed(SEED)
    shap = np.random.randint(40,size=2)
    flatshap = shap[0]*shap[1]
    bins = np.zeros(flatshap,dtype=np.int32)
    p = 0
    k = 0
    while k<flatshap:
        if p==0:
            b = np.random.randint(2)
            p = np.random.randint(12)
            if np.random.random(1)<0.15:
                p += np.random.randint(70)
        bins[k] = b
        p -= 1
        k += 1
    #
    bins = bins.reshape(shap)
    bytesec = compress_binary(bins)
    # Perform decompression
    bins2,red = decompress_binary(shap,bytesec)
    if VERBOSE>=1:
        print("--- bins ---")
        print(bins)
        print("--- bins2 ---")
        print(bins2)
    assert(np.all(bins==bins2))
    print("%d same."%SEED)
