import sys

import numpy as np

from numba import jit

@jit(nopython=True)
def bit_cost(xs):
    """
    Max bits required to represent up to the given numbers
    """
    bits = np.zeros(xs.shape,dtype=np.int32)
    bits[xs>=1] = 1
    bits[xs>=2] = 2
    bits[xs>=4] = 3
    bits[xs>=8] = 4
    bits[xs>=16] = 5
    bits[xs>=32] = 6
    bits[xs>=64] = 7
    return bits

@jit(nopython=True)
def s_bit_cost(v):
    """
    Max bits required to represent up to the given number
    """
    if v>=64: return  7
    if v>=32: return  6
    if v>=16: return  5
    if v>=8: return 4
    if v>=4: return 3
    if v>=2: return 2
    if v>=1: return 1
    return 0

@jit(nopython=True)
def maxes_from_means(mean):
    """
    From the mean values, the maximum value that the delta may have for each cell
    """
    shape = mean.shape
    mean = mean.flatten()
    maxv = np.zeros(mean.shape,dtype=np.int32)
    maxv[mean<=127] = mean[mean<=127]
    maxv[mean>=128] = 255-mean[mean>=128]
    maxv = maxv.reshape(shape)
    return maxv

@jit(nopython=True)
def compress_nexts(delta,maxv):
    """
    Compress some of the values in delta in one word,
    retuns grabs: how many values were added
    retuns final: the final word
    retuns outl_arrays: arrays of the outlier bit values
    """
    GRABS_PER_MODE = (0,28,14,10,7,6,5,4)
    PREFIX_HEADER = (0,0b1100,0b1101,0b0000,0b1110,0b0100,0b1000,0b1111) # [3],[5], and [6], just use the first 2 bits
    FULLMASK = (0,0b1,0b11,0b111,0b1111,0b11111,0b111111,0b1111111)
    max_density = 0.0
    final_mode = 0
    # Identify mode that results in more density
    for mode in range(1,8):
        grabs = min(delta.shape[0],GRABS_PER_MODE[mode])
        outlier_maxv = maxv[:grabs]-((1<<mode)-1)
        outliers = (delta[:grabs]>=((1<<mode)-1))
        outlier_bitcost = outliers*bit_cost(outlier_maxv)
        density = grabs/(32.0+np.sum(outlier_bitcost))
        if density > max_density:
            max_density = density
            final_mode = mode
    # binary encoding and outlier string
    mode = final_mode
    final = np.uint32(0)
    final |= (PREFIX_HEADER[mode]<<28)
    #
    outl_arrays = [] # outlayer binary representation
    grabs = min(delta.shape[0],GRABS_PER_MODE[mode])
    outlier_maxv = maxv[:grabs]-((1<<mode)-1)
    outliers = (delta[:grabs]>=((1<<mode)-1))
    outlier_bitcost = outliers*bit_cost(outlier_maxv)
    for i in range(grabs):
        despl = GRABS_PER_MODE[mode]-1-i
        if outliers[i]:
            # Put value that represets outlier
            final |= FULLMASK[mode]<<(mode*despl)
            # Get binary representation of outlier
            extra =  delta[i]-((1<<mode)-1)
            out_bits = outlier_bitcost[i]
            bitsp = np.zeros(out_bits,dtype=np.uint8)
            for i in range(out_bits):
                bitsp[out_bits-1-i] = (extra & 1)
                extra >>= 1
            assert(extra==0)
            outl_arrays.append(bitsp)
        else:
            final |= delta[i]<<(mode*despl)
    return grabs,final,outl_arrays

@jit(nopython=True)
def decompress_word(word):
    """
    Returns the delta values stored on a word.
    Outliers are retrieved as -((1<<mode)-1)
    """
    GRABS_PER_MODE = (0,28,14,10,7,6,5,4)
    FULLMASK = (0,0b1,0b11,0b111,0b1111,0b11111,0b111111,0b1111111)
    if (word>>30)==0b00:
        mode = 3
    elif (word>>30)==0b01:
        mode = 5
    elif (word>>30)==0b10:
        mode = 6
    elif (word>>30)==0b11:
        if (word>>28)==0b1100:
            mode = 1
        elif (word>>28)==0b1101:
            mode = 2
        elif (word>>28)==0b1110:
            mode = 4
        elif (word>>28)==0b1111:
            mode = 7
    # Get each number from the word
    grabs = GRABS_PER_MODE[mode]
    numbs = np.zeros(grabs,dtype=np.int32)
    for i in range(grabs):
        ii = grabs-1-i
        numbs[i] = (word & (FULLMASK[mode]<<(ii*mode)) )>>(ii*mode)
        # FULLMASK[mode]<<(32-mode*(grabs-1-i))
        # s = 32-(grabs-i)*mode
        # numbs[i] = (word<<s)>>(32-mode)
    assert(np.all(numbs<=((1<<mode)-1)))
    numbs[numbs==((1<<mode)-1)] = -((1<<mode)-1) # t.b.d. outliers.
    return numbs

@jit(nopython=True)
def compress_deltas(delta,means):
    """
    Compresses a matrix of deltas,
    the matrix of means is used to know the maximum values that the deltas can
    aquire.
    """
    # Flatten deltas and means
    means = means[:delta.shape[0],:delta.shape[1]]
    delta = delta.flatten().astype(np.int32)
    means = means.flatten().astype(np.int32)
    # Compute maxes from means
    maxv = maxes_from_means(means)
    # Discard cells that require 0 bits
    delta = delta[maxv>0]
    means = means[maxv>0]
    maxv = maxv[maxv>0]
    # Until there is no more to compress
    numvalues = []
    outlbinseq = []
    n = 0
    while n<means.shape[0]:
        # Compress
        g,i32,outls = compress_nexts(delta[n:],maxv[n:])
        n += g
        numvalues.append(i32)
        for ou in outls:
            for v in ou:
                outlbinseq.append(v)
    numvalues = np.array(numvalues,dtype=np.uint32)
    # Get uint32s for outlbindseqs
    outlbinseq = np.array(outlbinseq,dtype=np.uint32)
    outl_ints = np.zeros((outlbinseq.shape[0]+31)//32,dtype=np.uint32)
    for i in range(outlbinseq.shape[0]):
        outl_ints[i//32] |= outlbinseq[i]<<(31-(i%32))
    # Done compressing
    return numvalues,outl_ints

@jit(nopython=True)
def divide_word(word):
    """
    Divides a word, creating an array of 32 bits
    """
    res = []
    for i in range(32):
        b = (word & (1<<(31-i))) >> (31-i)
        assert(b==0 or b==1)
        res.append(b)
    return res

@jit(nopython=True)
def recompose_number(bits):
    number = np.int32(0)
    for b in bits:
        number <<= 1
        number |= b
    return number

@jit(nopython=True)
def decompress_deltas(shape,means,bytesec):
    """
    Reads a matrix of deltas of the given shape considering the
    previous matrix of means, bytesec is the sequence of uint32 to read.
    Returns the matrix of deltas and the number of uint32 read.
    """
    w_read = 0
    means = means[:shape[0],:shape[1]]
    means = means.flatten().astype(np.int32)
    # Compute maxes from means
    maxv = maxes_from_means(means)
    # The array of deltas:
    deltas = np.zeros(means.shape,dtype=np.int32)
    n = 0
    nmbs = np.array([np.int32(0)])[:0]
    i = 0
    for n in range(means.size):
        if maxv[n]==0: continue
        if i==nmbs.size:
            nmbs = decompress_word(bytesec[w_read])
            w_read += 1
            i = 0
        deltas[n] = nmbs[i]
        i += 1
    # Indentify outliers
    bit_buffer = [np.int32(0)][:0]
    for i in range(means.size):
        # If outlier
        if deltas[i]<0:
            max_outside = maxv[i]+deltas[i]
            bits = s_bit_cost(max_outside)
            # Read more outlier bits if needed
            if len(bit_buffer)<bits:
                bit_buffer += divide_word(bytesec[w_read])
                w_read += 1
            # Rediscover outliers
            if bits==0:
                num = 0
            else:
                num = recompose_number(bit_buffer[:bits])
                bit_buffer = bit_buffer[bits:]
            deltas[i] = num-deltas[i]
        assert(deltas[i]<=maxv[i])
    # Retrieve deltas
    assert(np.all(deltas<128))
    assert(np.all(deltas>=0))
    deltasx = deltas.reshape(shape)
    return deltasx,w_read

if __name__ == '__main__':
    VERBOSE = 0
    #
    assert(len(sys.argv)==2)
    SEED = int(sys.argv[1])
    np.random.seed(SEED)
    shap = np.random.randint(40,size=2)
    ms = (np.random.random(shap)*256).astype(np.int32)
    ds = maxes_from_means(ms)
    ds = (ds.astype(np.float64)*np.random.random(ds.shape)).astype(np.int32)
    if VERBOSE>=1:
        print("-- ds:")
        print(ds.shape)
        print(ds)
    nums,outls = compress_deltas(ds,ms)
    bytesec = np.concatenate((nums,outls))
    assert(bytesec.dtype==np.uint32)
    if VERBOSE>=2:
        print([bin(x)[2:].zfill(32) for x in bytesec])
    ds2,w_read = decompress_deltas(ds.shape,ms,bytesec)
    # Result
    if VERBOSE>=1:
        print("-- ds2:")
        print(ds2.shape)
        print(ds2)
    maxs = maxes_from_means(ms)
    # Moment of truth
    for i in range(ds.shape[0]):
        for j in range(ds.shape[1]):
            if ds[i,j] != ds2[i,j]:
                print("-- error: --")
                print("%d,%d"%(i,j))
                print("ds[i,j]")
                print(ds[i,j])
                print("ds2[i,j]")
                print(ds2[i,j])
                print("ms[i,j]")
                print(ms[i,j])
                print("maxs[i,j]")
                print(maxs[i,j])
    assert(np.all(ds==ds2))
    print("%d same."%SEED)
