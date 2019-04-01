import numpy as np

from numba import jit

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
    """
    From an array of binary values, computes a number.
    """
    number = np.int32(0)
    for b in bits:
        number <<= 1
        number |= b
    return number

@jit(nopython=True)
def max_delta_abs(mean_array):
    max_delta = np.copy(mean_array)
    max_delta[max_delta>127] = 255-max_delta
