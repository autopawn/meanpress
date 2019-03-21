import numpy as np

def decompose_matrix(array):
    assert(len(array.shape)==2)
    assert(array.shape[0]>1)
    #
    x_size = array.shape[0]//2
    y_size = array.shape[1]
    x_odd = array.shape[0]%2
    # Pixel parts
    pixs_even = array[0:2*x_size:2]
    pixs_odd  = array[1::2]
    pixs_off  = array[2*x_size:]
    # Get the mean array:
    mean_array = np.zeros((x_size+x_odd,y_size),dtype='int16')
    # Add halved odds rounded
    mean_array[:x_size] += pixs_odd
    mean_array[:x_size] += 1
    mean_array[:x_size] //= 2
    # Add halved evens
    mean_array[:x_size] += pixs_even//2
    bit_array = pixs_even[:x_size]&1;
    # Add offpixels (only if size_x is odd)
    mean_array[x_size:] += pixs_off
    # Compute deltas array:
    delta_array = (mean_array[:x_size]-pixs_odd)*-1
    # Return useful arrays
    return mean_array,delta_array,bit_array

def decompose_channel(array):
    # Final arrays
    means = []
    deltas = []
    bits = []
    axes = []
    #
    front = array
    axis = 0
    while front.shape!=(1,1):
        # Perform on the opposite axis
        if front.shape[1-axis]!=1:
            axis = 1-axis
        # Code channel, invert if axis==1
        if axis==1: front = front.T
        mean_a,delta_a,bit_a = decompose_matrix(front);
        if axis==1: front = front.T
        # Add to final arrays
        means.append(mean_a)
        deltas.append(delta_a)
        bits.append(bit_a)
        axes.append(axis)
        # Update front to new mean
        front = mean_a
    # Return all components (means are extra, as they are not needed).
    start = front[0,0]
    return (deltas,bits,start),axes,means




def max_delta_abs(mean_array):
    max_delta = np.copy(mean_array)
    max_delta[max_delta>127] = 255-max_delta
