import numpy as np

def decompose_matrix(array,axis):
    assert(len(array.shape)==2)
    assert(axis in (0,1))
    assert(array.shape[axis]>1)
    # Invert if axis is 1
    if axis==1:
        array = array.T
    # Sizes
    x_size = array.shape[0]//2
    y_size = array.shape[1]
    x_odd = array.shape[0]%2
    # Pixel parts
    pixs_even = array[0:2*x_size:2]
    pixs_odd  = array[1::2]
    pixs_off  = array[2*x_size:]
    # Get dominant pixel
    k = np.array(pixs_even>pixs_odd,dtype='int8')
    # Get the rounded mean array:
    mean_array = np.zeros((x_size+x_odd,y_size),dtype='int16')
    mean_array[:x_size] += 1
    mean_array[:x_size] += pixs_odd
    mean_array[:x_size] += pixs_even
    b = mean_array[:x_size]%2
    mean_array[:x_size] //= 2
    # Add offpixels (only if size_x is odd)
    mean_array[x_size:] += pixs_off
    # Get delta of max
    delta = (1-k)*pixs_odd + k*pixs_even - mean_array[:x_size]
    #
    # Return useful arrays
    if axis==0:
        return (delta,b,k),mean_array
    else:
        return (delta.T,b.T,k.T),mean_array.T

def predict_shapes(shape):
    # Retrieves the sizes of the delta,b,k matrices on each resizing
    # Also indicates the axes on which each resizing should be done
    axes = []
    shapes = []
    axis = 1
    while shape!=(1,1):
        # Swap axis:
        if shape[1-axis]>1:
            axis = 1-axis
        # Save matrix sizes and axes
        if axis==0:
            shapes.append((shape[0]//2,shape[1]))
        else:
            shapes.append((shape[0],shape[1]//2))
        axes.append(axis)
        # Decrease on axis:
        if axis==0:
            shape = ((shape[0]+1)//2,shape[1])
        else:
            shape = (shape[0],(shape[1]+1)//2)
    # Return predictions
    return shapes,axes

def decompose_channel(array):
    shapes,axes = predict_shapes(array.shape)
    arrays = []
    means = []
    #
    front = array
    for i in range(len(shapes)):
        pred_shape = shapes[i]
        axis = axes[i]
        # Code channel, invert if axis==1
        (delta,b,k),mean = decompose_matrix(front,axis);
        assert(pred_shape==delta.shape)
        assert(pred_shape==b.shape)
        assert(pred_shape==k.shape)
        # Add to final arrays
        arrays.append((delta,b,k))
        means.append(mean)
        axes.append(axis)
        # Update front to new mean
        front = mean
    # Return all components
    # - means are extra, as they are not needed.
    # - axes are extra, as they can be calculated.
    start = front[0,0]
    return arrays,axes,means

def max_delta_abs(mean_array):
    max_delta = np.copy(mean_array)
    max_delta[max_delta>127] = 255-max_delta
