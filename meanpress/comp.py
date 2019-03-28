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
    k = np.array(pixs_even<pixs_odd,dtype='int8')
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
    delta = (1-k)*pixs_even + k*pixs_odd - mean_array[:x_size]
    assert(np.all(delta>=0))
    assert(np.all(delta<128))
    assert(np.all(mean_array[:x_size]+delta<=255))
    # Return useful arrays
    if axis==0:
        return (delta,b,k),mean_array
    else:
        return (delta.T,b.T,k.T),mean_array.T

def recompose_matrix(mean,arrays,tgt_shape,axis):
    (delta,b,k) = arrays
    # Transpose everything if axis is 0
    if axis==1:
        mean = mean.T
        delta = delta.T
        b = b.T
        k = k.T
        tgt_shape = tgt_shape[::-1]
    # Recompute x_size
    x_size = tgt_shape[0]//2
    assert(mean.shape[0] in (x_size,x_size+1))
    assert(delta.shape[0]==x_size)
    # Compute maximums
    maxs = mean[:x_size]+delta
    assert(np.all(maxs<=255))
    # Compute minimums
    sum_array = 2*np.array(mean[:x_size],dtype='int16')+b
    mins = sum_array-maxs-1
    # Rebuild target
    target = np.zeros(tgt_shape,dtype='uint8')
    #
    target[:2*x_size:2]  = (1-k)*maxs + k*mins
    target[1:2*x_size:2] = k*maxs + (1-k)*mins
    if target.shape[0]>2*x_size:
        target[2*x_size]    = mean[x_size] # possible last column
    # Return reconstruction
    if axis==0:
        return target
    else:
        return target.T

def predict_shapes(shape):
    # Retrieves the sizes of the delta,b,k matrices on each resizing
    # Also indicates the axes on which each resizing should be done
    axes = []
    shapes = []
    tshapes = [shape]
    axis = int(not (shape[1]>shape[0]))
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
        tshapes.append(shape)
    # Return predictions
    return shapes,tshapes,axes

def decompose_channel(array):
    shapes,tshapes,axes = predict_shapes(array.shape)
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
        assert(tshapes[i+1]==mean.shape)
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
    return arrays,start,axes,means

def recompose_channel(arrays,start,target_shape):
    shapes,tshapes,axes = predict_shapes(target_shape[:2])
    means = [np.array([[start]])]
    assert(len(shapes)==len(arrays))
    for i in range(len(shapes)-1,-1,-1):
        assert(arrays[i][0].shape==shapes[i])
        assert(arrays[i][1].shape==shapes[i])
        assert(arrays[i][2].shape==shapes[i])
        res = recompose_matrix(means[-1],arrays[i],tshapes[i],axes[i])
        means.append(res)
    return means[-1]

def max_delta_abs(mean_array):
    max_delta = np.copy(mean_array)
    max_delta[max_delta>127] = 255-max_delta
