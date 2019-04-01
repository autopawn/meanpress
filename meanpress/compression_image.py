import numpy as np

from .compression_channel import compress_channel,decompress_channel

def compress_image(image,verbose=True):
    assert(len(image.shape)==3)
    bytes = [np.array(image.shape,dtype=np.uint32)]
    for c in range(image.shape[2]):
        channel = image[...,c]
        byteseq = compress_channel(channel)
        if verbose:
            rate = 4.0*byteseq.size/(image.shape[0]*image.shape[1])
            print("Channel %d rate: %9f"%(c,rate))
        bytes.append(byteseq)
    return np.concatenate(bytes)

def decompress_image(byteseq):
    x,y,z = byteseq[:3]
    x = int(x)
    y = int(y)
    w_read = 3
    image = []
    for c in range(z):
        channel,read = decompress_channel(byteseq[w_read:],(x,y))
        w_read += read
        image.append(channel)
    image = np.array(image,dtype=np.uint8)
    image = np.moveaxis(image,0,-1)
    return image
