import argparse
import numpy as np

from meanpress import *

# Using argparse is a good thing :)
ap = argparse.ArgumentParser(description='Compress and decompress images using meanpress')
ap.add_argument("-c",action='store_true',help='Compress mode')
ap.add_argument("-x",action='store_true',help='Decompress mode')
ap.add_argument("input",help='Input file')
ap.add_argument("output",help='Output file')
args = vars(ap.parse_args())

if not (args['c'] or args['x']):
    ap.error('No action requested, add -c or -x')

image = load_image(args['input'])
target_shape = image.shape

# Simulate compression
if True:
    imagei32 = np.array(image,dtype=np.int32)
    prediction = (imagei32[:-1,:-1]+imagei32[1:,:-1]+imagei32[:-1,1:]+1)//3
    error = np.abs(imagei32[1:,1:]-prediction)
    byteseq = compress_image(error)
else:
    byteseq = compress_image(image)

image2 = decompress_image(byteseq)
# assert(np.all(image==image2))
# Compute radio
rad = 4*byteseq.size/(target_shape[0]*target_shape[1]*target_shape[2])

print("data : %d"%(4*byteseq.size))
print("radio: %f"%rad)
