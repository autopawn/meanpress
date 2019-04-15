import argparse
import numpy as np

from meanpress import *

# Using argparse is a good thing :)
ap = argparse.ArgumentParser(description='Compress and decompress images using meanpress')
ap.add_argument("-c",action='store_true',help='Compress mode')
ap.add_argument("-x",action='store_true',help='Decompress mode')
ap.add_argument("-d",action='store_true',help='Save matrix displays instead')
ap.add_argument("input",help='Input file')
ap.add_argument("output",help='Output file')
args = vars(ap.parse_args())

prediction = False

if not (args['c'] or args['x']):
    ap.error('No action requested, add -c or -x')
if not args['c'] and args['d']:
    ap.error('Cannot display matrixes unless -c')

image = load_image(args['input'])
target_shape = image.shape

# Simulate compression
if prediction:
    imagei32 = np.array(image,dtype=np.int32)
    prediction = (imagei32[:-1,:-1]+imagei32[1:,:-1]+imagei32[:-1,1:]+1)//3
    error = np.abs(imagei32[1:,1:]-prediction)
    byteseq = compress_image(error,verbose=2,display=args['d'],
        display_base_fname=args['output'])
else:
    byteseq = compress_image(image,verbose=2,display=args['d'],
        display_base_fname=args['output'])

print("Decompressing to assert that the reconstruction is right.")
image2 = decompress_image(byteseq)
assert(np.all(image==image2))
# Compute radio
rad = 4*byteseq.size/(target_shape[0]*target_shape[1]*target_shape[2])

print("data : %d"%(4*byteseq.size))
print("radio: %f"%rad)
