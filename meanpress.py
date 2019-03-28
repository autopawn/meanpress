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
    parser.error('No action requested, add -c or -x')

image = load_image(args['input'])
target_shape = image.shape

# Simulate compression
channel = image[...,0]
arrays,start,axes,means = decompose_channel(channel)
for i in range(len(arrays)):
    delta,b,k = arrays[i]

# Simulate reconstruction
result = recompose_channel(arrays,start,target_shape)

print(np.all(channel==result))
assert(np.all(channel==result))
