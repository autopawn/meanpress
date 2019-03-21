import argparse

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

arrays,axes,means = decompose_channel(image[...,0])
for i in range(len(arrays)):
    delta,b,k = arrays[i]
    print("%d %10s %10s %10s"%(axes[i],delta.shape,k.shape,b.shape))
