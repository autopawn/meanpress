import numpy as np

mode_a = lambda x,y: (x+y+1)//2
mode_b = lambda x,y: (x+1)//2 + y//2
mode_c = lambda x,y: (x+y)//2
mode_d = lambda x,y: x//2 + y//2

combs = {'mode_a':{},'mode_b':{},'mode_c':{},'mode_d':{}}
funcs = {'mode_a':mode_a,'mode_b':mode_b,'mode_c':mode_c,'mode_d':mode_d}


for i in range(256):
    for ko in combs:
        combs[ko][i] = []

for x in range(256):
    for y in range(256):
        for ko in combs:
            res = funcs[ko](x,y)
            combs[ko][res].append((x,y))

modes = sorted([x for x in combs])
for res in (0,1,2,3,4,251,252,253,254,255):
    print("res: %d"%res)
    for ko in modes:
        if len(combs[ko][res])==0:
            print("%s : never"%(ko))
        else:
            min_x = np.min([x for (x,y) in combs[ko][res]])
            max_x = np.max([x for (x,y) in combs[ko][res]])
            print("%s : %3d - %3d"%(ko,min_x,max_x))

# mode_b seems better
