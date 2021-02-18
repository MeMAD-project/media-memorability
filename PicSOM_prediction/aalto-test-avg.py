#! /usr/bin/env python3

import sys
import numpy as np

n = len(sys.argv)-1

if n<1:
    print('Expected file names as arguments')
    exit(0)

a = np.zeros((2000, n))
v = []

for fi in range(0, n):
    with open(sys.argv[fi+1]) as fp:
        i = 0
        for line in fp:
            line = line.strip()
            p = line.split(',')
            # print(p[0], p[1], p[2])
            if len(v)<=i:
                v.append(p[0])
            else:
                assert v[i]==p[0]
            a[i, fi] = float(p[1])
            if False and i<3:
                print(fi, sys.argv[fi+1], i, float(p[1]))
            i += 1

if False:
    print(a[0:3,:])
            
# print('***')

avg = np.average(a, 1)
for i in range(0, 2000):
    print(v[i], avg[i], 1, sep=',')
