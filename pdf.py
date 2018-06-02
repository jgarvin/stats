#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import matplotlib
matplotlib.use("TkAgg") # without this nothing appears?!
import matplotlib.pyplot as plt

def ch2e4rX():
    "1/4 for 0<x<1"
    "3/8 for 3<x<5"
    choose_case = random.uniform(0, 1)
    if choose_case <= 1/4:
        return random.uniform(0, 1)
    else:
        return random.uniform(3, 5)

samples = [ch2e4rX() for x in range(10000)]

n, bins, patches = plt.hist(samples, range(6))
print(n)

plt.ylabel('Occurrences of X')
plt.show()
