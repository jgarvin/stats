#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import matplotlib
matplotlib.use("TkAgg") # without this nothing appears?!
import matplotlib.pyplot as plt
import numpy as np
import sys

def ch2e4rX():
    "1/4 for 0<x<1"
    "3/8 for 3<x<5"
    choose_case = random.uniform(0, 1)
    if choose_case <= 1/4:
        return random.uniform(0, 1)
    else:
        return random.uniform(3, 5)

def ch2e4rZ():
    # "1/4 for 0<x<1"
    # "3/8 for 3<x<5"
    choose_case = random.uniform(0, 1)
    if choose_case <= 1/4:
        return random.uniform(1, sys.float_info.max)
    else:
        return random.uniform(1/5, 1/3)

def plot_ch2e4rX():
    samples = [ch2e4rX() for x in range(10000)]

    n, bins, patches = plt.hist(samples, range(6))
    print(n)

    plt.ylabel('Occurrences of X')
    plt.show()

def plot_ch2e4rY():
    fig,(ax1,ax2) = plt.subplots(1, 2)

    samples = [1/ch2e4rX() for x in range(10000)]

    hist, bins = np.histogram(samples, [0, 1/5, 1/3, 1, float("inf")])
    print(hist)
    ax1.bar(range(len(hist)),hist,width=1,align='center',tick_label=
        ['{:.2f} - {:.2f}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])
    ax1.set_ylabel('Occurrences of Y')

    samples = [ch2e4rZ() for x in range(10000)]
    hist, bins = np.histogram(samples, [0, 1/5, 1/3, 1, float("inf")])
    print(hist)
    ax2.bar(range(len(hist)),hist,width=1,align='center',tick_label=
        ['{:.2f} - {:.2f}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])
    ax2.set_ylabel('Occurrences of Z')

    plt.show()

plot_ch2e4rY()