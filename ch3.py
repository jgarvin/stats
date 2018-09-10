#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import matplotlib
import math
matplotlib.use("TkAgg") # without this nothing appears?!
import matplotlib.pyplot as plt
import numpy as np
import sys

def calculate_sample_mean(n):
    return sum([random.uniform(0, 1) for i in range(n)]) / n

def distribution_of_sample_means():
    sample_size = 100
    samples = [calculate_sample_mean(sample_size) for x in range(10000)]

    mean = (sum(samples)/len(samples))
    print(mean)
    variance = sum([(x - mean)**2 for x in samples])/len(samples)
    print(variance)
    print(1/(12*sample_size))

    n, bins, patches = plt.hist(samples, bins='auto')
    print(n)

    plt.ylabel('Occurrences of X')
    plt.show()

distribution_of_sample_means()