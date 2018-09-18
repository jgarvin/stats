#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import matplotlib
import math
matplotlib.use("TkAgg") # without this nothing appears?!
import matplotlib.pyplot as plt
import numpy as np
import sys

def calculate_bernoulli_sample_mean(n, p):
    # s = sum([(1 if random.uniform(0, 1) <= p else 0) for i in range(n)])
    # print("n:",n," s:",s," s/n:",s/n)
    # return sum([(1 if random.uniform(0, 1) <= p else 0) for i in range(int(n))]) / int(n)
    return np.sum(np.random.binomial(1,p, int(n))) / int(n)

def calculate_epsilon(n, alpha):
    return math.sqrt((1/(2*n))*math.log(2/alpha))

def simulate():
    p = 0.4
    alpha = 0.05
    coverages = []
    sample_sizes = []
    interval_lengths = []
    simulations = 10000
    for sample_size in np.logspace(1, 5, 150):
        print(sample_size)
        samples = [calculate_bernoulli_sample_mean(sample_size, p) for x in range(simulations)]
        epsilon = calculate_epsilon(sample_size, alpha)

        coverage = 0
        for sample in samples:
            # print(sample)
            interval_lower_bound = sample - epsilon
            interval_upper_bound = sample + epsilon
            if interval_lower_bound < p and p < interval_upper_bound:
                coverage += 1

        coverage /= len(samples)
        # print(coverage)
        coverages.append(coverage)
        sample_sizes.append(sample_size)

    plt.plot(sample_sizes, coverages)
    plt.show()

def interval_versus_size():
    alpha = 0.05
    interval_lengths = []
    sample_sizes = []
    for sample_size in np.linspace(1, 100, 150):
        epsilon = calculate_epsilon(sample_size, alpha)
        interval_lengths.append(2*epsilon)
        sample_sizes.append(sample_size)

    plt.plot(sample_sizes, interval_lengths)
    plt.show()

# simulate()
interval_versus_size()