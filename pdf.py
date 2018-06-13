#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import matplotlib
import math
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

def pdf_ch2e13(y):
    return (1/math.sqrt(2 * math.pi) * (y**(-math.log(y, math.e) / 2 -1)))
    # return (1/math.sqrt(2 * math.pi) * (math.e**(-(math.log(y, math.e)**2) / 2)) * (1/y))

def pdf_ch2e13_pdf():
    plt.plot(np.linspace(0.001, 10), [pdf_ch2e13(x) for x in np.linspace(0.001, 10)])
    plt.show()

def pdf_ch2e13_hist():
    x = [np.random.normal() for i in range(100000)]
    y = [math.e**i for i in x]

    fig = plt.figure()
    ax2 = fig.add_subplot(111)

    r = np.linspace(0.0001, 10, 1000)
    hist, bins = np.histogram(y, list(r) + [float("inf")], density=True)
    ax2.plot(r, [pdf_ch2e13(x) for x in r], color="red", lw=3)
    ax2.hist(y, bins, normed=True)
    ax2.set_ylabel('e^(random_normal)')

    plt.show()

def exp_pdf(beta, x):
    return (1/beta) * (math.e**(-x/beta))

def exp_inverse_cdf(beta, y):
    return -beta * math.log(1 - y, math.e)

def random_exp(beta):
    return exp_inverse_cdf(beta, random.uniform(0,1))

def pdf_ch2e15_hist():
    beta = 2

    x = [random_exp(beta) for i in range(100000)]

    fig = plt.figure()
    ax2 = fig.add_subplot(111)

    r = np.linspace(0.0001, 10, 1000)
    hist, bins = np.histogram(x, list(r) + [float("inf")], density=True)
    ax2.plot(r, [exp_pdf(beta, i) for i in r], color="red", lw=3)
    ax2.hist(x, list(r) + [float("inf")], normed=True) # correct graph

    plt.show()

def plot_norm():
    samples = [np.random.normal(0,1) for x in range(10000)]

    n, bins, patches = plt.hist(samples)
    print(n)

    plt.ylabel('Occurrences of X')
    plt.show()

def plot_ch3e9():
    samples = [np.random.normal(0,1) for x in range(10000)]
    # samples = [np.random.standard_cauchy() for x in range(10000)]
    means = []
    total = 0
    for i, sample in enumerate(samples):
        total += sample
        means.append(total / (i+1))
    plt.plot(range(len(means)), means)
    plt.show()

def plot_ch3e10():
    for i in range(30):
        ticks = [random.choice([-1, 1]) for x in range(10000)]
        timeseries = []
        total = 0
        for tick in ticks:
            total += tick
            timeseries.append(total)
        print(total)
        plt.plot(range(len(timeseries)), timeseries)
    plt.show()

def plot_ch3e13():
    values = []
    total = 0
    sum_of_squares = 0
    for i in range(100000):
        flip = random.choice([0, 1])
        if flip == 0:
            values.append(random.uniform(3, 4))
        elif flip == 1:
            values.append(random.uniform(0, 1))

        total += values[-1]
        mean = total / len(values)
        print("mean ", mean)
        # sum_of_squares += (values[-1])
        variance = sum([(x - 2)**2 for x in values]) / len(values)
        print("variance ", variance)

def plot_norm2():
    # samples = [min(1/np.random.normal(0,1), 0.2) for x in range(10000)]
    # samples = [np.random.normal(0,1) for x in range(10000)]
    samples = [1/np.random.normal(0,1) for x in range(10000)]
    lowest = abs(min(samples))
    highest = abs(max(samples))
    clamp = min(lowest, highest)
    # samples = [max(min(x, clamp), -clamp) for x in samples]
    # samples = [x/clamp for x in samples]
    print(min(samples))
    print(max(samples))
    # samples = [min(x, 0) for x in samples]
    # samples = [max(x, 1) for x in samples]

    n, bins, patches = plt.hist(samples, bins='auto')
    print(n)
    print(max(n))
    print(np.argmax(n))
    print(len(n))
    print(bins)
    print(patches)

    plt.ylabel('Occurrences of X')
    plt.show()

def calculate_percentile(sorted_samples, percentile):
    assert(0 <= percentile)
    assert(1 >= percentile)
    index = math.floor(len(sorted_samples) * percentile)
    return (index, sorted_samples[index])

def interquartile_range(sorted_samples):
    return calculate_percentile(sorted_samples, 0.75) - calculate_percentile(sorted_samples, 0.25)

def calculate_bin_width(sorted_samples):
    # Freedman–Diaconis rule
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    return 2*interquartile_range(sorted_samples)/(len(sorted_samples)**(1/3))

def sample_to_bucket(sample, minimum_sample, maximum_sample, bin_width):
    return ☢
    
    samples_range = sorted_samples[-1] - sorted_samples[0]

def calculate_sample_probability(sorted_samples):
    samples_range = sorted_samples[-1] - sorted_samples[0]


# plot_norm()
plot_norm2()
# plot_ch3e9()
# plot_ch3e10()
# plot_ch3e13()
# pdf_ch2e15_hist()
