#!/usr/bin/env python

from multiprocessing import Pool
import time 
import numpy as np


def f(i):
    if i < len(learning_rates):
        obj = learning_rates[i]
        print(f'{obj} started, {i}')
        return 1
    else:
        i -= len(learning_rates)

    if i < len(epsilon_decays):
        obj = epsilon_decays[i]
        print(f'{obj} started, {i}')
        return 1
    else:
        i -= len(epsilon_decays)

    if i < len(batch_sizes):
        obj = batch_sizes[i]
        print(f'{obj} started, {i}')
        return 1
    else:
        i -= len(batch_sizes)

    if i < len(target_updates):
        obj = target_updates[i]
        print(f'{obj} started, {i}')
        return 1
    else:
        i -= len(target_updates)


learning_rates = np.geomspace(0.0001, 0.1, 10)
epsilon_decays = [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]
target_updates = [5, 10, 25, 50, 75, 100, 250, 500, 1000]

with Pool(9) as p:
    p.map(f, list(range(len(learning_rates)+len(epsilon_decays)+len(batch_sizes)+len(target_updates))))
