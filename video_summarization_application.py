import os
import sys
import pickle
import numpy as np
from algorithms import *
from utility_functions import *
import time


reg_coef = 0.1
lambdaa = 1.0
dataset = get_video_data(lambdaa = lambdaa, reg_coef = reg_coef)

alpha = 20
max_each_genres = 10
f = LogDet(dataset, alpha = alpha)
capacity = 2.0
cardinality = 10000


costs = copy.copy(dataset['costs']) # keep a copy of the original costs.
# Normalizing the knapsack cost
for e in dataset['elements']:
	dataset['costs'][e] = [val / capacity for val in costs[e]]

constraint = MatroidConstraint(dataset, max_movies = cardinality, 
	max_each_genres = max_each_genres)


print('We report the objective values and the number of Oracle calls.')


f.oracle_calls = 0
val, sol = greedy(f, dataset, constraint)
print('Greedy:', f.evaluate(sol)[0], f.oracle_calls)

f.oracle_calls = 0
val, sol = density_greedy(f, dataset, constraint)
print('Density-Greedy', f.evaluate(sol)[0], f.oracle_calls)


f.oracle_calls = 0
val, sol = barrier_heuristic(f, dataset, constraint, a = dataset['nb_knapsacks'])
print('Barrier-Greedy', f.evaluate(sol)[0], f.oracle_calls)

f.oracle_calls = 0
val, sol = fast_algorithms(f, dataset, constraint)
print('FAST', f.evaluate(sol)[0], f.oracle_calls)