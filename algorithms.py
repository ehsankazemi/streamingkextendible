from __future__ import print_function
import numpy as np
import time
import random
import sys
from utility_functions import *
import math
import copy


def barrier_greedy(f,dataset,constraint,epsilon = 0.5):
	#
	# Algorithm 1 BARRIER-GREEDY
	# The BARRIER-GREEDY algorithm starts with an empty set S and performs 
	# the following steps for at most r log(1/ε) iterations or 
	# till it reaches a solution S such that f(S) ≥ (1−ε) / (k+1) * Ω
	#	1. Firstly, it finds an element b ∈ N \S with the maximum
	#		value of delta_b − \sum delta_{a_i} such that S−a_i + b∈ I_i.
	#	2. In the second step, BARRIER-GREEDY removes all elements with δa ≤ 0 form set S. 
	#
	# BARRIER-GREEDY computes values of δa from Eq. (3). Note that, in this step, 
	# we need to compute δa for all el- ements a ∈ N only once and store them; 
	# then we can use these pre-computed values to find the best candidate b. 
	# The goal of this step is to find an element b such that its addition to set S 
	# and removal of a corresponding set of elements from S decrease the potential 
	# function by a large margin while still keeping the solution feasible.
	#
	a = 1.0
	M = max([f.evaluate(e)[0] for e in dataset['elements']]) # finds the maximum singelton value
	opt_guesses = []

	#
	# finds the set of potential guesses for the optimum value
	#
	guess = 5 * M
	while  guess <= 5 * M * constraint.max_cardinality: 
		opt_guesses.append(guess)
		guess *= (1 + epsilon)

	final_val = -1 * sys.maxsize
	max_iterations = constraint.max_cardinality * math.log(1 / epsilon)
	final_soluton = []
	for guess in opt_guesses:
		guess_flag = True
		iterations = 0
		S = CandidateSol()
		S_val = 0
		gamma_S = 0
		while gamma_S <= 1.0 and iterations < max_iterations:
			iterations+=1
			J_b_sol, b = find_best_replacement(dataset, constraint, f, guess, S, a)
			S.remove_elements(list(J_b_sol))
			S.add_elements(b)
			S_val, S.auxiliary = f.evaluate(S.S_list)
			gamma_S = gamma_set(dataset, S.S_list)
			flag = True
			#
			# Remove all the elements with non-positve delta
			#
			while flag:
				flag = False
				for e in copy.copy(S.S_list):
					val = delta_e(dataset, constraint, f, guess, S, e, S_val, gamma_S, a)[0]
					if val <= 0:
						S.remove_elements(e)
						flag = True
						S_val, S.auxiliary = f.evaluate(S.S_list)
						gamma_S = gamma_set(dataset, S.S_list)

		S_costs = [constraint.costs[e] for e in S.S_list]
		if constraint.is_feasible(S.S_list):
			current_val, __ = f.evaluate(S.S_list)	
			if current_val > final_val:
				final_val = current_val
				final_soluton = copy.copy(S.S_list)
		else:
			# 
			# If the final solution is not feasible we need to remove the last added element
			#
			S_b = copy.copy(S.S_list)
			S_b.remove(b)
			S_b_val, __ = f.evaluate(S_b)
			b_val, __ = f.evaluate(b)
			if S_b_val > final_val:
				final_val = S_b_val
				final_soluton = copy.copy(S_b)
			if b_val > final_val:
				final_val = b_val
				final_soluton = [b]
		if len(final_soluton)>0:
			S_costs = [constraint.costs[e] for e in final_soluton]

	return final_val,  final_soluton

def best_swap(f, dataset, constraint, OPT, S, compute_deltas, S_counts, S_genres):
	#
	# If finds the possible elements for swapping based line 6 of 
	# Algorithm 1 BARRIER-GREEDY.
	#
	max_delta = -1 * sys.maxsize
	J_b_sol = set()
	b_sol = -1
	for b in dataset['elements']:
		if b not in S.S_set:
			J_b = set()
			for genre in constraint.genres[b]:
				#
				# Check if adding b violates the matroid constraint.
				#
				if S_counts[genre] >= constraint.max_each_genres:
					min_delta = sys.maxsize
					a_b = -1
					for e in S_genres[genre]:
						val = compute_deltas[e]
						if val < min_delta:
							min_delta = val
							a_b = e
					J_b.add(a_b)

			if len(S.S_list) == constraint.max_movies and len(J_b) == 0:
				min_delta = sys.maxsize
				a_b = -1
				for e in S.S_list:
					val = compute_deltas[e]
					if val < min_delta:
						min_delta = val
						a_b = e
				J_b.add(a_b)

			J_b_sum = compute_deltas[b]
			for e in J_b:
				J_b_sum -= compute_deltas[e]
			#
			# Keep the element with the largest delta_b - \sum_{i \in J_b} \delta_{a_i}
			#
			if J_b_sum > max_delta:
				J_b_sol = copy.copy(J_b)
				max_delta = J_b_sum
				b_sol = b

	return J_b_sol, b_sol

def gamma_set(dataset, e):
	#
	# Calculated gamma(e) = \sum_{i=1}^{\ell} c_{i,e} for an element e
	# I could be also used to calculate the gamma for a set.
	#
	if isinstance(e, int):
		return round(sum(dataset['costs'][e]), 4)
	else:
		return round(sum( [sum(dataset['costs'][u]) for u in e]), 4)

def W_e(dataset, f, e, S, current_val = None):
	#
	# for an element a ∈ N the contribution of a to S (denoted by wa) is the marginal
	# gain of adding element a to all elements of S that are smaller than a, i.e., 
	# wa = f(S∩[a])−f(S∩[a−1]).
	# The benefit of adding b ∈/ S to set S (denoted by wb) is the marginal gain of 
	# dding el-
	#
	#
	if e not in S.S_set:
		return f.marginal_gain(e, S.S_list, current_val, S.auxiliary)[0]
	else:
		e_ind = S.find_index(e)
		S_before = S.S_list[:e_ind]
		return f.marginal_gain(e, S_before)[0]

def delta_e(dataset, constraint, f, OPT, S, e, S_val = None, gamma_S = None, a = 1.):
	if gamma_S == None:
		gamma_S = gamma_set(dataset, S.S_list)
	if S_val == None:
		S_val, S.auxiliary = f.evaluate(S.S_list)
	return ( (constraint.k + 1) * (a - gamma_S )* W_e(dataset, f, e, S, S_val)) - \
	 ( (OPT - (constraint.k + 1) * S_val)) * gamma_set(dataset, e), S_val, gamma_S 


def find_best_replacement(dataset, constraint, f, OPT, S, a):
	#
	# To perform line 6 of Algorithm 1 and line 8 of Algoirthm 4 we first 
	# pre-calculate the values of delta for all elements
	# a = 1: runs best_swap() for algorithm 1
	# a > 1: run best_swap_heuristic() for algorithm 4
	#
	compute_deltas = dict()
	S_val = None
	gamma_S = None
	for e in dataset['elements']:
	    compute_deltas[e], S_val, gamma_S \
	    = delta_e(dataset, constraint, f, OPT, S, e, S_val, gamma_S, a)
	S_counts = dict()
	S_genres = defaultdict(list)
	for genre in constraint.recommended_genres: 
		S_counts[genre] = 0
	for e in S.S_list:
		for genre in dataset['genres'][e]: #This is the matroids of each element
			if genre in constraint.recommended_genres:
				S_counts[genre] += 1
				S_genres[genre].append(e)

	if a == 1.0:     
		return best_swap(f, dataset, constraint, OPT, S, compute_deltas, S_counts, S_genres)
	else:
		return best_swap_heuristic(f, dataset, constraint, OPT, S, compute_deltas, S_counts, S_genres)

def best_swap_heuristic(f, dataset, constraint, OPT, S, compute_deltas, S_counts, S_genres):
	#
	# Implement line 8 of Algorithm 4 - BARRIER-HEURISTIC
	#
	max_delta = -1 * sys.maxsize
	J_b_sol = set()
	b_sol = -1
	for b in dataset['elements']:
		if b not in S.S_set:
			J_b = set()
			for genre in constraint.genres[b]:
				if S_counts[genre] >= constraint.max_each_genres:
					min_delta = sys.maxsize
					a_b = -1
					for e in S_genres[genre]:
						val = compute_deltas[e]
						if val < min_delta:
							min_delta = val
							a_b = e
					J_b.add(a_b)

			if len(S.S_list) == constraint.max_movies and len(J_b) == 0:
				min_delta = sys.maxsize
				a_b = -1
				for e in S.S_list:
					val = compute_deltas[e]
					if val < min_delta:
						min_delta = val
						a_b = e
				J_b.add(a_b)
			J_b_sum = compute_deltas[b]
			for e in J_b:
				J_b_sum -= compute_deltas[e]
			
			# In the heuristic algorithm we need to check if the swaps are feasbile or not,
			# because we are not sure anymore that all the knapsack constraints are satisfied.
			if J_b_sum > max_delta and constraint.is_feasible( (S.S_set - J_b) | {b} ):
				J_b_sol = copy.copy(J_b)
				max_delta = J_b_sum
				b_sol = b
	return J_b_sol, b_sol

def barrier_heuristic(f,dataset,constraint,epsilon = 0.5, a = 1.0, lower=2, higher=5):
	M = max([f.evaluate(e)[0] for e in dataset['elements']])
	opt_guesses = []
	guess = lower * M
	while  guess <= higher * M * constraint.max_cardinality: #changed
		opt_guesses.append(guess)
		guess *= (1 + epsilon)
	final_val = -1 * sys.maxsize
	for guess in reversed(opt_guesses):
		guess_flag = True
		iterations = 0
		max_iterations = constraint.max_cardinality * math.log(1 / epsilon)
		S = CandidateSol()
		S_val = None
		while guess_flag and iterations < max_iterations: # Line 5 of Algorithm 4
			iterations+=1
			J_b_sol, b = find_best_replacement(dataset, constraint, f, guess, S, a)
			if b != -1: #changed
				S.remove_elements(list(J_b_sol))
				S.add_elements(b)
				flag = True
				S_val, S.auxiliary = f.evaluate(S.S_list)
				gamma_S = gamma_set(dataset, S.S_list)
			else:
				guess_flag = False
			# if guess flag is False then N′ = ∅ and as the line 7 of Algorithm 4 the algorithm ends.
			if guess_flag:
				#
				# Remove all the elements with non-positve delta
				#
				while flag:
					flag = False
					for e in copy.copy(S.S_list):
						val = delta_e(dataset, constraint, f, guess, S, e, S_val, gamma_S, a)[0]
						if val <= 0:
							S.remove_elements(e)
							flag = True
							S_val, S.auxiliary = f.evaluate(S.S_list)
							gamma_S = gamma_set(dataset, S.S_list)
		S_costs = [constraint.costs[e] for e in S.S_list]
		current_val = f.evaluate(S.S_list)[0]
		if current_val > final_val:
			final_val = current_val
			final_soluton = copy.copy(S.S_list)
	pairs = []
	for e in dataset['elements']:
		if e not in set(final_soluton):
			pairs.append((e, constraint.costs[e][0]))
	pairs.sort(key=lambda x: x[1])
	for pair in pairs:
		if constraint.is_add_feasible(pair[0], final_soluton):
			final_soluton.append(pair[0])
	final_val = f.evaluate(S.S_list)[0]
	return final_val,  final_soluton
	
		
def FANTOM(f,dataset,constraint,epsilon = 0.5):
	#
	# Algorihm 3 from:
	# Mirzasoleiman, B., Badanidiyuru, A., and Karbasi, A. 
	# Fast Constrained Submodular Maximization: Personalized Data Summarization. 
	# In ICML, pp. 1358–1367, 2016a.
	#
	n = len(dataset['elements'])
	vals_elements = [f.evaluate(e)[0] for e in dataset['elements']]
	M = max(vals_elements)
	U = []
	omgea = set([e for e in dataset['elements']])
	S = CandidateSol()
	p = constraint.nb_matroids
	ell = constraint.nb_knapsacks
	gamma = (2 * p * M) / ( (p+1) * (2 * p + 1))
	R = []
	val = 1
	while  val <= n:
		R.append(val * gamma)                                                                                         
		val *= (1 + epsilon)

	for rho in R:
		omega = set([e for e in dataset['elements']])
		sols = iterated_GDT(f,dataset,constraint,omega,rho)
		for sol in sols:
			U.append(sol)

	final_soluton = []
	final_val = -1 * sys.maxsize
	for sol in U:
		current_val, __ = f.evaluate(sol)
		if current_val > final_val:
			final_val = current_val
			final_soluton = copy.copy(sol)

	return final_val,  final_soluton

def iterated_GDT(f, dataset,constraint,ground,rho):
	#
	# Algorihm 2: IGDT - Iterated greedy with density threshold
	# Mirzasoleiman, B., Badanidiyuru, A., and Karbasi, A. 
	# Fast Constrained Submodular Maximization: Personalized Data Summarization. 
	# In ICML, pp. 1358–1367, 2016a.
	#
	S = CandidateSol()
	p = constraint.nb_matroids
	ell = constraint.nb_knapsacks
	S_i = []
	for i in range(p+1):
		S = GDT(f,dataset,constraint,ground,rho)
		S_i.append(copy.copy(S.S_list))
		ground = ground - S.S_set
	return S_i

def GDT(f, dataset, constraint, ground, rho):
	#
	# Algorihm 1: GDT - Greedy with density threshold
	# Mirzasoleiman, B., Badanidiyuru, A., and Karbasi, A. 
	# Fast Constrained Submodular Maximization: Personalized Data Summarization. 
	# greedy with density threshold
	# runs the greedy on the on the groundser
	# an element is added if its marginal gain is larger than a threshold
	#
	S = CandidateSol()
	current_val = None
	flag = True
	while flag:
		flag = False
		cand = -1
		cand_val = -1 * sys.maxsize
		for e in ground:
			if constraint.is_add_feasible(e, S.S_list):
				val, __ = f.marginal_gain(e, S.S_list, current_val, S.auxiliary)
				if  val / (sum(constraint.costs[e]) + 0.000001) >= rho:
					if val > cand_val:
						cand = e
						cand_val = val
						flag = True
		if cand != -1:
			S.add_elements(cand)
			current_val, S.auxiliary = f.evaluate(S.S_list)
	return S

def fast_algorithms(f,dataset,constraint, epsilon = 0.2):
	#
	# Algorithm 10 from the following paper:
	# Badanidiyuru, A. and Vondrák, J. 
	# Fast algorithms for maximizing submodular functions. 
	# In ACM-SIAM symposium on Discrete algorithms (SODA), pp. 1497–1514, 2014.
	#
	n = len(dataset['elements'])
	vals_elements = [f.evaluate(e)[0] for e in dataset['elements']]
	M = max(vals_elements)

	p = constraint.nb_matroids
	ell = constraint.nb_knapsacks

	threshold_guesses = []
	guess = M / (p + ell * 1.0)
	while  guess <= (2 * constraint.max_cardinality * M) / (p + ell * 1.0):
		threshold_guesses.append(guess)
		guess *= (1 + epsilon)

	T_sols = []
	T_prime_sols = []
	for rho in threshold_guesses:
		vals_elements = [f.evaluate(e)[0] for e in dataset['elements']]
		rho_vals = [vals_elements[i] for i in range(n) if  (sum(constraint.costs[i]) == 0 or (vals_elements[i] / sum(constraint.costs[i])) >= rho )]
		if len(rho_vals) > 0:
			M_rho = max(rho_vals)
			tau = M_rho
			S = CandidateSol()
			current_val = None
			S_rho = set()
			violate_flag = False
			while tau >= (epsilon / n ) * M_rho and constraint.check_costs(S.S_list):
				for j in dataset['elements']:
					if all([c <= 1 for c in constraint.costs[j]]):
						margin_j, __ = f.marginal_gain(j, S.S_list, current_val, S.auxiliary)
						if margin_j >= tau and (sum(constraint.costs[j]) == 0 or margin_j / sum(constraint.costs[j]) >= rho) and constraint.is_psystem_add_feasible(j, S.S_list):
							S.add_elements(j)
							current_val, S.auxiliary = f.evaluate(S.S_list)
							violate_flag = False
							if not constraint.check_costs(S.S_list):
								S_rho = copy.copy(S.S_set)
								T_rho = copy.copy(S.S_set)
								T_rho.remove(j)
								T_prime_pho = {j}
								violate_flag = True
								break
						if violate_flag:
							break
				if violate_flag:
					break
				tau = tau / (1.0 + epsilon)

		if not violate_flag:
			T_rho = copy.copy(S.S_set)
			T_prime_pho = []

		T_sols.append(copy.copy(T_rho))
		if len(T_prime_pho) > 0:
			T_prime_sols.append(copy.copy(T_prime_pho))

	final_soluton = []
	final_val = -1 * sys.maxsize
	for sol in T_prime_sols:
		current_val, __ = f.evaluate(sol)
		if current_val > final_val:
			final_val = current_val
			final_soluton = copy.copy(sol)

	for sol in T_sols:
		current_val, __ = f.evaluate(sol)
		if current_val > final_val:
			final_val = current_val
			final_soluton = copy.copy(sol)
	return final_val, final_soluton



def greedy(func, dataset, constraint):
	#
	# It starts with an empty set S and keeps adding elements one by one greedily 
	# (according to their marginal gain) while the k-system and l-knapsack constraints 
	# are both satisfied.
	#
	auxiliary = set()
	current_val = 0
	S = CandidateSol()
	auxiliary = set()
	flag = True
	while flag:
		flag = False
		max_margin = -1 * sys.maxsize
		best_element = -1
		best_auxiliary = set()
		for e in dataset['elements']:
			if constraint.is_add_feasible(e, S.S_list):
				flag = constraint.is_feasible([e] + S.S_list)
				if not flag:
					exit()
				gain_g, new_auxiliary = func.marginal_gain(e, S.S_list, current_val, auxiliary)
				margin = gain_g 
				if margin > max_margin:
					max_margin = margin
					best_element = e
		if max_margin > 0:
			current_val, auxiliary = func.add_one_element(best_element, S.S_list, current_val, auxiliary)
			S.add_elements(best_element)
			S.auxiliary = auxiliary
			flag = True
	return current_val, S.S_list

def density_greedy(func, dataset, constraint):
	#
	# Density Greedy, starts with an empty set S and keeps adding elements greedily 
	# by the ratio of their marginal gain to the total knapsack cost of each element 
	# while the k-system and l-knapsack constraints are satisfied. 
	#
	auxiliary = set()
	current_val = 0
	S = CandidateSol()
	auxiliary = set()
	flag = True
	while flag:
		flag = False
		max_margin = -1 * sys.maxsize
		best_element = -1
		best_auxiliary = set()
		for e in dataset['elements']:
			if constraint.is_add_feasible(e, S.S_list):
				gain_g, new_auxiliary = func.marginal_gain(e, S.S_list, current_val, auxiliary)
				#
				#we add a very small value to each cost in order to avoid division by zero 
				#
				coset_e = gamma_set(dataset, e) + 0.000000001
				margin = gain_g / coset_e
				if margin > max_margin:
					max_margin = margin
					best_element = e
		if max_margin > 0:
			current_val, auxiliary = func.add_one_element(best_element, S.S_list, current_val, auxiliary)
			S.add_elements(best_element)
			S.auxiliary = auxiliary
			flag = True
	return current_val, S.S_list


def check_potential(f, constraint, S, guess, S_val):
	if f.evaluate(S.S_list) > guess /  (constraint.k + 1): 
		return False
	return True