import os
import csv
import numpy as np
from multiprocessing import Pool
import random as rand
from collections import defaultdict
import scipy.spatial.distance as sci
import math
import sys
import copy
from operator import add
import json
import random
from scipy.stats import expon
import datetime
import networkx as nx
from itertools import combinations
import pickle
from numpy import linalg as LA

def get_video_data(lambdaa, reg_coef = 1.0):  ## data_file has the features for all frames
	#
	# Our objective is to select a subset of frames from these videos in order to 
	# maximize a utility function f(S) (which represents the diversity of frames). 
	# We set limits for the maximum num- ber of allowed frames from each video 
	# (referred to as mi), where we consider the same value of mi for all five videos.
	# We also want to bound the total entropy of the selection as a proxy for the 
	# storage size of the selected summary.
	# We get the frames from 6 different videos and merge them into a single dataset
	# and the cost of each frame is proportional to its entropy
	#
	print('Importing data...')
	video_nbs = ['11', '12', '13', '14', '15', '16']
	genres = []
	elements = []
	cnt = 0
	costs = []
	matrices = []
	for video_nb in video_nbs:
		file_name = './datasets/features/v'+ video_nb + '.p'
		entropy_name = './datasets/entropy/v' + video_nb +'_entropy_quan4.p'
		data_file = pickle.load(open(file_name,'rb'))
		matrices.append(data_file)
		ent = pickle.load(open(entropy_name,'rb'))
		n = data_file.shape[0]
		for val in ent:
			costs.append([val[0] * reg_coef])
			cnt += 1
			genres.append([video_nb])
	data_file = np.concatenate((matrices[0],matrices[1],matrices[2],
		matrices[3],matrices[4],matrices[5]), axis=0)

	n = data_file.shape[0]
	elements = [i for i in range(n)]

	M = sci.squareform(sci.pdist(data_file, 'euclidean'))
	M = np.exp(-1 * lambdaa * M)

	nb_knapsacks = 1
	data_dict = {'similarity':M, 'sim_mat':M, 'elements': elements, 'nb_elements':len(elements), 
	'costs': costs, 'nb_knapsacks':nb_knapsacks, 'recommended_genres':set(video_nbs),
	'genres':genres}
	return data_dict


def get_graph_data(data_file, graph_file, directed = False):
	#
	# It loads a graph with its communities.
	#
	edges = defaultdict(set)
	edge_weights = dict()
	vertex_weights = dict()
	elements = set()
	graph=nx.Graph()
	with open(data_file, 'r') as f:
		for line in f:
			[e_0, e_1] = [int(e) for e in line.replace("\n","").split()]
			elements.add(e_0)
			elements.add(e_1)
			edges[e_0].add(e_1)
			ew = random.uniform(1,5)
			edge_weights[(e_0,e_1)] = ew
			if not directed:
				edges[e_1].add(e_0)
				edge_weights[(e_1,e_0)] = ew
			graph.add_edge(e_0,e_1)

	#
	# We have a parition matroid where each node belongs to a community.
	#
	loaded_comms = json.load(open('dataset/'+graph_file+'_comm.json'))

	genres = defaultdict(list)
	nb_matroids = 5 # We restricted the number of communities in each garph to 5.
	recommended_genres = set()

	for node,comm in loaded_comms.items():
		genres[int(node)] = [str(comm)]
		recommended_genres.add(str(comm))

	costs = dict()
	orig_cost= dict()
	nb_knapsacks = 1
	tot=0
	costs = defaultdict(list)
	for e in elements:
		cost = []
		for i in range(nb_knapsacks):
			val=max(1, min(graph.degree[e],20) - 6)
		tot+=val
		cost.append(val)
		costs[e] = cost
		orig_cost[e] = val
	
	vals = [0]
	for e in elements:
		for i in range(1):
			vals[i] += costs[e][i]
	rate = len(elements) / 5
	tot=0
	for e in elements:
		cost = costs[e]
		costs[e] = [rate * cost[i] / vals[i] for i in range(1)]
		tot+=costs[e][0]
	elements = list(elements)
	for e in elements:
		vertex_weights[e] = 1

	data_dict = {'recommended_genres': recommended_genres,
	'edge_weights':edge_weights, 'vertex_weights': vertex_weights, 'edges': edges, 'elements': elements, 
	'nb_elements':len(elements), 'costs': costs, 'nb_knapsacks':nb_knapsacks, 'genres':genres}
	return data_dict




def loadTwitterData(filename, nb_tweets=1000,application=1, startDate='2000-01-01',endDate='2020-01-01',reg_coef=1.0):
	#
    # This function loads the Twitter data
    #
	recommended_genres = {'@CNNBrk', '@BBCSport', '@WSJ', '@BuzzfeedNews', '@nytimes', '@espn'}
	counts = dict()
	for genre in recommended_genres:
		counts[genre]=0
	count = 0
	tweets = [] #List of tweets.
	M = {} #List of handles (i.e. unique streams we scraped from).
	with open(filename,'r') as f:
		for line in f:
			arr = line.split(',,') #Value are split up by double commas ',,'.
			temp = []
			
			handle = arr[0]
			if handle in recommended_genres:
				temp.append(handle) #handle.
				
				cleanText = arr[3]
				cleanText = cleanText.split(' ')
				temp.append(cleanText) #list of (cleaned) words in this tweet.
				
				averageRetweets = int(arr[5])/float(len(cleanText))
				temp.append(averageRetweets) #Number of retweets divided by number of words
				
				timestamp = arr[7].strip()
				temp.append(timestamp) #Date/time of tweet.
				temp.append(arr[2]) #Raw text of tweet.
				
				if timestamp[:10] >= startDate and timestamp[:10] <= endDate and averageRetweets > 0:
					if handle not in M:
						M[handle] = 0
					M[handle] += 1
					count += 1
					tweets.append(temp)	

	stream = []
	for tweet in tweets:
		timestamp = tweet[3]
		stream.append([tweet,timestamp]) 
		
	cleanStream = []
	for tweet in stream:
		cleanStream.append(tweet[0])

	cleanStream = cleanStream[:nb_tweets]
	elements = [i for i in range(nb_tweets)]

	tweets_len = dict()

	for e in elements:
		tweets_len[e] = len(cleanStream[e][1])

	genres = [] # It is used to define the partition matroid constraint

	#
    # For the first knapsack the cost of each tweet is proportional to the absolute
    # time difference (in months) between the tweet and the first of January 2019.
    #
    # For the second knapsack he cost of tweet e is proportional to the length of 
    # each tweet |W_e| which enables us to provide shorter summaries.
    #
	base=datetime.date(2019,1,1)
	costs = []
	tot_m = 0
	tot_l = 0
	for e in elements:
		date_time_str=cleanStream[e][3]
		date=datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S').date()
		delta = date - base
		counts[cleanStream[e][0]]+=1
		genres.append([cleanStream[e][0]])
		costs.append([abs(delta.days/30), len(cleanStream[e][1])])
		tot_m += (abs(delta.days/30)) 
		tot_l += len(cleanStream[e][1])

	#
	# Normalizing the knsapsack costs according to the explanations from the paper
	#
	vals = [0,0]
	for e in elements:
		for i in range(2):
			vals[i] += costs[e][i]
	rate = len(elements) / 5
	if application == 1:
		for e in elements:
			cost = costs[e]
			costs[e] = [rate * cost[i] / vals[i] for i in range(1)] 
		nb_knapsacks=1
	if application == 2:
		for e in elements:
			cost = costs[e]
			costs[e] = [rate * cost[i] / vals[i] for i in range(2)]
		nb_knapsacks=2

	dataset = {'tweets':cleanStream, 'handles':M, 'nb_elements':nb_tweets,
	'elements':elements, 'tweets_len':tweets_len, 'reg_coef':reg_coef,'genres':genres,
	'costs':costs,'nb_knapsacks':nb_knapsacks, 'recommended_genres':recommended_genres}

	return dataset

class TweetDiversity():
	#
    # The monotone submodular function used in the  twitter summarization application.
    #
	def __init__(self, dataset, power=0.5):
		self.dataset = dataset
		self.tweets = dataset['tweets']
		self.power = power
		self.oracle_calls = 0

	def evaluate(self, S):
		self.oracle_calls += 1
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if isinstance(S, list):
			S = list(set(S))
		if len(S) == 0:
			return 0, None
		counts = dict()
		for e in S:
			tweet = self.tweets[e]
			user = tweet[0]
			words = tweet[1]
			retweets = tweet[2]
			for word in words:
				if word not in counts:
					counts[word] = 0
				counts[word] += retweets
		score = 0.0 
		for word in counts:
			if counts[word] > 0:
				points = counts[word] ** (self.power)
			else:
				points = 0
			score += points
		return score, None

	def marginal_gain(self, e, S, current_val = None, auxiliary = None):
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if e in S:
			return 0.0, None
		if current_val == None:
			current_val, __ = self.evaluate(S)

		if isinstance(e, int):
			val, __ = self.evaluate(S + [e])
		else:
			val, __ = self.evaluate(S + e)
		return val - current_val, None

	def add_one_element(self, e, S, current_val = None, auxiliary = None):
		if e in S:
			return 0, None
		if isinstance(S, int):
			S = [S]
		if isinstance(e, list):
			return self.evaluate(S + e)
		else:
			return self.evaluate(S + [e])



def get_movie_recommendation_data(data_folder, lambdaa = 1, recommended_genres=None, movie_lim=None, application = 1, rate_coef = 1.0):
	#
	# our objective is to recommend a set of diverse movies to a user. 
	# For designing our recommender system, we use ratings from MovieLens dataset, 
	# and apply the method proposed by Lindgren et al. (2015) to extract a set of 
	# attributes for each movie. 
	#
	print('Importing data...', end='')

	data_mat = []
	genres = []
	titles = []
	# Adventure, Animation and Fantasy are the default genres.
	if recommended_genres is None:
		recommended_genres = ['Adventure', 'Animation', 'Fantasy']
	if recommended_genres is "all":
		recommended_genres = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
							  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
							  'Thriller', 'War', 'Western']
	recommended_genres = set(recommended_genres)

	# import meta data from movies.csv
	movie_info_dict = {}
	movie_info_file = os.path.join(data_folder, 'movies.csv')
	with open(movie_info_file, 'r') as movie_info_csv:
		reader = csv.reader(movie_info_csv)
		header = next(reader)
		# go through all rows
		for row in reader:
			# put info into dictionary
			id = int(row[0])
			title = row[1]
			genre_list = row[2].split('|')
			movie_info_dict[id] = {'title': title, 'genres': genre_list}

	# import feature matrix from mvs.csv
	dict_id_title = dict()
	movie_feat_file = os.path.join(data_folder, 'mvs.csv')
	with open(movie_feat_file, 'r') as movie_feat_csv:

		# create reader
		reader = csv.reader(movie_feat_csv)
		header = next(reader)
		# go through all rows
		for row in reader:
			# put info into
			id = int(row[0])
			feat_vec = [float(x) for x in row[1:]]
			# check if this movie has at least one of the desired genres
			if recommended_genres.isdisjoint(movie_info_dict[id]['genres']):
				continue
			else:
				data_mat.append(feat_vec)
				titles.append(movie_info_dict[id]['title'])
				genres.append(list( set(movie_info_dict[id]['genres']) &  recommended_genres)) 
				dict_id_title[movie_info_dict[id]['title']] = str(id)
	# format movie feature matrix into

	data_mat = np.array(data_mat)
	if movie_lim is not None:
		if movie_lim < data_mat.shape[0]:
			data_mat = data_mat[:movie_lim, :]
			titles = titles[:movie_lim]
			genres = genres[:movie_lim]
	M = sci.squareform(sci.pdist(data_mat, 'euclidean'))
	M = np.exp(-1 * lambdaa * M)

	print('data contains %d movies' % data_mat.shape[0])
	nb_knapsacks = 0
	movie_costs = []
	elements = [i for i in range(len(titles))]


	movie_year_rating_file = os.path.join(data_folder, 'year_rating')
	year_rating_dict = json.load(open(movie_year_rating_file))


	#
	# Normalizing the knsapsack costs according to the explanations from the paper
	#
	rate = len(elements)/ 10.0
	rate_coef = rate / 6058.3
	year_coef_one = rate / 31307.0
	year_coef_two = rate / 25798.0
	rate_base = 10
	year_base_one = 1985
	year_base_two = 2004
	if application == 1:
		for i,title in enumerate(titles):
			id = dict_id_title[title]
			rating_c = abs(year_rating_dict[id][3] - rate_base) * rate_coef
			movie_costs.append([rating_c])
			nb_knapsacks=1
	if application == 2:
		for i,title in enumerate(titles):
			id = dict_id_title[title]
			rating_c = abs( year_rating_dict[id][3] - rate_base) * rate_coef
			year_c = abs(year_rating_dict[id][2] - year_base_one) * year_coef_one 
			movie_costs.append([rating_c, year_c])
		nb_knapsacks=2
	if application == 4:
		for i,title in enumerate(titles):
			id = dict_id_title[title]
			year_c = abs(year_rating_dict[id][2] - year_base_one) * year_coef_one 
			movie_costs.append([year_c])
			nb_knapsacks=1
	vals = [0,0,0]
	new = [0,0,0]
	if application == 3:
		for i,title in enumerate(titles):
			id = dict_id_title[title]
			rating_c = abs( year_rating_dict[id][3] - rate_base) * rate_coef
			year_c = abs(year_rating_dict[id][2] - year_base_one) * year_coef_one 
			year_d = abs(year_rating_dict[id][2] - year_base_two) * year_coef_two
			movie_costs.append([rating_c, year_c, year_d])
			vals[0] += abs( year_rating_dict[id][3] - rate_base)
			vals[1] += abs(year_rating_dict[id][2] - year_base_one)
			vals[2] += abs(year_rating_dict[id][2] - year_base_two)
			new[0] += abs( year_rating_dict[id][3] - rate_base) * rate_coef
			new[1] += abs(year_rating_dict[id][2] - year_base_one) * year_coef_one 
			new[2] += abs(year_rating_dict[id][2] - year_base_two)* year_coef_two
		nb_knapsacks=3

	data_dict = {'matrix': data_mat, 'genres': genres, 'titles': titles, 
	'costs' : movie_costs, 'recommended_genres':recommended_genres, 
	'lambda': lambdaa,'sim_mat': M, 'nb_knapsacks':nb_knapsacks,
	'elements':elements}
	return data_dict


class LogDet():
	#
    # The monotone and submodular logdet function
    # It has three functions:
    #   1. evaluates the utility for any set S
    #   2. calculates the marginal gain of adding an element e to a set S
    #
	def __init__(self, dataset, alpha = 10):
		self.dataset = dataset
		self.alpha = alpha
		self.oracle_calls = 0

	def evaluate(self, S):
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if isinstance(S, list):
			S = list(set(S))

		n = len(S) 
		if n == 0:
			return 0, None
		M = np.identity(n) + self.alpha * self.dataset['sim_mat'][np.ix_(S,S)]
		self.oracle_calls+=1
		return math.log(np.linalg.det(M)), None

	def marginal_gain(self, e, S, current_val = None, auxiliary = None):
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if isinstance(S, list):
			S = list(set(S))

		if e in S:
			return 0.0, auxiliary

		if current_val == None:
			current_val, auxiliary = self.evaluate(S)

		if isinstance(e, int):
			val, __ = self.evaluate(S + [e])
		else:
			val, __ = self.evaluate(S + e)
		return val - current_val, auxiliary


	def add_one_element(self, e, S, current_val = None, auxiliary = None):
		if isinstance(S, int):
			S = [S]
		if e in S:
			return 0.0, None
		if isinstance(e, list):
			return self.evaluate(S + e)
		else:
			return self.evaluate(S + [e])

	def add_a_remove_b(self, a, b, S, current_val = None, auxiliary = None):
		if not isinstance(S, list):
			S = [S]
		if b is not None:
			if isinstance(b, list):
				S_b = [e for e in S if e not in set(b)]
			else:
				S_b = [e for e in S if e != b]
		else:
			S_b = S
		if a is not None:
			if isinstance(a, list):
				return self.evaluate(S_b + a)
			else:
				return self.evaluate(S_b + [a])
		else:
			return self.evaluate(S_b)

class MatroidConstraint():
	#
    # This the constaint which defines our matroid constraint.
    # Here generes represent categorires of itemes in differrent datasets: 
    # 	(i) genre for movies, 
    # 	(ii) video number for videos,
    #   (iii) handles for twitter accounts and 
    #	(iv) cities for Yelp dataset. 
    #
	def __init__(self, dataset, max_movies = None, max_each_genres = 5):
		self.genres = dataset['genres']
		self.costs = dataset['costs']
		self.nb_elements = len(dataset['elements'])
		self.recommended_genres = dataset['recommended_genres']
		self.max_movies = max_movies
		self.max_each_genres = max_each_genres
		self.nb_knapsacks = dataset['nb_knapsacks']
		self.dataset = dataset
		self.nb_matroids = len(dataset['recommended_genres'])
		if max_movies is not None:
			self.nb_matroids += 1
		self.k = max(self.nb_knapsacks, self.nb_matroids)
		if self.max_movies is None:
			self.max_movies = sys.maxsize

		self.max_cardinality = self.max_each_genres * self.k
		if max_movies is not None:
			self.max_cardinality = min(self.max_cardinality, self.max_movies)

	def is_feasible(self, S):
		if len(S) == 0:
			return True
		if not isinstance(S, set):
			S = set(S)
		if len(S) > self.max_movies:
			return False

		genre_count = dict()
		for genre in self.recommended_genres:
			genre_count[genre] = 0

		for e in S:
			for genre in self.genres[e]:
				if genre_count[genre] >= self.max_each_genres:
					return False
				else:
					genre_count[genre] += 1
		S_costs = [self.costs[e] for e in S]
		if round(max([sum(x) for x in zip(*S_costs)]), 4) > 1:
			return False

		return True
	def is_add_feasible(self, e, S):
		if isinstance(S, int):
			S = [S]
		if not isinstance(e, list):
			if e in S: 
				return False
			return self.is_feasible(S + [e])
		else:
			return self.is_feasible(S + e)

	def is_psystem_add_feasible(self, e, S):
		if isinstance(S, int):
			S = [S]
		if isinstance(e, int):
			e = [e]
		S_e = S + e
		S = set(S_e)
		if len(S) > self.max_movies:
			return False
		genre_count = dict()
		for genre in self.recommended_genres:
			genre_count[genre] = 0
		for e in S:
			for genre in self.genres[e]:
				if genre_count[genre] >= self.max_each_genres:
					return False
				else:
					genre_count[genre] += 1
		return True

	def check_costs(self,S):
		if len(S) == 0:
			return True
		S_costs = [self.costs[e] for e in S]
		if max([sum(x) for x in zip(*S_costs)]) > 1:
			return False
		return True


class CandidateSol():
	#
    # This class is used to keep a candidate solution.
    # It keeps the order of elements added to the solution.
    # auxiliary is used to store some extra information for the set in order to make
    # function evaluation and calculating the marginal values easier.
    #
	def __init__(self):
		self.S_list = []
		self.S_set = set()
		self.auxiliary = None

	def add_elements(self, e):
		if isinstance(e, int):
			if e not in self.S_set:
				self.S_set.add(e)
				self.S_list.append(e)
		else:
			set_e = set(e)
			for u in e:
				if u not in self.S_set:
					self.S_list.append(u)
			self.S_set.update(set_e)

	def remove_elements(self, e):
		if isinstance(e, int):
			if e in self.S_set:
				self.S_set.remove(e)
				self.S_list.remove(e)
		else:
			if len(e) > 0:
				set_e = set(e)
				self.S_set -= set_e
				current_S = copy.copy(self.S_list)
				self.S_list = [u for u in current_S if u not in set_e]

	def find_index(self, e):
		if e  not in self.S_set:
			return -1
		return self.S_list.index(e)



def get_yelp_data(yelp_info_file, lambdaa = 1.0, reg_coef = 1.0, max_datapoints = 1000, nb_sample = 100, application = 1):
	#
    # Load the yelp dataset
    # Our objective is to find a representative summary of the locations from the 
    # following cities: Charlotte, Edinburgh, Las Vegas,Madison, Phoenix, and Pittsburgh.
    #
    # For this experiment, we impose a combination of several constraints: 
    # (i) there is a limit m on the total size of summary, (ii) the maximum number of 
    # locations from each city is mi , and (iii) three knapsacks c1 , c2 , and c3 
    # where c_i(j) = distance(j, POI_i) is the distance of location j to a point of 
    # interest in the corresponding city of j. For POIs we consider down-town, an 
    # international airport and a national museum in each one of the six cities. 
    # One unit of budget is equivalent to 100km, which means the sum of distances of
	# every set of feasible locations to the point of interests (i.e., down-towns, 
	# airports or museums) is at most 100km if we set knapsack budget to one.
    #
	recommended_genres = ['Pittsburgh', 'Charlotte', 'Phoenix', 'Madison', 'Las Vegas', 
	'Edinburgh']
	print("lambdaaaa: ", lambdaa)

	distance = []
	state = []
	city = []
	address = []
	data_mat = []
	elements = []
	locations = []
	cnt = 0
	costs = defaultdict(list)
	with open(yelp_info_file, 'r') as yelp_info_csv:
		reader = csv.reader(yelp_info_csv)
		header = next(reader)
		for row in reader:
			if cnt < max_datapoints:
				city.append([row[0]])
				elements.append(cnt)
				distance.append([float(row[1]), float(row[2]), float(row[3])])
				costs[cnt] = [float(row[1]), float(row[2]), float(row[3])]
				state.append(row[7])
				address.append(row[8])
				vals = [float(val) for val in row[9:]]
				data_mat.append(vals)
				locations.append((float(row[4]), float(row[5])))
				cnt+=1


	data_mat = np.array(data_mat)
	M = sci.squareform(sci.pdist(data_mat, 'euclidean'))
	M = np.exp(-1 * lambdaa * M )
	nb_knapsacks = 0
	elements = elements[nb_sample:]
	M = M[:,:nb_sample]


	#
	# Normalizing the knsapsack costs according to the explanations from the paper
	#
	vals = [0,0,0]
	for e in elements:
		for i in range(3):
			vals[i] += costs[e][i]
	rate = len(elements) / 5
	if application == 1:
		for e in elements:
			cost = costs[e]
			costs[e] = [rate * cost[i] / vals[i] for i in range(1)]
			nb_knapsacks = 1
	if application == 4:
		for e in elements:
			cost = costs[e]
			costs[e] = [rate * cost[i] / vals[i] for i in range(1,2)]
			nb_knapsacks = 1
	if application == 2:
		for e in elements:
			cost = costs[e]
			costs[e] = [rate * cost[i] / vals[i] for i in range(2)]
			nb_knapsacks = 2
	if application == 3:
		for e in elements:
			cost = costs[e]
			costs[e] = [rate * cost[i] / vals[i] for i in range(3)]
			
		nb_knapsacks=3

	vals = [0,0,0]
	for e in elements:
		for i in range(3):
			vals[i] += costs[e][i]
	
	dataset ={}
	data_dict = {'matrix': data_mat, 'genres': city, 'titles': address, 
	 'recommended_genres':recommended_genres, 'nb_sample':nb_sample, 
	 'nb_elements': len(elements),
	'lambda': lambdaa,'sim_mat': M, 'nb_knapsacks':nb_knapsacks,
	'elements':elements, 'distance':distance, 'reg_coef': reg_coef, 'costs':costs}
	return data_dict

class FacilityLocation():
	#
    # The monotone and submodular set function Eq (10) used for Yelp location data summarization.
    #
	def __init__(self, dataset):
		self.dataset = dataset
		self.sim_mat = dataset['sim_mat']
		self.nb_sample = dataset['nb_sample']
		self.oracle_calls = 0

	def evaluate(self, S):
		self.oracle_calls += 1
		if isinstance(S, int):
			S = [S]

		if isinstance(S, set):
			S = list(S)

		if isinstance(S, list):
			S = list(set(S))

		if len(S) == 0:
			return 0, None
		#print(S, np.max(self.sim_mat[np.ix_(S, )], axis = 0))
		val  = np.sum(np.max(self.sim_mat[np.ix_(S, )], axis = 0))
		return val, None

	def marginal_gain(self, e, S, current_val = None, auxiliary = None):
		if isinstance(S, int):
			S = [S]
		if isinstance(S, set):
			S = list(S)
		if e in S:
			return 0.0, None
		if current_val == None:
			current_val, __ = self.evaluate(S)

		if isinstance(e, int):
			val, __ = self.evaluate(S + [e])
		else:
			val, __ = self.evaluate(S + e)
		return val - current_val, None


	def add_one_element(self, e, S, current_val = None, auxiliary = None):
		if isinstance(S, int):
			S = [S]
		if isinstance(e, list):
			return self.evaluate(S + e)
		else:
			return self.evaluate(S + [e])

class VertexCover():
	# 
	# We define a monotone and submodular function over vertices of a directed 
	# real-world graph G = (V,E). Let’s w : V → R≥0 denotes a weight function on the 
	# vertices of graph G. For a given vertex set S ⊆ V , assume N(S) is the set of 
	# vertices which are pointed to by S. We define f :2^V → R≥0 asfollows:
	# f(S)= 􏰊\sum{u∈N(S)∪S} w_u and we assign to each vertex u a weight of one. 
	#
	def __init__(self, dataset):
		self.dataset = dataset
		self.oracle_calls = 0
	def evaluate(self, S):
		if isinstance(S, list) or isinstance(S, set):
			nodes = set(S)
			for e in S:
				for nei in self.dataset['edges'][e]:
					nodes.add(nei)
		else:
			nodes = set()
			nodes.add(S)
			for nei in self.dataset['edges'][S]:
				nodes.add(nei)

		val = 0
		for e in nodes:
			val += self.dataset['vertex_weights'][e]
		self.oracle_calls += 1
		return val, nodes


	def marginal_gain(self, e, S, current_val = None, auxiliary = None):
		if current_val == None or auxiliary == None:
			current_val, auxiliary = self.evaluate(S)
		nodes = set()
		nodes.add(e)
		gain = 0
		new_auxiliary = set()
		if e not in auxiliary:
			gain += self.dataset['vertex_weights'][e]
			new_auxiliary.add(e)

		for nei in self.dataset['edges'][e]:
			if nei not in auxiliary:
				gain += self.dataset['vertex_weights'][nei]
				new_auxiliary.add(nei)
		self.oracle_calls += 1
		return gain, new_auxiliary


	def add_one_element(self, e, S, current_val = None, auxiliary = None):
		if e in S:
			return 0
		if current_val == None or auxiliary == None:
			current_val, auxiliary = self.evaluate(S)
		nodes = set()
		nodes.add(e)
		val = current_val
		if e not in auxiliary:
			val += self.dataset['vertex_weights'][e]
			auxiliary.add(e)

		for nei in self.dataset['edges'][e]:
			if nei not in auxiliary:
				val += self.dataset['vertex_weights'][nei]
				auxiliary.add(nei)
		self.oracle_calls += 1
		return val, auxiliary

class NodeCommunityLimit():
	def __init__(self, dataset, max_movies = None, max_each_genres = 5):
		self.genres = dataset['genres']
		self.costs = dataset['costs']
		self.nb_elements = len(dataset['elements'])
		self.recommended_genres = dataset['recommended_genres']
		self.max_movies = max_movies
		self.max_each_genres = max_each_genres
		self.nb_knapsacks = dataset['nb_knapsacks']
		self.dataset = dataset
		self.max_cardinality = self.max_each_genres * self.nb_knapsacks
		if max_movies is not None:
			self.max_cardinality = min(self.max_cardinality, self.max_movies)
		self.nb_matroids = len(dataset['recommended_genres'])
		if max_movies is not None:
			self.nb_matroids += 1
		self.k = max(self.nb_knapsacks, self.nb_matroids)
		if self.max_movies is None:
			self.max_movies = sys.maxsize

	def is_feasible(self, S):
		if len(S) == 0:
			return True
		if not isinstance(S, set):
			S = set(S)
		if len(S) > self.max_movies:
			return False

		genre_count = dict()
		for genre in self.recommended_genres:
			genre_count[genre] = 0

		for e in S:
			for genre in self.genres[e]:
				if genre_count[genre] >= self.max_each_genres:
					return False
				else:
					genre_count[genre] += 1

		S_costs = [self.costs[e] for e in S]
		if round(max([sum(x) for x in zip(*S_costs)]), 4) > 1.0:
			return False

		return True
	def is_add_feasible(self, e, S):
		if isinstance(S, int):
			S = [S]
		if not isinstance(e, list):
			return self.is_feasible(S + [e])
		else:
			return self.is_feasible(S + e)

	def is_psystem_add_feasible(self, e, S):
		if isinstance(S, int):
			S = [S]
		if isinstance(e, int):
			e = [e]
		S_e = S + e
		S = set(S_e)
		if len(S) > self.max_movies:
			return False
		genre_count = dict()
		for genre in self.recommended_genres:
			genre_count[genre] = 0
		for e in S:
			for genre in self.genres[e]:
				if genre_count[genre] >= self.max_each_genres:
					return False
				else:
					genre_count[genre] += 1
		return True

	def check_costs(self,S):
		if len(S) == 0:
			return True
		S_costs = [self.costs[e] for e in S]
		if round(max([sum(x) for x in zip(*S_costs)]),4) > 1:
			return False
		return True