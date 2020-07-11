import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from kneebow.rotor import Rotor #must download python package to use findeps()

filein = str(sys.argv[1])

def main():
	columns = ["videoTitle", "videoViews"]
	df = pd.read_csv(filein, usecols = columns)
	titles  = df["videoTitle"]
	views = df["videoViews"]
	titleLengths = strLengths(titles)
	titleLengths= pd.concat([titleLengths, views], axis = 1)
	numCaps = strNumCaps(titles)
	numCaps = pd.concat([numCaps, views], axis = 1)
	numPuncs = strNumPuncs(titles)
	numPuncs= pd.concat([numPuncs, views], axis = 1)
	#findeps(titleLengths)
	#findeps(numCaps)
	#findeps(numPuncs)
	dbscan(titleLengths, 0.4, "Length of Title (#characters)", "Number of Views")
	dbscan(numCaps, 0.515, "Number of Capital Letters in Title", "Number of Views")
	dbscan(numPuncs,0.5, "Number of Punctuation Marks in Title", "Number of Views")

def strLengths(t):
	l = t.str.len()
	lr = l.max() - l.min()
	return l

def strNumCaps(t):
	nc = t.str.count(r'[A-Z]')
	cr = nc.max() - nc.min()
	return nc

def strNumPuncs(t):
	nee = t.str.count('!')
	neq = t.str.count('\?')
	ne = nee + neq
	pr = ne.max() - ne.min()
	return ne

def findeps(data):
	d = StandardScaler().fit_transform(data)
	d = np.nan_to_num(data)
	neighbors = NearestNeighbors(n_neighbors=2).fit(d)
	distances, indices = neighbors.kneighbors(d)
	distances = np.sort(distances, axis=0)
	distances = distances[:,1]
	plt.plot(distances)
	plt.show()
	rotor = Rotor()
	rotor.fit_rotate(np.concatenate((indices[:,0].reshape(-1, 1), distances.reshape(-1, 1)), axis = 1))
	epsx = rotor.get_elbow_index()
	eps = distances[epsx]
	return eps

def dbscan(data, e, xaxis, yaxis):
	d = StandardScaler().fit_transform(data)
	d = np.nan_to_num(d)
	data = np.nan_to_num(data.to_numpy())
	db = DBSCAN(eps = e, min_samples = 4).fit(d)
	core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
	core_samples_mask[db.core_sample_indices_ ] = True
	clusters = db.labels_

	numclusters = len(set(clusters)) - (1 if -1 in clusters else 0)
	numnoise = list(clusters).count(-1)
	fig, ax = plt.subplots()
	ax.set_axisbelow(True)
	ax.grid()
	plt.scatter(data[:,0], data[:, 1], c = clusters)
	plt.title('Estimated number of clusters: %d' % numclusters)
	plt.xlabel(xaxis)
	plt.ylabel(yaxis)
	plt.show()

if __name__ == '__main__':
	main()
