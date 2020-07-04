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
from kneebow.rotor import Rotor #must download python package

filein = str(sys.argv[1])

def main():
	columns = ["title", "views"]
	df = pd.read_csv(filein, usecols = columns)
	titles  = df["title"]
	views = df["views"]
	titleLengths, lrange = strLengths(titles)
	titleLengths= pd.concat([titleLengths, views], axis = 1)
	#print(titleLengths)
	numCaps, crange = strNumCaps(titles)
	numCaps = pd.concat([numCaps, views], axis = 1)
	#print(numCaps)
	numPuncs, prange = strNumPuncs(titles)
	numPuncs= pd.concat([numPuncs, views], axis = 1)
	#print(numPuncs)
	dbscan(titleLengths, findeps(titleLengths), lrange)
	dbscan(numCaps, findeps(numCaps), crange)
	dbscan(numPuncs, findeps(numCaps), prange)

def strLengths(t):
	l = t.str.len()
	lr = l.max() - l.min()
	return l, lr

def strNumCaps(t):
	nc = t.str.count(r'[A-Z]')
	cr = nc.max() - nc.min()
	return nc, cr

def strNumPuncs(t):
	nee = t.str.count('!')
	neq = t.str.count('\?')
	ne = nee + neq
	pr = ne.max() - ne.min()
	return ne, pr

def findeps(data):
	d = StandardScaler().fit_transform(data)
	d = np.nan_to_num(d)
	neighbors = NearestNeighbors(n_neighbors=2).fit(d)
	distances, indices = neighbors.kneighbors(d)
	distances = np.sort(distances, axis=0)
	distances = distances[:,1]
	rotor = Rotor()
	rotor.fit_rotate(np.concatenate((indices[:,0].reshape(-1, 1), distances.reshape(-1, 1)), axis = 1))
	epsx = rotor.get_elbow_index()
	eps = distances[epsx]
	return eps #- (0.1 if (eps-0.1) >0 else 0)

def dbscan(data, e, drange):
	d = StandardScaler().fit_transform(data)
	d = np.nan_to_num(d)
	db = DBSCAN(eps = e, min_samples = int(np.sqrt(np.size(data)/(drange)))).fit(d)
	core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
	core_samples_mask[db.core_sample_indices_ ] = True
	clusters = db.labels_

	numclusters = len(set(clusters)) - (1 if -1 in clusters else 0)
	numnoise = list(clusters).count(-1)
	#print(numclusters)
	#print(numnoise)
	#print(clusters)

	colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive',
		'goldenrod', 'lightcyan','navy','beige', 'brown', 'chartreuse', 'coral', 'lavender', 'pink', 'silver']
	vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
	plt.scatter(d[:,0], d[:, 1], c = vectorizer(clusters))


	'''
	#version #2 (me fiddling around)
	unique_labels = set(clusters)
	colors = np.append(np.array(np.random.rand(3,)), 1)
	for c in range(np.size(unique_labels)):
		colors = np.vstack([colors, np.append(np.random.rand(3,), 1)]);
	
	for k, col in zip(unique_labels, colors):
		if k == -1:
			col = [0, 0, 0, 0.5]

		class_member_mask = (clusters == k)

		xy = d[class_member_mask & core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

		xy = d[class_member_mask & ~core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)'''

	'''
	#version #1 (example code)
	colors = [plt.cm.get_cmap("Spectral")
		for each in np.linspace(0, 1, len(unique_labels))]

	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        col = [0, 0, 0, 1]

	    class_member_mask = (clusters == k)

	    xy = d[class_member_mask & core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

	    xy = d[class_member_mask & ~core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)'''

	plt.title('Estimated number of clusters: %d' % numclusters)
	plt.show()

if __name__ == '__main__':
	main()
