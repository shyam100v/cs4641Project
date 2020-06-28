import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

filein = str(sys.argv[1])

def main():
	columns = ["title", "views"]
	df = pd.read_csv(filein, usecols = columns)
	titles  = df["title"]
	views = df["views"]
	titleLengths = pd.concat([strLengths(titles), views], axis = 1)
	#print(titleLengths)
	numCaps = pd.concat([strNumCaps(titles), views], axis = 1)
	#print(numCaps)
	numPuncs = pd.concat([strNumPuncs(titles), views], axis = 1)
	#print(numPuncs)
	dbscan(titleLengths)
	dbscan(numCaps)
	dbscan(numPuncs)

def strLengths(t):
	l = t.str.len()
	return l

def strNumCaps(t):
	nc = t.str.count(r'[A-Z]')
	return nc

def strNumPuncs(t):
	ne = t.str.count('!')
	return ne

def dbscan(data):
	d = StandardScaler().fit_transform(data)
	d = np.nan_to_num(d)
	db = DBSCAN(eps = 0.3, min_samples = 10).fit(d)
	core_sampes_mask = np.zeros_like(db.labels_, dtype = bool)
	core_samples_mask[db.core_sample_indices_ ] = True
	labels = db.labels_

	numclusters = len(set(lables)) - (1 if -1 in labels else 0)
	numnoise = list(labels).count(-1)
	#print(numclusters)
	#print(numnoise)

	unique_lables = set(labels)
	colors = [plt.cm.Spectral(each)
		for each in np.linspace(0, 1, len(unique_labels))]

	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        col = [0, 0, 0, 1]

	    class_member_mask = (labels == k)

	    xy = d[class_member_mask & core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=14)

	    xy = d[class_member_mask & ~core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=6)

	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.show()

if __name__ == '__main__':
	main()