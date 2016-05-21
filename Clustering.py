import sys
import random
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#reads file give filepath and converts strings to int
def readFile(filePath):
	values = []
	f = open(filePath, "r")
	
	for line in f:
		s = line.split()
		values.append([float(s[0]),float(s[1])])

	f.close()
	return values

def main():
	
	if len(sys.argv) != 3:
		print("Usage: python Clustering.py <NUMBER_OF_CLUSTERS> <DATA_FILE.txt>")
		return 0

	k = int(sys.argv[1])
	filePath = sys.argv[2]
	colors = []

	for i in range(k):
		colors.append((random.random(),random.random(),random.random()))

	X = readFile(filePath)
	#create KMeans model with k clusters
	knn = KMeans(k)
	knn.fit(X)

	#save predictions
	lables = knn.predict(X)
	
	i = 0
	xs = []
	ys = []
	clusters = [[] for i in range(k)]
	for pair in X:
		xs.append(pair[0])
		ys.append(pair[1])
		clusters[lables[i]].append(pair)
		i += 1  
	
	x_min = min(xs)
	x_max = max(xs)
	y_min = min(ys)
	y_max = max(xs)

	#create graph
	plt.figure(1)
	plt.clf()

	#plot each cluster seperatlly 
	i = 0
	for cluster in clusters:
		xs = [pair[0] for pair in cluster]
		ys = [pair[1] for pair in cluster]
		plt.scatter(xs, ys, color = colors[i])
		i += 1

	#find and plot centers for each cluster
	centers = knn.cluster_centers_
	for aa in centers:
		plt.scatter(aa[0], aa[1], marker = "*", s = 16, linewidths = 3, color = "k")

	plt.title("K-means")
	plt.xlim(x_min*6/5,x_max*6/5)    
	plt.ylim(y_min*6/5,y_max*6/5)
	plt.show()

main()