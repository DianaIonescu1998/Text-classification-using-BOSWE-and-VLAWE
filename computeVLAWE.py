import numpy as np

def computeVLAWE(nearestNeighbours, centroids, descriptors, numberOfCentroids):
	#vlad ajusted for word embeddings
	vlawe= np.zeros((numberOfCentroids, len(descriptors[0])), dtype='float64')
	for i, nn in enumerate(nearestNeighbours):
		if nn!= -1: #eliminate the noise from the VLAWE representation
			vlawe[nn]+=np.subtract( np.array(descriptors[i]), np.array(centroids[nn]), dtype='float64')
	
	return vlawe