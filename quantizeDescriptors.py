from Vocabulary import *
import numpy as np
from dbscanKdtrees import *
#returns a list of positions of centers that are the closest to the descriptors

def quantizeDescriptors (vocabulary:Vocabulary, descriptors):
	#if the vocabulary is created using dbscan
	if vocabulary.dbscan!=None:
		nearestNeighbours=[]
		#predict each feature:
		for embs in descriptors:
			nn=dbscanGetLabel(embs, vocabulary.dbscanListOfKDTrees, vocabulary.eps)
			#print(nn, type(nn))
			nearestNeighbours.append(nn)
		#print("Quantize Descriptors:", nearestNeighbours)
		
	#if the vocabulary is created using k-means
	else:
		#!descriptors must be a list
		nearestNeighbours = vocabulary.kdtree.query( descriptors , k=1, return_distance=False)
		# nearestNeighbours is a list of number_of_words lists that have 1 element
		#Example output: [[1] [2]]
		nearestNeighbours= np.concatenate(np.array(nearestNeighbours)) #converting the data into one list of integers
	
	return nearestNeighbours