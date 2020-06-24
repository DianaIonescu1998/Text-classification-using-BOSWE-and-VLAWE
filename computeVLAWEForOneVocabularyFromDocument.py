import numpy as np
from quantizeDescriptors import *
from computeFeatures import *
from computeVLAWE import *

def computeVLAWEForOneVocabularyFromDocument(vocabulary: Vocabulary, documentPath, numberOfCentroids):
	descriptors=[]
	vlawe=[]
	nn=[]
	try:
		embeddings=np.genfromtxt(documentPath, delimiter=',')
	except IOError:
		return np.array(vlawe) #if the documentPath was not found
	
	if embeddings.shape[0] == 0:
		return np.array(vlawe) #if the file is empty
	else:
		_, descriptors = computeFeatures(embeddings)
		nn= quantizeDescriptors(vocabulary, descriptors)
		
		if vocabulary.dbscan!=None:
			print("in computeVLAWEForOneVocabularyFromDocument => Vocabulary DBSCAN")
			#vocabulary.dbscanMeanCluster contains a list with the mean for each cluster; this value will be used as centroid for VLAWE representation
			#vocabulary.dbscanClusters number of clusters created by DBSCAN
			vlawe=computeVLAWE(nn, vocabulary.dbscanMeanCluster, descriptors, vocabulary.dbscanClusters) 
		else:
			vlawe=computeVLAWE(nn, vocabulary.centers, descriptors, numberOfCentroids) 
		return vlawe