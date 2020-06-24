import numpy as np
from quantizeDescriptors import *
from computeFeatures import *
from Vocabulary import *

def computeHistogramForOneVocabularyFromDocument(vocabulary: Vocabulary, documentPath, numberOfCentroids):
	
	try:
		embeddings=np.genfromtxt(documentPath, delimiter=',')
		if(len(embeddings.shape)==1):
			embeddings=np.array([embeddings])
	except IOError:
		return np.array([]) #if the documentPath was not found
	
	if embeddings.shape[0] == 0:
		return np.array([]) #if the file is empty
	else:
		if vocabulary.dbscan!=None:
			#print("computeHistogramForOneVocabularyFromDocument")
			keypoints, descriptors = computeFeatures(embeddings)
			nn= quantizeDescriptors(vocabulary, descriptors)
			histogram=np.zeros(vocabulary.dbscanClusters)
			for centroid in nn:
				if centroid!=-1:
					histogram[centroid]+=1
			#print("histogram", histogram)
			return histogram
		else:
			keypoints, descriptors = computeFeatures(embeddings)
			nn= quantizeDescriptors(vocabulary, descriptors)
			histogram=np.zeros(numberOfCentroids)
			for centroid in nn:
				histogram[centroid]+=1
			#print("histogram shape", histogram.shape)
			#print("histogram", histogram)
			return histogram