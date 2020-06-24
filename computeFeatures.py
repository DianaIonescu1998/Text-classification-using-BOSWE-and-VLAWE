import numpy as np


def computeFeatures(documentEmbeddings):
	if len(np.array(documentEmbeddings).shape) == 1 :
		documentEmbeddings=np.array([documentEmbeddings])
		
	(words, features)=np.array(documentEmbeddings).shape
	
	keypoints=np.ones((words, 2))
	descriptors= np.array(documentEmbeddings).astype('float32')
	
	return keypoints, descriptors
	