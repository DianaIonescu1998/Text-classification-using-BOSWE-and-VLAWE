import numpy as np
from computeVLAWEForOneVocabularyFromDocument import *

def computeVLAWEForTwoVocabulariesFromDocumentList (vocabularyPos: Vocabulary, vocabularyNeg: Vocabulary, listOfDocumentsPath, numberOfCentroids):
	
	vlaweForDocuments=[]
	
	for path in listOfDocumentsPath:
		print("Compute VLAWE for document from two vocabularies", path)
		
		vlawePos=computeVLAWEForOneVocabularyFromDocument(vocabularyPos, path, numberOfCentroids)
		vlaweNeg=computeVLAWEForOneVocabularyFromDocument(vocabularyNeg, path, numberOfCentroids)
		#print(vlawePos.shape, vlaweNeg.shape)
		##get linearized descriptor for each text by concatenating the results obtained on negative and positive vocabulary  
		vlawe=np.concatenate((vlawePos, vlaweNeg), axis=0) 
		
		##use this to ensure that the vlawe representation has the correct shape
		if vocabularyPos.dbscan!=None or (np.array(vlawe).shape[0]==numberOfCentroids*2):
			vlaweForDocuments.append(vlawe)
		
	print("vlawe for documents shape", np.array(vlaweForDocuments).shape)
	return np.array(vlaweForDocuments)