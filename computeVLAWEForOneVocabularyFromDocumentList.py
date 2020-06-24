from computeVLAWEForOneVocabularyFromDocument import *

def computeVLAWEForOneVocabularyFromDocumentList (vocabulary: Vocabulary, listOfDocumentsPath, numberOfCentroids):
	
	vlaweForDocuments=[]
	
	for path in listOfDocumentsPath:
		print("Compute VLAWE for document ", path)
		vlawe=computeVLAWEForOneVocabularyFromDocument(vocabulary, path, numberOfCentroids)
		if (np.array(vlawe).shape[0]==numberOfCentroids):
			vlaweForDocuments.append(vlawe)
	
	return vlaweForDocuments