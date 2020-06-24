import numpy as np
from computeHistogramForOneVocabularyFromDocument import *
from Vocabulary import *

def computeHistogramsForTwoVocabulariesFromDocumentList (vocabularyPos: Vocabulary, vocabularyNeg: Vocabulary, listOfDocumentsPath, numberOfCentroids):
	
	histograms=[]
	
	for path in listOfDocumentsPath:
		print("Compute histogram for document from two vocabularies for", path)
		
		histogramPos=computeHistogramForOneVocabularyFromDocument(vocabularyPos, path, numberOfCentroids)
		histogramNeg=computeHistogramForOneVocabularyFromDocument(vocabularyNeg, path, numberOfCentroids)
		#print(histogramPos, histogramNeg)
		
		##get linearized descriptor for each text by concatenating the results obtained on negative and positive vocabulary  
		histogram=np.concatenate((histogramPos, histogramNeg), axis=0) 

		
		##use this to ensure that the histogram has the correct shape
		if vocabularyPos.dbscan!=None or (np.array(histogram).shape[0]==numberOfCentroids*2):
			histograms.append(histogram)
		
	#print("histogram for documents shape", np.array(histograms).shape, histograms)
	return np.array(histograms)