from computeHistogramForOneVocabularyFromDocument import *
from Vocabulary import *

def computeHistogramsForOneVocabularyFromDocumentList(vocabulary: Vocabulary, listOfDocumentsPath, numberOfCentroids):
	histograms=[]
	
	for path in listOfDocumentsPath:
		print("Compute histograms for document ", path)
		histogram=computeHistogramForOneVocabularyFromDocument(vocabulary, path, numberOfCentroids)
		histograms.append(histogram)
		
	print("Histograms shape", np.array(histograms).shape)
	return np.array(histograms)