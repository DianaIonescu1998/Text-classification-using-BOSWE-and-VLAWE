import numpy as np
from sklearn.decomposition import PCA
#for normalization, use: https://scikit-learn.org/stable/modules/preprocessing.html#normalization

from sklearn.preprocessing import normalize

def makeLinearKernelBOSWE (ListOfHistograms):
	
	print('>Make linear kernel') 
	ListOfHistograms = np.array(ListOfHistograms)
	
	print("Shape of the list of documents", np.array(ListOfHistograms).shape)
	
	##apply normalization steps for the BOSWE representations of the documents
	
	#apply L2 normalization on each element
	ListOfHistograms=normalize(ListOfHistograms, norm='l2')
	
	print('ListOfDocumentsDescriptors', ListOfHistograms.shape)
	
	kernel=np.matmul(ListOfHistograms, ListOfHistograms.transpose(), dtype='float32')
	print("Kernel shape ", kernel.shape)
	
	return kernel
	