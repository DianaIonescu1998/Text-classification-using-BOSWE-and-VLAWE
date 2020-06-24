import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def makeLinearKernelVLAWEusingPCA (ListOfDocumentsDescriptors):
	ListOfDocumentsDescriptors = np.array(ListOfDocumentsDescriptors, dtype='float32')
	
	##apply normalization steps for the VLAWE representations of the documents
	
	#Power Normalization: f(z)=sign(z) * |z|^alpha; alpha=0.5
	ListOfDocumentsDescriptors = np.sign(ListOfDocumentsDescriptors)* np.sqrt(np.abs(ListOfDocumentsDescriptors))
	
	#Linearize descriptors
	liniarizedDescriptors=[np.concatenate(descriptor, axis=0) for descriptor in ListOfDocumentsDescriptors ] 
	print("liniarizedDescriptors shape ", np.array(liniarizedDescriptors).shape) 
	
	##Apply PCA to reduce the dimensions of the data
	pca=PCA(n_components=768)
	listForTrainingPca=[]
	valueToEliminate=int(0.2*len(liniarizedDescriptors))
	listForTrainingPca=liniarizedDescriptors[:int(len(ListOfDocumentsDescriptors)/2)-valueToEliminate]
		+liniarizedDescriptors[int(len(ListOfDocumentsDescriptors)/2):len(ListOfDocumentsDescriptors)-valueToEliminate]
	listForTrainingPca=np.array(listForTrainingPca)
	#fit PCA model using only a part from the data
	pca.fit(listForTrainingPca)
	#reduce dimensions
	liniarizedDescriptors=pca.transform(liniarizedDescriptors)
	
	#apply L2 normalization on each element
	liniarizedDescriptors=normalize(liniarizedDescriptors, norm='l2')
	
	kernel=np.matmul(liniarizedDescriptors, liniarizedDescriptors.transpose(), dtype='float32')
	
	return kernel
	
	#return liniarizedDescriptors
	
	
	
