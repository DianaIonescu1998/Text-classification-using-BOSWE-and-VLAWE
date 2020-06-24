import numpy as np
from sklearn.decomposition import PCA
#for normalization, use: https://scikit-learn.org/stable/modules/preprocessing.html#normalization

from sklearn.preprocessing import normalize

def makeLinearKernelVLAWE (ListOfDocumentsDescriptors):
	
	print('>Make linear kernel') #using Zipf's law for alpha=0.5 and L2 normalization calculated on columns
	ListOfDocumentsDescriptors = np.array(ListOfDocumentsDescriptors, dtype='float32')
	
	print("Shape of the list of documents", np.array(ListOfDocumentsDescriptors).shape)
	
	##apply normalization steps for the VLAWE representations of the documents
	
	#Power Normalization: f(z)=sign(z) * |z|^alpha; alpha=0.5
	ListOfDocumentsDescriptors = np.sign(ListOfDocumentsDescriptors)* np.sqrt(np.abs(ListOfDocumentsDescriptors))
	
	#Linearize descriptors
	liniarizedDescriptors=[np.concatenate(descriptor, axis=0) for descriptor in ListOfDocumentsDescriptors ] 
	print("liniarizedDescriptors shape ", np.array(liniarizedDescriptors).shape) 
	
	#apply L2 normalization on each element
	liniarizedDescriptors=normalize(liniarizedDescriptors, norm='l2')
	
	print('ListOfDocumentsDescriptors', liniarizedDescriptors.shape)
	
	kernel=np.matmul(liniarizedDescriptors, liniarizedDescriptors.transpose(), dtype='float32')
	print("Kernel shape ", kernel.shape)
	
	return kernel
	
	#return liniarizedDescriptors
	
	
	##Apply PCA to reduce the dimensions of the data
	'''
	pca=PCA(n_components=ListOfDocumentsDescriptors.shape[2])
	listOfConcatenatedValues=[np.concatenate(descriptor, axis=0) for descriptor in ListOfDocumentsDescriptors ]
	ListOfDocumentsDescriptors=listOfConcatenatedValues
	listForTrainingPca=[]
	valueToEliminate=int(0.2*len(ListOfDocumentsDescriptors))
	listForTrainingPca=ListOfDocumentsDescriptors[:int(len(ListOfDocumentsDescriptors)/2)-valueToEliminate]+ListOfDocumentsDescriptors[int(len(ListOfDocumentsDescriptors)/2):len(ListOfDocumentsDescriptors)-valueToEliminate]
	listForTrainingPca=np.array(listForTrainingPca)
	print("listForTrainingPca", listForTrainingPca.shape)
	pca.fit(listForTrainingPca)
	
	ListOfDocumentsDescriptors=pca.transform(ListOfDocumentsDescriptors)
	
	print("afterPCA" , ListOfDocumentsDescriptors.shape)
	'''
	#descriptors=[]
	
	
	'''
	ListOfDocumentsDescriptors are numdoc x trasaturi_liniarizate
	K = listT* List => numdoc*numdoc
	
	matricea kernel are dimensiunea 
	'''
	
	'''
	for i, docDesc in enumerate(ListOfDocumentsDescriptors):
		print(docDesc.shape)
		std_dev= np.maximum(10**(-10), np.sqrt( np.sum(np.power(docDesc,2), axis=0)))
		#ListOfDocumentsDescriptors[i]=normalize(docDesc, norm='l2', axis=0, copy=False)
		#bxfun implementation:
		descriptors.append(docDesc * 1/np.maximum(10**(-10), np.sqrt( np.sum(np.power(docDesc,2), axis=0))) )
		histograms.append(np.matmul(ListOfDocumentsDescriptors[i].transpose(), ListOfDocumentsDescriptors[i]))
		#print(np.matmul(std_dev.transpose(), std_dev).shape)
		#histograms.append(std_dev)
		
	print("histograms - shape", np.array(histograms).shape)
	descriptors=np.array(descriptors)
	'''
	
	#print("makeLinearKernelVLAWE: ", "shape ",descriptors.shape) 
	