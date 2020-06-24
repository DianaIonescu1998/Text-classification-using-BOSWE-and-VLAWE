import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans, DBSCAN
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pickle

from Vocabulary import *
from writeFile import *
from createVocabularyFromListOfPaths import *
from getTXTsPath import *
from makeLinearKernelBOSWE import *
from computeHistogramsForOneVocabularyFromDocumentList import *
from computeHistogramsForTwoVocabulariesFromDocumentList import *


def runBOSWE(vocabularyDBSCAN, documentsPathPos, documentsPathNeg, numberOfVocabularies, numberOfCentroids, createCentroids=False, cSVM=[0.1, 1, 10, 100], fileForCVS="boswe_output.txt", dbscan_eps_val=50, dbscan_min_samples_val=10):
	print("Run BOSWE...")
	
	print("Make vocabulary...")
	if numberOfVocabularies==1:
		
		vocabulary=Vocabulary()
		
		print("Create one vocabulary...")
		listOfDocumentsPath= getTXTsPath(documentsPathPos)
		positive=len(listOfDocumentsPath)
		listOfDocumentsPath+=getTXTsPath(documentsPathNeg)
		labels=np.zeros(len(listOfDocumentsPath))
		labels[0:positive]=1
		
		if vocabularyDBSCAN==True:
			print("dbscan vocabulary")
			
			vocabulary.dbscan= DBSCAN(eps=dbscan_eps_val, min_samples=dbscan_min_samples_val, algorithm= 'kd_tree')
			#createVocabularyFromListOfPaths( vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids=10, dbscan_eps=0.3, dbscan_min_samples=100)
			#(vocabulary.dbscan, vocabulary.dbscanClusters)=createVocabularyFromListOfPaths( vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids=10,dbscan_eps=50, dbscan_min_samples=10)
			vocabulary.dbscanClusters=createVocabularyFromListOfPaths(vocabulary, vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids=10, dbscan_eps=dbscan_eps_val, dbscan_min_samples=dbscan_min_samples_val)
			print(vocabulary.dbscan.labels_)
		else:
			if createCentroids==True:
				#To create centroids 
				(vocabulary.centers, vocabulary.kdtree) = createVocabularyFromListOfPaths(vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids)
				writeFile("boswe_centers_voc.txt",vocabulary.centers)
		
			else:
				#To load centroids
				documentPathCenters='C:/Users/Dell/Desktop/THISisTheCharmedOne/centers_voc.txt'
				vocabulary.centers=np.genfromtxt(documentPathCenters, delimiter=',')
				print("Centers: ",(vocabulary.centers).shape)
				vocabulary.kdtree= KDTree(vocabulary.centers)	
			
		#get histograms for each document
		print("Compute histograms...")
		histograms= computeHistogramsForOneVocabularyFromDocumentList(vocabulary, listOfDocumentsPath, numberOfCentroids)
		print("Histograms shape: ", np.array(histograms).shape)
		
		#make linear kernel 
		kernelRepresentation=makeLinearKernelBOSWE(histograms)
		
		scores=[]
		for cValue in cSVM:
			print("linearSVM", cValue)
			linearSVM= svm.SVC(C=cValue, kernel='precomputed')
			cvs= cross_val_score(linearSVM, kernelRepresentation, labels, cv=10)
			scores.append(cvs)
			print("Cross valid for 10 experiments: ", cvs)
		writeFile(fileForCVS, np.array(scores))
		print("The file", fileForCVS, " was created")
		
		#print(predictions)
		#filename = 'finalized_model.sav'
		#pickle.dump(linearSVM, open(filename, 'wb'))
		
		
		return scores
	
		
		
	elif numberOfVocabularies==2:
		
		vocabularyPos=Vocabulary()
		vocabularyNeg=Vocabulary()
		
		
		print("Create two vocabularies...")
		listOfDocumentsPathPos= getTXTsPath(documentsPathPos)
		listOfDocumentsPathNeg=getTXTsPath(documentsPathNeg)
		labelsPos=np.ones(len(listOfDocumentsPathPos))
		labelsNeg=np.zeros(len(listOfDocumentsPathNeg))
		listOfDocumentsPath=np.concatenate((listOfDocumentsPathPos,listOfDocumentsPathNeg), axis=0)
		labels=np.concatenate((labelsPos, labelsNeg), axis=0)
		
		if vocabularyDBSCAN==True:
			print("dbscan vocabulary")
			#doar pentru pozitive!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			###vocabularyNeg.dbscan= DBSCAN(eps=dbscan_eps_val, min_samples=dbscan_min_samples_val, algorithm= 'kd_tree', leaf_size=1000, n_jobs=-1)
			#createVocabularyFromListOfPaths( vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids=10, dbscan_eps=0.3, dbscan_min_samples=100)
			#(vocabulary.dbscan, vocabulary.dbscanClusters)=createVocabularyFromListOfPaths( vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids=10,dbscan_eps=50, dbscan_min_samples=10)
			vocabularyPos.dbscan= DBSCAN(eps=dbscan_eps_val, min_samples=dbscan_min_samples_val, algorithm= 'kd_tree', leaf_size=1000, n_jobs=-1)
			(vocabularyPos.dbscanClusters, vocabularyPos.dbscanLabels, vocabularyPos.dbscanListOfKDTrees, vocabularyPos.dbscanMeanCluster)=createVocabularyFromListOfPaths(vocabularyPos, vocabularyDBSCAN, listOfDocumentsPathPos, numberOfCentroids=10, dbscan_eps=dbscan_eps_val, dbscan_min_samples=dbscan_min_samples_val, file="positive_embeddings_from_documents.txt")
			vocabularyPos.eps=dbscan_eps_val
			#print("Vocabulary positive", vocabularyPos.dbscanClusters, vocabularyPos.dbscanLabels, vocabularyPos.dbscanListOfKDTrees, vocabularyPos.dbscanMeanCluster)
			#print("Check if true:", len(vocabularyPos.dbscanLabels), len(vocabularyPos.dbscan.labels_))
			###print(vocabularyNeg.dbscan.labels_)
			
			vocabularyNeg.dbscan= DBSCAN(eps=dbscan_eps_val, min_samples=dbscan_min_samples_val, algorithm= 'kd_tree', leaf_size=1000, n_jobs=-1)
			(vocabularyNeg.dbscanClusters, vocabularyNeg.dbscanLabels, vocabularyNeg.dbscanListOfKDTrees, vocabularyNeg.dbscanMeanCluster)=createVocabularyFromListOfPaths(vocabularyNeg, vocabularyDBSCAN, listOfDocumentsPathNeg, numberOfCentroids=10, dbscan_eps=dbscan_eps_val, dbscan_min_samples=dbscan_min_samples_val, file="negative_embeddings_from_documents.txt")
			vocabularyNeg.eps=dbscan_eps_val
			#print("Vocabulary negative", vocabularyNeg.dbscanClusters, vocabularyNeg.dbscanLabels, vocabularyNeg.dbscanListOfKDTrees, vocabularyNeg.dbscanMeanCluster)
			#print("Check if true:", len(vocabularyNeg.dbscanLabels), len(vocabularyNeg.dbscan.labels_))
			

		elif createCentroids==True and vocabularyDBSCAN==False:
			#create and save 2 vocabularies
			(vocabularyPos.centers, vocabularyPos.kdtree) = createVocabularyFromListOfPaths(vocabularyDBSCAN,listOfDocumentsPathPos, numberOfCentroids)
			writeFile("boswe_centers_voc_pos.txt",vocabularyPos.centers)
			(vocabularyNeg.centers, vocabularyNeg.kdtree) = createVocabularyFromListOfPaths(vocabularyDBSCAN, listOfDocumentsPathNeg, numberOfCentroids)
			writeFile("boswe_centers_voc_neg.txt",vocabularyNeg.centers)
			
		elif createCentroids==False and vocabularyDBSCAN==False:
			
			#load negative centroids
			documentPathCenters='C:/Users/Dell/Desktop/THISisTheCharmedOne/boswe_centers_voc_neg.txt'
			vocabularyNeg.centers=np.genfromtxt(documentPathCenters, delimiter=',')
			print("Centers for negative vocabulary: ",(vocabularyNeg.centers).shape)
			vocabularyNeg.kdtree= KDTree(vocabularyNeg.centers)
			
			#load positive centroids
			documentPathCenters='C:/Users/Dell/Desktop/THISisTheCharmedOne/boswe_centers_voc_pos.txt'
			vocabularyPos.centers=np.genfromtxt(documentPathCenters, delimiter=',')
			print("Centers for positive vocabulary: ",(vocabularyPos.centers).shape)
			vocabularyPos.kdtree= KDTree(vocabularyPos.centers)
		
		
		#get histograms for each document
		print("Compute histograms...")
		histograms= computeHistogramsForTwoVocabulariesFromDocumentList(vocabularyPos, vocabularyNeg, listOfDocumentsPath, numberOfCentroids)
		print("Histograms shape: ", np.array(histograms).shape)
		
		#make linear kernel 
		kernelRepresentation=makeLinearKernelBOSWE(histograms)
		print("Kernel shape", kernelRepresentation.shape)
		
		scores=[]
		for cValue in cSVM:
			print("linearSVM", cValue)
			linearSVM= svm.SVC(C=cValue, kernel='precomputed')
			cvs= cross_val_score(linearSVM, kernelRepresentation, labels, cv=10)
			scores.append(cvs)
			print("Cross valid for 10 experiments: ", cvs)
		writeFile(fileForCVS, np.array(scores))
		print("The file", fileForCVS, " was created")
		
		#print(predictions)
		#filename = 'finalized_model.sav'
		#pickle.dump(linearSVM, open(filename, 'wb'))
		
		
		return scores
		
	
		



#path of files that contain word embeddings
documentsPathPos="C:/Users/Dell/Desktop/THISisTheCharmedOne/pos/"
documentsPathNeg="C:/Users/Dell/Desktop/THISisTheCharmedOne/neg/"

numberOfVocabularies=1
numberOfCentroids=500
vocabularyDBSCAN=True

cvs=runBOSWE(vocabularyDBSCAN, documentsPathPos, documentsPathNeg, numberOfVocabularies=2, numberOfCentroids=500, createCentroids=True, cSVM=[0.1, 1, 10, 100], fileForCVS="dbscan_eps_80_dbscan_min_samples_90"+"centroids_KMEANS_boswe.txt",  dbscan_eps_val=50, dbscan_min_samples_val=70)
for c in cvs:
	print(cvs, "acuratetea pentru cross validation")



'''
			Rezultate
			
		numberOfVocabularies=2
		numberOfCentroids=10 => histograms =20
	C=1

'''