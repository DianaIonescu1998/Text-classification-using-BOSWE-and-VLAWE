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
from computeVLAWEForOneVocabularyFromDocumentList import *
from makeLinearKernelVLAWE import *
from saveLoadVLAWE import *
from makeLinearKernelVLAWEusingPCA import *
from computeVLAWEForTwoVocabulariesFromDocumentList import *



def runBOSWE_VLAWE(runBOSWE=False, runVLAWE=False, vocabularyDBSCAN=False, documentsPathPos=None, documentsPathNeg=None, numberOfVocabularies=1, numberOfCentroids=10, createCentroids=False, cSVM=[0.1, 1, 10, 100], fileForCVS="output.txt", dbscan_eps_val=50, dbscan_min_samples_val=10):
	print("Run BOSWE and VLAWE...")
	
	print("Make vocabulary...")
	if numberOfVocabularies==1:
		
		vocabulary=Vocabulary()
		scoresBOSWE=[]
		scoresVLAWE=[]
		fileBOSWE="BOSWE"+str(numberOfVocabularies)+"centroids"+str(numberOfCentroids)
		fileVLAWE="VLAWE"+str(numberOfVocabularies)+"centroids"+str(numberOfCentroids)
		
		print("Create one vocabulary...")
		listOfDocumentsPath= getTXTsPath(documentsPathPos)
		positive=len(listOfDocumentsPath)
		listOfDocumentsPath+=getTXTsPath(documentsPathNeg)
		labels=np.zeros(len(listOfDocumentsPath))
		labels[0:positive]=1
		
		if vocabularyDBSCAN==True:
		
			print("dbscan vocabulary")
			vocabulary.dbscan= DBSCAN(eps=dbscan_eps_val, min_samples=dbscan_min_samples_val, algorithm= 'kd_tree', leaf_size=1000, n_jobs=-1)
			(vocabulary.dbscanClusters, vocabulary.dbscanLabels, vocabulary.dbscanListOfKDTrees, vocabulary.dbscanMeanCluster)=createVocabularyFromListOfPaths(vocabulary, vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids=10, dbscan_eps=dbscan_eps_val, dbscan_min_samples=dbscan_min_samples_val, file="positive_negative.txt")
			vocabulary.eps=dbscan_eps_val
			vocabulary.minSamples= dbscan_min_samples_val
			
			fileBOSWE="dbscanBOSWE_2V_Eps"+str(vocabularyPos.eps)+"_MinPts_"+str(vocabularyPos.minSamples)+".txt"
			fileVLAWE="dbscanVLAWE_2V_Eps"+str(vocabularyPos.eps)+"_MinPts_"+str(vocabularyPos.minSamples)+".txt"
			
			
		else:
			if createCentroids==True:
				#To create centroids 
				(vocabulary.centers, vocabulary.kdtree) = createVocabularyFromListOfPaths(vocabulary, vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids=numberOfCentroids, dbscan_eps=70, dbscan_min_samples=10, file="none.txt")
				writeFile("boswe_centers_voc.txt",vocabulary.centers)
		
			else:
				#To load centroids
				documentPathCenters='C:/Users/Dell/Desktop/THISisTheCharmedOne/centers_voc.txt'
				vocabulary.centers=np.genfromtxt(documentPathCenters, delimiter=',')
				print("Centers: ",(vocabulary.centers).shape)
				vocabulary.kdtree= KDTree(vocabulary.centers)	
		
		################### BOSWE #######################
		if runBOSWE==True:
			#get histograms for each document
			print("Compute histograms...")
			histograms= computeHistogramsForOneVocabularyFromDocumentList(vocabulary, listOfDocumentsPath, numberOfCentroids)
			print("Histograms shape: ", np.array(histograms).shape)
		
			#make linear kernel 
			kernelRepresentation=makeLinearKernelBOSWE(histograms)
			print("Kernel shape", kernelRepresentation.shape)
		
			#evaluate the model
			for cValue in cSVM:
				print("linearSVM", cValue)
				linearSVM= svm.SVC(C=cValue, kernel='precomputed')
				cvs= cross_val_score(linearSVM, kernelRepresentation, labels, cv=10)
				scoresBOSWE.append(cvs)
				print("Cross valid for 10 experiments: ", cvs)
			writeFile(fileBOSWE, np.array(scoresBOSWE))
			print("The file", fileBOSWE, " was created")
		
		
		################### VLAWE ###########################
		if runVLAWE==True:
			print("Compute VLAWE embeddings")
			vlaweDocuments=computeVLAWEForOneVocabularyFromDocumentList(vocabulary, listOfDocumentsPath, numberOfCentroids)
			print("Vlawe descriptors", np.array(vlaweDocuments).shape)
		
			kernelRepresentation= makeLinearKernelVLAWEusingPCA(vlaweDocuments)
			print("kernel shape", kernelRepresentation.shape)
		
		
			for cValue in cSVM: 
				print("linearSVM", cValue)
				linearSVM= svm.SVC(C=cValue, kernel='precomputed')
				cvs= cross_val_score(linearSVM, kernelRepresentation, labels, cv=10)
				scoresVLAWE.append(cvs)
				print("Cross valid for 10 experiments: ", cvs)
			#save the results 
			writeFile(fileVLAWE, np.array(scoresVLAWE))
			print("The file", fileVLAWE, " was created")
		
		return scoresBOSWE, scoresVLAWE
		
		
		
	elif numberOfVocabularies==2:
		
		vocabularyPos=Vocabulary()
		vocabularyNeg=Vocabulary()
		fileBOSWE="BOSWE"+str(numberOfVocabularies)+"centroids"+str(numberOfCentroids)
		fileVLAWE="VLAWE"+str(numberOfVocabularies)+"centroids"+str(numberOfCentroids)
		scoresBOSWE=[]
		scoresVLAWE=[]
		
		print("Create two vocabularies...")
		listOfDocumentsPathPos= getTXTsPath(documentsPathPos)
		listOfDocumentsPathNeg=getTXTsPath(documentsPathNeg)
		labelsPos=np.ones(len(listOfDocumentsPathPos))
		labelsNeg=np.zeros(len(listOfDocumentsPathNeg))
		listOfDocumentsPath=np.concatenate((listOfDocumentsPathPos,listOfDocumentsPathNeg), axis=0)
		labels=np.concatenate((labelsPos, labelsNeg), axis=0)
		
		############create vocabulary using DBSCAN clustering
		if vocabularyDBSCAN==True:
			print("dbscan vocabulary")
			
			vocabularyPos.dbscan=  DBSCAN(eps=dbscan_eps_val, min_samples=dbscan_min_samples_val, algorithm= 'kd_tree', leaf_size=10000, n_jobs=-1)
			
			(vocabularyPos.dbscanClusters, vocabularyPos.dbscanLabels, vocabularyPos.dbscanListOfKDTrees, vocabularyPos.dbscanMeanCluster)=createVocabularyFromListOfPaths(vocabularyPos, vocabularyDBSCAN, listOfDocumentsPathPos, numberOfCentroids=10, dbscan_eps=dbscan_eps_val, dbscan_min_samples=dbscan_min_samples_val, file="positive_embeddings_from_documents.txt")
			vocabularyPos.eps=dbscan_eps_val
			vocabularyPos.minSamples= dbscan_min_samples_val
			
			vocabularyNeg.dbscan= DBSCAN(eps=dbscan_eps_val, min_samples=dbscan_min_samples_val, algorithm= 'kd_tree', leaf_size=10000, n_jobs=-1)
			(vocabularyNeg.dbscanClusters, vocabularyNeg.dbscanLabels, vocabularyNeg.dbscanListOfKDTrees, vocabularyNeg.dbscanMeanCluster)=createVocabularyFromListOfPaths(vocabularyNeg, vocabularyDBSCAN, listOfDocumentsPathNeg, numberOfCentroids=10, dbscan_eps=dbscan_eps_val, dbscan_min_samples=dbscan_min_samples_val, file="negative_embeddings_from_documents.txt")
			vocabularyNeg.eps=dbscan_eps_val
			vocabularyNeg.minSamples= dbscan_min_samples_val
			print("Eps and MinPts for pozitive ", vocabularyPos.eps, vocabularyPos.minSamples," and negative vocabulary ", vocabularyNeg.eps, vocabularyNeg.minSamples)
			
			fileBOSWE="dbscanBOSWE_2V_Eps"+str(vocabularyPos.eps)+"_MinPts_"+str(vocabularyPos.minSamples)+".txt"
			fileVLAWE="dbscanVLAWE_2V_Eps"+str(vocabularyPos.eps)+"_MinPts_"+str(vocabularyPos.minSamples)+".txt"
			
		############create vocabulary using KMeans clustering
		elif createCentroids==True and vocabularyDBSCAN==False:
			#create and save 2 vocabularies
			(vocabularyPos.centers, vocabularyPos.kdtree) = createVocabularyFromListOfPaths(vocabularyPos, vocabularyDBSCAN,listOfDocumentsPathPos, numberOfCentroids, file="positive_negative.txt")
			writeFile("boswe_centers_voc_pos.txt",vocabularyPos.centers)
			(vocabularyNeg.centers, vocabularyNeg.kdtree) = createVocabularyFromListOfPaths(vocabularyNeg, vocabularyDBSCAN, listOfDocumentsPathNeg, numberOfCentroids, file="positive_negative.txt")
			writeFile("boswe_centers_voc_neg.txt",vocabularyNeg.centers)
		
		############load vocabulary using KMeans clustering
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
		
		
		
		################### BOSWE #######################
		if runBOSWE==True:
			#get histograms for each document
			print("Compute histograms...")
			histograms= computeHistogramsForTwoVocabulariesFromDocumentList(vocabularyPos, vocabularyNeg, listOfDocumentsPath, numberOfCentroids)
			print("Histograms shape: ", np.array(histograms).shape)
		
			#make linear kernel 
			kernelRepresentation=makeLinearKernelBOSWE(histograms)
			print("Kernel shape", kernelRepresentation.shape)
		
			#evaluate the model
			for cValue in cSVM:
				print("linearSVM", cValue)
				linearSVM= svm.SVC(C=cValue, kernel='precomputed')
				cvs= cross_val_score(linearSVM, kernelRepresentation, labels, cv=10)
				scoresBOSWE.append(cvs)
				print("Cross valid for 10 experiments: ", cvs)
			#save the results
			writeFile(fileBOSWE, np.array(scoresBOSWE))
			print("The file", fileBOSWE, " was created")
		
		
		################### VLAWE ###########################
		if runVLAWE==True:
			print("Compute VLAWE embeddings")
			vlaweDocuments=computeVLAWEForTwoVocabulariesFromDocumentList(vocabularyPos, vocabularyNeg, listOfDocumentsPath, numberOfCentroids)
			print("Vlawe descriptors", vlaweDocuments.shape)
		
			kernelRepresentation= makeLinearKernelVLAWE(vlaweDocuments)
			print("kernel shape", kernelRepresentation.shape)
		
		
			for cValue in cSVM: 
				print("linearSVM", cValue)
				linearSVM= svm.SVC(C=cValue, kernel='precomputed')
				cvs= cross_val_score(linearSVM, kernelRepresentation, labels, cv=10)
				scoresVLAWE.append(cvs)
				print("Cross valid for 10 experiments: ", cvs)
			writeFile(fileVLAWE, np.array(scoresVLAWE))
			print("The file", fileVLAWE, " was created")
		
		return scoresBOSWE, scoresVLAWE
		
	
		



#path of files that contain word embeddings
documentsPathPos="C:/Users/Dell/Desktop/THISisTheCharmedOne/pos/"
documentsPathNeg="C:/Users/Dell/Desktop/THISisTheCharmedOne/neg/"



#cvs=runBOSWE_VLAWE(runBOSWE=True, runVLAWE=True, vocabularyDBSCAN=False, documentsPathPos=documentsPathPos, documentsPathNeg=documentsPathNeg, numberOfVocabularies=1, numberOfCentroids=50, createCentroids=True, cSVM=[0.1, 1, 10, 100], fileForCVS="dbscan_eps_80_dbscan_min_samples_90"+"centroids_KMEANS_boswe.txt")
cvs=runBOSWE_VLAWE(runBOSWE=True, runVLAWE=True, vocabularyDBSCAN=True, documentsPathPos=documentsPathPos, documentsPathNeg=documentsPathNeg, numberOfVocabularies=2, numberOfCentroids=2, createCentroids=False, cSVM=[0.1, 1, 10, 100], fileForCVS="dbscan_eps_80_dbscan_min_samples_90"+"centroids_KMEANS_boswe.txt", dbscan_eps_val=50, dbscan_min_samples_val=100)
for c in cvs:
	print(c, "acuratetea pentru cross validation")



'''
			Rezultate
			
		numberOfVocabularies=2
		numberOfCentroids=10 => histograms =20
	C=1

'''