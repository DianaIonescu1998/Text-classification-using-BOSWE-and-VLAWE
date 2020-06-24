from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KDTree
import numpy as np
import time

from linesubset import *
from writeFile import *
from dbscanKdtrees import *

#given a list of paths that contains word embeddings it
#creates vocabulary from the word embeddings extracted from each text
##OUTPUT: list of size (numberOfCentroids x features) = centers of the kmeans clusterization

def createVocabularyFromListOfPaths(vocabulary, vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids=10, dbscan_eps=70, dbscan_min_samples=10, file="none.txt"):
	#listOfDocumentsPath - here is the list of documents that contain word embeddings
	#numberOfCentroids = number of centers for the kmeans algorithm
	
	features = 240000
	descriptors=np.zeros((0,768)) # will contain some embeddings from each document
	number_of_names=len(listOfDocumentsPath)
	numberMaximOfWords=int(features/number_of_names)
	
	
	if file=="positive_embeddings_from_documents.txt" or file=="negative_embeddings_from_documents.txt":
	###################################read all the embeddings from file
		descriptors=[]
		
		
		print("Reading embeddings from file...", file)
		
		f=open(file, 'r+')
		
		for line in f:
			descriptors.append([float(l) for l in line.split(',')])
			
				
	elif file=="positive_negative.txt":
		print("elif.....................")
		descriptors=[]
		
		
		file="positive_embeddings_from_documents.txt" 
		print("Reading embeddings from file...", file)
		
		f=open(file, 'r+')		
		for line in f:
			descriptors.append([float(l) for l in line.split(',')])
			
		
		file="negative_embeddings_from_documents.txt"
		print("Reading embeddings from file...", file)
		
		f=open(file, 'r+')
		for line in f:
			descriptors.append([float(l) for l in line.split(',')])
			
		
		
	
		
	else:
		#read embeddings for each document and create descriptors
	
		print(">Create vocabulary.......\n")
		for documentPath in listOfDocumentsPath:
			print("Processing data from document", documentPath, "...")
			#read features from each document
			des =np.genfromtxt(documentPath, dtype='float32', delimiter=',') #shape: (word, features)
			#check if the file contains data
			if (len(des)>0):
				if len(des.shape)==1:
					des=np.array([des])
				#linesubset returns minimum number of word embeddings between int(features/number_of_names) and number of words from document
				descriptors=np.concatenate((descriptors,linesubset(des, numberMaximOfWords)), axis=0) 
	
		#write file with descriptors
		writeFile(file, np.array(descriptors))
	
	
	
	print(len(descriptors),"descriptors shape ", np.array(descriptors).shape)
	descriptors=np.array(descriptors)
	
	
	if vocabularyDBSCAN==True:
	
		print(vocabulary.dbscan)
		print("fitting ", int(len(descriptors)), "elements.......")
		
		tic = time.perf_counter()
		vocabulary.dbscan.fit(descriptors)
		toc = time.perf_counter()
		print(f"Executed dbscan.fit in {toc - tic:0.4f} seconds")
		
		labels=vocabulary.dbscan.labels_
		corePointsIndices=vocabulary.dbscan.core_sample_indices_
		
		
		#get the total number of clusters, without the noise
		dbscanClusters=len(set(labels)) - (1 if -1 in labels else 0)
		print("Number of clusters", dbscanClusters)
		if(dbscanClusters==0 or dbscanClusters ==len(descriptors)):
			print("Not ok")
			return None, None, None, None
		
		#create a list of kdtrees that contain the core points of each cluster
		##create a list of centers for each cluster
		(listOfKdtreesOfCorePoints, clustersMean)=dbscanGetKDTreesAndClustersMean(descriptors, labels, corePointsIndices, dbscanClusters)

		return dbscanClusters, labels, listOfKdtreesOfCorePoints, clustersMean 
	
	
	else:
		print("Create k-means vocabulary")
		kmeans= KMeans(n_clusters=numberOfCentroids, verbose=0, algorithm='elkan')
	
		batch=50
		processedDescriptors=0
		print("word embeddings to create vocabulary", np.array(descriptors).shape)
	
		kmeans.fit(descriptors)
		
	
		#get centers of the clusters
		centers=kmeans.cluster_centers_
	
		#put the centers in a kd-tree
		kdtree= KDTree(centers)
	
		return centers, kdtree