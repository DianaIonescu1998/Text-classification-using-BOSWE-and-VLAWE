from sklearn.neighbors import KDTree
import numpy as np

def dbscanGetKDTreesAndClustersMean(listOfEmbeddings, listOfDbscanLabels, listOfCorePoints, numberOfClusters):
	kdtrees=[]
	clustersMean=[]
	#labels have values between 0 and numberOfClusters-1
	for label in range(numberOfClusters):
	
		#get indices for features with labels equals label
		indices= np.where(np.array(listOfDbscanLabels) == label) 
		
		#get the mean for the current cluster
		meanForCluster=np.mean(np.array(listOfEmbeddings)[indices],axis=0)
		clustersMean.append(meanForCluster)
		
		#create KDTree for the core points
		kdtreeIndices=np.intersect1d(indices, np.array(listOfCorePoints))
		
		embeddings=np.array(listOfEmbeddings)[kdtreeIndices]
		tree=KDTree(embeddings, leaf_size=5000)
		kdtrees.append(tree)
		
	return kdtrees, clustersMean

def dbscanGetLabel(feature, listOfKDtrees, eps):
	
	for label, kdtree in enumerate(listOfKDtrees):
		#for a KDTree return the minimum distance and the index
		(distance, _) = kdtree.query( [feature] , k=1,  return_distance=True)
		#print("dbscan kdtrees",type(distance), distance[0], np.array(distance).shape)
		if distance[0] <= eps:
			#print(distance[0][0] <= eps, eps)
			return label
			
	return -1 #if it is a noise point

