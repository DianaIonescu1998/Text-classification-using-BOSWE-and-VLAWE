
class Vocabulary:
	def __init__(self):
		#create vocabulary using k-means
		self.kdtree=None #the centroids will be stored in a kd-tree structure
		self.centers=None #the centroids will be also stored in a list and the position of the centroid will be its label
		#create vocabulary using DBSCAN
		self.dbscan=None #model to be used
		self.dbscanClusters=None #number of clusters that were created
		self.dbscanLabels=None # labels for the train data
		self.dbscanListOfKDTrees=None #list of N clusters that contains the core points for each cluster
		self.dbscanMeanCluster=None #list of the centroids to be used for VLAWE representation
		self.eps=None #the maximum distance between 2 points so that they are called neighbours
		self.minSamples=None # minimun number of points located in the self.eps vicinity of a point to create a new cluster 
	def setValues(kdtree, centers):
		self.kdtree=kdtree
		self.centers=centers
	def setDBSCAN(dbscan):
		self.dbscan=dbscan