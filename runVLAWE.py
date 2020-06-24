from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neighbors import KDTree

from createVocabularyFromListOfPaths import *
from getTXTsPath import *
from Vocabulary import *
from computeVLAWEForOneVocabularyFromDocumentList import *
from makeLinearKernelVLAWE import *
from writeFile import *
from saveLoadVLAWE import *
from makeLinearKernelVLAWEusingPCA import *
from computeVLAWEForTwoVocabulariesFromDocumentList import *

#path of files that contain word embeddings
documentsPathPos="C:/Users/Dell/Desktop/THISisTheCharmedOne/pos/"
documentsPathNeg="C:/Users/Dell/Desktop/THISisTheCharmedOne/neg/"

numberOfVocabularies=1
numberOfCentroids=10


def runVLAWE(vocabularyDBSCAN=True, documentsPathPos, documentsPathNeg, numberOfVocabularies, numberOfCentroids, createCentroids=False, cSVM=[0.1, 1, 10, 100], fileForCVS=str(numberOfVocabularies)+"_vlawe_output.txt"):
	
	if numberOfVocabularies==1:

		vocabulary= Vocabulary() #centroids and kd-tree for centroids
		
		print("Create one vocabulary...")
		
		#get all the names of documents from file and create a list of labels
		listOfDocumentsPath= getTXTsPath(documentsPathPos)
		positive=len(listOfDocumentsPath)
		listOfDocumentsPath+=getTXTsPath(documentsPathNeg)
		labels=np.zeros(len(listOfDocumentsPath))
		labels[0:positive]=1
		
		if vocabularyDBSCAN==True:
			#createVocabularyFromListOfPaths( vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids=10, dbscan_eps=0.3, dbscan_min_samples=100):
			vocabulary.dbscan=createVocabularyFromListOfPaths(vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids)
		else:
			if createCentroids==True:
				#To create centroids 
				(vocabulary.centers, vocabulary.kdtree) = createVocabularyFromListOfPaths(vocabularyDBSCAN, listOfDocumentsPath, numberOfCentroids)
				writeFile("centers_voc.txt",vocabulary.centers)
			else:
				#To load centroids
				documentPathCenters='C:/Users/Dell/Desktop/THISisTheCharmedOne/centers_voc'
				vocabulary.centers=np.genfromtxt(documentPathCenters, delimiter=',')
				print("Centers: ",(vocabulary.centers).shape)
				vocabulary.kdtree= KDTree(vocabulary.centers)
		
		
		print("Compute VLAWE embeddings")
		vlaweDocuments=computeVLAWEForOneVocabularyFromDocumentList(vocabulary, listOfDocumentsPath, numberOfCentroids)
		print(np.array(vlaweDocuments).shape)
		
		#save the vlawe descriptors for each document
		'''
		labelsPath='vlaweLabels'
		vlawePath='vlawePath'
		saveVLAWEForDocuments(labelsPath, vlawePath, labels, vlaweDocuments)
		'''
		
		#create kernel
		
		#kernelRepresentation= makeLinearKernelVLAWE(vlaweDocuments)
		kernelRepresentation= makeLinearKernelVLAWEusingPCA(vlaweDocuments)
		
		print("kernelRepresentation (should have 10662x10662 ) ->",kernelRepresentation.shape)
				
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
	
	if numberOfVocabularies==2:
		
		vocabularyPos=Vocabulary()
		vocabularyNeg=Vocabulary()
		
		
		print("Create two vocabularies...")
		listOfDocumentsPathPos= getTXTsPath(documentsPathPos)
		listOfDocumentsPathNeg=getTXTsPath(documentsPathNeg)
		labelsPos=np.ones(len(listOfDocumentsPathPos))
		labelsNeg=np.zeros(len(listOfDocumentsPathNeg))
		listOfDocumentsPath=np.concatenate((listOfDocumentsPathPos,listOfDocumentsPathNeg), axis=0)
		labels=np.concatenate((labelsPos, labelsNeg), axis=0)
		
		

		if createCentroids==True:
			#create and save 2 vocabularies
			(vocabularyPos.centers, vocabularyPos.kdtree) = createVocabularyFromListOfPaths(listOfDocumentsPathPos, numberOfCentroids)
			writeFile("centers_voc_pos.txt",vocabularyPos.centers)
			(vocabularyNeg.centers, vocabularyNeg.kdtree) = createVocabularyFromListOfPaths(listOfDocumentsPathNeg, numberOfCentroids)
			writeFile("centers_voc_neg.txt",vocabularyNeg.centers)
			
		else:
			
			#load negative centroids
			documentPathCenters='C:/Users/Dell/Desktop/THISisTheCharmedOne/centers_voc_neg.txt'
			vocabularyNeg.centers=np.genfromtxt(documentPathCenters, delimiter=',')
			print("Centers for negative vocabulary: ",(vocabularyNeg.centers).shape)
			vocabularyNeg.kdtree= KDTree(vocabularyNeg.centers)
			
			#load positive centroids
			documentPathCenters='C:/Users/Dell/Desktop/THISisTheCharmedOne/centers_voc_pos.txt'
			vocabularyPos.centers=np.genfromtxt(documentPathCenters, delimiter=',')
			print("Centers for positive vocabulary: ",(vocabularyPos.centers).shape)
			vocabularyPos.kdtree= KDTree(vocabularyPos.centers)
		
		
		print("Compute VLAWE embeddings")
		vlaweDocuments=computeVLAWEForTwoVocabulariesFromDocumentList(vocabularyPos, vocabularyNeg, listOfDocumentsPath, numberOfCentroids)
		print("Vlawe descriptors", vlaweDocuments.shape)
		
		kernelRepresentation= makeLinearKernelVLAWE(vlaweDocuments)
		print("kernel shape", kernelRepresentation.shape)
		
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
	
	
	
	

C=[0.1 , 1, 10, 100]
scores= runVLAWE(documentsPathPos, documentsPathNeg, numberOfVocabularies=1, numberOfCentroids=10, createCentroids=False, cSVM=[0.1, 1, 10, 100], fileForCVS="1vlaweKMEANS_results.txt")
for s in scores:
	print(s)
'''
CVS=[]
for c in C:
	CVS.append(runVLAWE(documentsPathPos, documentsPathNeg, numberOfVocabularies=2, numberOfCentroids=10, createCentroids=False, cSVM=c))
print("results:")	
for c in CVS:
	print(c)
	
writeFile("1vlaweKMEANS_results.txt",np.array(CVS))
'''
'''
def runVLAWE(documentsPathNeg, documentsPathPos)
	listOfDocumentsPath=[] 

	#get all the names of documents from file
	vocabulary= Vocabulary() #centroids and kd-tree for centroids

	vlaweDocuments=[]

	if numberOfVocabularies==1:

		print(">>Create one vocabulary...")
		listOfDocumentsPath= getTXTsPath(documentsPathPos)
		listOfDocumentsPath+=getTXTsPath(documentsPathNeg)
		(vocabulary.centers, vocabulary.kdtree) = createVocabularyFromListOfPaths(listOfDocumentsPath[:30], numberOfCentroids)
	

		print(">>Compute VLAWE embeddings")
		vlaweDocuments=computeVLAWEForOneVocabularyFromDocumentList(vocabulary, listOfDocumentsPath, numberOfCentroids)
	
		vlaweHistograms= makeLinearKernelVLAWE(vlaweDocuments)
'''




'''
####Results

[0.83973758 0.80599813 0.8358349  0.81425891 0.80581614 0.83020638
 0.83489681 0.80863039 0.81707317 0.81801126]




()numberOfCentroidsentroids=10

SVM C=0.1
Cross valid for 10 experiments:  [0.81068416 0.81818182 0.81050657 0.79268293 0.78236398 0.80393996
 0.81613508 0.78893058 0.80300188 0.80675422]

SVM C=1
Cross valid for 10 experiments:  [0.82380506 0.8163074  0.82926829 0.80769231 0.80393996 0.81801126
 0.81425891 0.79549719 0.80675422 0.83114447]

SVM C=10
Cross valid for 10 experiments:  [0.80506092 0.79943768 0.81238274 0.79831144 0.80018762 0.79737336
 0.80300188 0.78986867 0.78986867 0.81332083]
 
 
 
()numberOfCentroidsentroids=20
PCA => 768 
SVM, C=1
Cross valid for 10 experiments:  [0.83223993 0.82380506 0.8217636  0.81707317 0.81238274 0.8011257
 0.81332083 0.81050657 0.81801126 0.82551595]
 

'''