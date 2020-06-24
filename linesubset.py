import numpy as np
# returns "number" columns from a matrix 

def linesubset(matrix, number):
	new_matrix=[]
	
	matrix=np.array(matrix)
	if(len(matrix.shape)!=2):
		return []
	lines, columns= matrix.shape
	
	if lines <= number or lines<1:
		return matrix
	elif number <1:
		return [[]]
	elif lines > number:
		
		if number==2:
			new_matrix.append(matrix[0,:])
			new_matrix.append(matrix[lines-1, :])
			return new_matrix
			
		if number==1:
			
			new_matrix.append(matrix[lines-1,:])
			return new_matrix
		
		if number >= 3:
			listOnes=np.zeros(lines)
			listOnes[0]=1
			listOnes[lines-1]=1
			number-=2
			(listOnes, _)=onesArray(listOnes,1, lines-2, number)
			listOnes=np.array(listOnes)
			#print("listOnes", listOnes)
			ones=np.where(listOnes==1)
			#print(ones)
			for i in ones:
				new_matrix.append(matrix[i, :])
			
	return new_matrix[0]
	

#this function determinates an uniform repartization of number values of 1 for the zerosArray
#ok returns 0 or 1 -used just for implementation
def onesArray(zerosArray, first, last, number):
	ok=0
	if number == 0:
		return zerosArray, ok
	if number == 1:
		mean= int((first+last)/2)
		
		if first<= mean and mean<= last and zerosArray[mean]==0:
			zerosArray[mean]=1
			ok=1
		elif ok==0:
			for i in range(1, len(zerosArray)-1):
				if zerosArray[i]==0:
					zerosArray[i]=1
					return zerosArray, ok
	elif number > 1:
		mean= int((first+last)/2)
		if first<= mean and mean<= last :
			
			onesArray(zerosArray, first, mean, int(number/2))
			left_number= number- int(number/2)
			onesArray(zerosArray, mean+1, last, left_number)
	#print('zerosArray ', zerosArray)
	return zerosArray, ok

'''
l=[]
for i in range(25):
	l.append([i, i, i,i,i,i,i,i])

#l=np.array(l).transpose()

print(l)
print('\n',np.array(linesubset(l,15)))
'''