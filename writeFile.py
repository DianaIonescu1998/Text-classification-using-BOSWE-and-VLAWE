import  numpy as np

def writeFile(fileName, npArray):
	if len(np.array(npArray).shape)==1:
		npArray=[npArray]
		
	f=open(fileName,'w+')
	for line in npArray:
		for i in range(len(line)):
			f.write(str(line[i]))
			if i != len(line)-1:
				f.write(',')
		f.write('\n')	
	f.close()
