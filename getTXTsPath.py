import os

#return a list with the path and the name of each .txt file

def getTXTsPath(path):
	txt_files=[]
	for file in os.listdir(path):
		if file.endswith(".txt"):
			txt_files.append(os.path.join(path, file))
	return txt_files
