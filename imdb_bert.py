import warnings
warnings.filterwarnings("ignore")

import multiprocessing
import bertWordEmbeddings
from pytorch_pretrained_bert import BertTokenizer, BertModel



def apply_bert(listOfSnippets, type):
	
	bert_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased') #load the pre-trained tokenizer
	bert_model=BertModel.from_pretrained('bert-base-uncased') #load the pre-trained model
	
	for d in range(len(listOfSnippets)):
		# dd will be the name for the file that will be created
		dd='C:/Users/Dell/Desktop/THISisTheCharmedOne/'+type+'/' + type+  '_'+ str(d) + ".txt" # dd will be the name for the file that will be created
		print(dd)
		text_embeddings=bertWordEmbeddings.bertWordVectors(bert_tokenizer, bert_model,listOfSnippets[d])
		f=open(dd,'w+')
		for embs in text_embeddings:
			for i in range(len(embs)):
				f.write(str(embs[i]))
				if i != len(embs)-1:
					f.write(',')
			f.write('\n')	
		f.close()
		print('Document %s was created' %dd)
	return

#read and create list of snippets for negative and positive data

neg_file=open("rt-polaritydata/rt-polarity.neg","r")
neg_snippets=[]
for snip in neg_file:
	neg_snippets.append(snip)

pos_file=open("rt-polaritydata/rt-polarity.pos","r")
pos_snippets=[]
for snip in pos_file:
	pos_snippets.append(snip)

#apply BERT to obtain the word embeddings 
apply_bert(neg_snippets, 'neg')
apply_bert(pos_snippets, 'pos')