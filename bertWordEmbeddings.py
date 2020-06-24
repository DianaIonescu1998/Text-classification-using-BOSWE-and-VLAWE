import warnings
warnings.filterwarnings("ignore")

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import numpy as np


def addSpecialTokens(sentence1, sentence2=None):
	special_words=['[CLS] ' , ' [SEP] ']
	final_sentence= special_words[0]+sentence1
	if sentence2 != None:
		final_sentence+= special_words[1]+sentence2
	final_sentence+=special_words[1]
	return final_sentence
	
def decideIndex(tokens):
	#sentence_idexes is a list of zeros and/or ones and it is used to decide 
	#which token belongs to the first sentence (0) and which not (1)
	sentence_idexes=[1]*len(tokens) #we assume that all tokens belong to the second sentence 
	index=0
	#the first sentence begins with '[CLS]' and ends with the first '[SEP]'
	'''
	while '[SEP]' not in tokens[index]:
		sentence_idexes[index]=0
		index+=1
	sentence_idexes[index]=0 #the first '[SEP]' separator belongs to the first sentence
	'''
	return sentence_idexes


def separateWords(tokens):
	separate_words=[]
	start_word=-1
	end_word=-1
	i=0
	while i<len(tokens):
		if '[CLS]' not in tokens[i] and '[SEP]' not in tokens[i]:
			start_word=i
			j=i+1
			while j < len(tokens) and '##' in tokens[j]:
				j+=1
			if j==i+1:
				i+=1
			else:
				i=j
			end_word=j-1
			separate_words.append((start_word,end_word))
		else:
			i+=1
	return separate_words


def bertWordVectors(bert_tokenizer, bert_model, text):
	text_embeddings=[]
	
	bert_model.eval()
	
	#prepare text for tokenize sentences:
	#sentences=sent_tokenize(text)
	number_sentences=0
	while number_sentences <1 : #< len(sentences):
		number_sentences=1
		sentence1=text
		sentence2=None
		word_embeddings=[] # list with elements: [word, vector_of_features]
		
		#for sentence1 and sentence2  add separators:
		sentence= addSpecialTokens(sentence1, sentence2)
		#using bert_tokenizer, tokenize the sentence
		tokens=bert_tokenizer.tokenize(sentence)
		#using bert_tokenizer get the indexes for each token from the pre-trained voabulary
		indexes_tokens=bert_tokenizer.convert_tokens_to_ids(tokens)
		#separate tokens from sentence1 and from sentence2
		indexes_sentence=decideIndex(tokens)
	
		#BERT is a PyTorch and works only with tensors:
		##indexes_tokens_tensor is the list with the indexes of the words from vocabulary
		indexes_tokens_tensor=torch.tensor([indexes_tokens])
		##indexes_sentence_tensor is the list with the contain of each phrase
		indexes_sentence_tensor=torch.tensor([indexes_sentence])
	
		#using bert_model, predict hidden states:
		with torch.no_grad():
			encoded_layers, _ = bert_model(indexes_tokens_tensor, indexes_sentence_tensor)
		#create a new dimension to the tensor
		token_embeddings = torch.stack(encoded_layers, dim=0)
		#remove te batches dimension (its value is 1)
		token_embeddings = torch.squeeze(token_embeddings, dim=1)
		#permute dimension 1 to 0
		token_embeddings = token_embeddings.permute(1,0,2)
	
		##to get the embeddings for each token, we summize the last 4 layers on each feature
		features_sum=[]
		for embedding in token_embeddings:
			sum_4_layers=torch.sum(embedding[-4:], dim=0)
			features_sum.append(sum_4_layers.tolist())
		
		separate_words=separateWords(tokens)
		for (start,stop) in separate_words:
			if start==stop:
				list=(np.float32(features_sum[start])).tolist()
				
			else: #if it is a word that was separated 
				word=''
				for w in tokens[start:stop+1]:
					word+=w
				word=word.replace('##','')
				list=((np.float32(np.mean(features_sum[start:stop+1],axis=0))).tolist())

			word_embeddings.append(list)
		
		text_embeddings+=word_embeddings

	return text_embeddings

'''

def bertWordVectors(text):
	text_embeddings=[]
	bert_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased') #load the pre-trained tokenizer
	bert_model=BertModel.from_pretrained('bert-base-uncased') #load the pre-trained model
	bert_model.eval()
	
	#prepare text for tokenize sentences:
	sentences=sent_tokenize(text)
	number_sentences=0
	while number_sentences < len(sentences):
		print("Status: ", number_sentences/len(sentences), '%.....')
		sentence1=sentences[number_sentences]
		number_sentences+=1
		if number_sentences == len(sentences):
			sentence2=None
		else:
			sentence2=sentences[number_sentences]
			number_sentences+=1
		
		word_embeddings=[] # list with elements: [word, vector_of_features]
		#example :['later', -2.410511016845703, -2.6429710388183594, 1.3048564195632935, ....] - 769 features including the word
	
		#for sentence1 and sentence2  add separators:
		sentence= addSpecialTokens(sentence1, sentence2)
		#using bert_tokenizer, tokenize the sentence
		tokens=bert_tokenizer.tokenize(sentence)
		#using bert_tokenizer get the indexes for each token from the pre-trained voabulary
		indexes_tokens=bert_tokenizer.convert_tokens_to_ids(tokens)
		#separate tokens from sentence1 and from sentence2
		indexes_sentence=decideIndex(tokens)
	
		#BERT is a PyTorch and works only with tensors:
		indexes_tokens_tensor=torch.tensor([indexes_tokens])
		indexes_sentence_tensor=torch.tensor([indexes_sentence])
	
		#using bert_model, predict hidden states:
		with torch.no_grad():
			encoded_layers, _ = bert_model(indexes_tokens_tensor, indexes_sentence_tensor)
		#create a new dimension to the tensor
		token_embeddings = torch.stack(encoded_layers, dim=0)
		#remove te batches dimension (its value is 1)
		token_embeddings = torch.squeeze(token_embeddings, dim=1)
		#permute dimension 1 to 0
		token_embeddings = token_embeddings.permute(1,0,2)
	
		#to get the embeddings for each token, we summize the last 4 layers on each feature
		
		features_sum=[]
		for embedding in token_embeddings:
			sum_4_layers=torch.sum(embedding[-4:], dim=0)
			features_sum.append(sum_4_layers.tolist())
		
		separate_words=separateWords(tokens)
		for (start,stop) in separate_words:
			#print('working....')
			if start==stop:
				list=[tokens[start]]+(np.float32(features_sum[start])).tolist()
				#print(list[0:5])
			else:
				word=''
				for w in tokens[start:stop+1]:
					word+=w
				word=word.replace('##','')
				list=[word]+((np.float32(np.mean(features_sum[start:stop+1],axis=0))).tolist())
				#print(word)
				#print(list[0:5])
			word_embeddings.append(list)
		#print(sentence1, sentence2)
		
		text_embeddings+=word_embeddings
	print('\a')
	return text_embeddings
'''
