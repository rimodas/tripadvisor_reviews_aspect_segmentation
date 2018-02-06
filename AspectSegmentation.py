
# coding: utf-8

# In[47]:

import sys, string
import csv
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import re
import os
import codecs
import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
stemmer = PorterStemmer()
aspectType =  {0:'Value', 1:'Rooms', 2:'Location', 3:'Cleanliness', 4:'Check in/Front Desk', 5:'Service', 6:'Business Service'}

#load all review texts
def load_file(file):
	reviews = []
	ratings = []
	f = open(file,'r')
	for line in f:
#		l = line.strip().split('>')
#		if l[0] == '<Content':
#			s = str(l[1])
		reviews.append(line)
#		elif l[0] == '<Rating':
#			r = l[1].split('\t')
#			ratings.append(int(r[1]))
	f.close()
	#print (len(reviews), reviews[1])
	return reviews , ratings

def parse_to_sentence(reviews):
	review_processed = []
	actual = []
	only_sent = []
	for r in reviews:
		sentences = nltk.sent_tokenize(r)
		actual.append(sentences)
		sent = []
		for s in sentences:
			#words to lower case
			s = s.lower()
			#remove punctuations and stopwords
			replace_punctuation = s.maketrans(string.punctuation, ' '*len(string.punctuation))
			s = s.translate(replace_punctuation)
			stop_words	 = list(stopwords.words('english'))
			additional_stopwords = ["'s","...","'ve","``","''","'m",'--',"'ll","'d"]
			# additional_stopwords = []
			stop_words = set(stop_words + additional_stopwords)
			# print stop_words
			# sys.exit()
			word_tokens = word_tokenize(s)
			s = [w for w in word_tokens if not w in stop_words]
			#Porter Stemmer
			stemmed = [stemmer.stem(w) for w in s]
			if len(stemmed)>0:
				sent.append(stemmed)
		review_processed.append(sent)
		only_sent.extend(sent)
	return review_processed, actual, only_sent

# sent = parse_to_sentence(reviews)
# print len(sent), sent[2]

def create_vocab(sent):
	words = []
	for s in sent:
		words += s
	freq = FreqDist(words)
	vocab = []
	for k,v in freq.items():
		if v > 5:
			vocab.append(k)
	#Assign a number corresponding to each word. Makes counting easier.
	vocab_dict = dict(zip(vocab, range(len(vocab))))
	return vocab, vocab_dict


#goal: map sentences to corresponding aspect.

def get_aspect_terms(file, vocab_dict):
	aspect_terms = []
	w_notfound = []
	f = open(file, "r")
	for line in f:
		s = line.strip().split(",")
		stem = [stemmer.stem(w.strip().lower()) for w in s]
		#we store words by their corresponding number.
		# aspect = [vocab_dict[w] for w in stem]
		aspect = []
		for w in stem:
			if w in vocab_dict:
				aspect.append(w)
			else:
				w_notfound.append(w)
		aspect_terms.append(aspect)
	#We are only using one hotel review file, as we keep inceasing the number of files words not found will decrease.
	# print "Words not found in vocab:", ' '.join(w_notfound)
	f.close()
	return aspect_terms

# def chi_sq(w, A, sent):

def chi_sq(a,b,c,d):
	c1 = a
	c2 = b - a
	c3 = c - a
	c4 = d - b - c + a
	nc =  d
	return nc * (c1*c4 - c2*c3) * (c1*c4 - c2*c3)/((c1+c3) * (c2+c4) * (c1+c2) * (c3+c4))

def chi_sq_mat():
	global aspect_words, aspect_sent, num_words
	asp_rank = np.zeros(aspect_words.shape)
	for i in range(len(aspect_terms)):
		for j in range(len(vocab)):
			asp_rank[i][j] = chi_sq(aspect_words[i][j], num_words[j], aspect_sent[i], len(sent))
	return asp_rank

def aspect_segmentaion(file, aspect_file):
	#Sentiment analysis
	sid = SIA()

	#INPUT
	#review, this algo needs all the review. Please process dataset.
	reviews, all_ratings = load_file(file)

	#selection threshold
	p = 5
	#Iterations 
	# I = 10
	I = 1

	#Create Vocabulary
	review_sent, review_actual, only_sent = parse_to_sentence(reviews)
	vocab, vocab_dict = create_vocab(only_sent)

	#Aspect Keywords
	aspect_terms = get_aspect_terms(aspect_file, vocab_dict)

	label_text = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Check in/Front Desk', 'Service', 'Business Service']
	# print aspect_terms

	#ALGORITHM
	review_labels = []
	k = len(aspect_terms)
	v = len(vocab)
	aspect_words = np.zeros((k,v))
	aspect_sent = np.zeros(k)
	num_words = np.zeros(v)

	for i in range(I):
		for r in review_sent:
			labels = []
			for s in r:
				count = np.zeros(len(aspect_terms))
				i = 0
				for a in aspect_terms:
					for w in s:
						if w in vocab_dict:
							num_words[vocab_dict[w]] += 1
							if w in a:
								count[i] += 1
					i = i + 1
				if max(count) > 0:
					la = np.where(np.max(count) == count)[0].tolist()
					labels.append(la)
					for i in la:
						aspect_sent[i] += 1
						for w in s:
							if w in vocab_dict:
								aspect_words[i][vocab_dict[w]] += 1
				else:
					labels.append([])
			review_labels.append(labels)
			# aspect_w_rank = chi_sq_mat()
			# new_labels = []
			# for na in aspect_w_rank:
			# 	x = np.argsort(na)[::-1][:p]
			# 	new_labels.append(x)
				# for k,v in vocab_dict.items():
				# 	if vocab_dict[k] in x:
				# 		print k
				# print 
			# sys.exit()


	ratings_sentiment = []
	for r in review_actual:
		sentiment = []
		#aspect ratings based on sentiment
		for s in r:
			ss = sid.polarity_scores(s)
			sentiment.append(ss['compound'])
		ratings_sentiment.append(sentiment)

	#Aspect Ratings Per Review
	aspect_ratings = []
	for i,r in enumerate(review_labels):
		rating = np.zeros(7)
		count = np.zeros(7)
		rs = ratings_sentiment[i] 
		for j,l in enumerate(r):
			for k in range(7):
				if k in l:
					rating[k] += rs[j]
			for k in range(7):
				if count[k] != 0:
					rating[k] /= count[k]
		#Map from -[-1,1] to [1,5]
		for k in range(7):
			if rating[k] != 0:
				rating[k] = int(round((rating[k]+1)*5/2))
		aspect_ratings.append(rating)
	return aspect_ratings, all_ratings

	# n = 0
	#print (review_actual[n], '\n', review_labels[n])
	#print (ratings_sentiment[n], '\n', aspect_ratings[n])
	#print (len(all_ratings), len(reviews), all_ratings[0])
	# sys.exit()

	#print (sent[5:9], labels[5:9])
	#print (zip(actual_sent, labels)[:10])
	#print (zip(actual_sent, sentiment)[:10])
	return aspect_ratings

def getAspects(base_review_url,hotelName):
	file_name = base_review_url+hotelName+'.csv'
	u_cols = ['header', 'review', 'rating']
	reviews = []
	reviewdata = pd.read_csv(file_name, sep=',', names=u_cols)
	for i in range(len(reviewdata)):		reviews.append(reviewdata.iloc[i,1])

	out_file = open(base_review_url+hotelName+'.txt','w')
	from six import string_types
	for s in reviews:
		try:
			out_file.write(s+'\n')
		except: pass
		#else:
		#	out_file.write(str(s.encode("utf-8").decode("utf-8").replace(u"\u2022", ''))+'\n')
	out_file.close()
	aspect_file	 = base_write_url+"aspect_keywords.csv"
	a_s = aspect_segmentaion(base_review_url+hotelName+'.txt',aspect_file)
	#os.remove(base_review_url+hotelName+'.txt')
	a_s = list(a_s[0])
	aspectsList = []
	for i in a_s: aspectsList.append(i.tolist())
	return aspectsList

def writeAspectsForRegression(hotelName):
	u_cols = ['Header', 'Reviews','Ratings']
	base_review_url = base_write_url+'Reviews/'
	base_aspect_url = base_write_url+'Aspects/'
	reviewsRating = []
	reviewData = pd.read_csv(base_review_url+hotelName+'.csv', sep=',', names=u_cols)
	for i in range(len(reviewData)):
		reviewsRating.append(reviewData.iloc[i,2])
	reviewAspects = []
	labelsForAspectCsv = ['Tripadvisor rating']
	labelsForAspectCsv.extend(aspectType.values())
	reviewAspects.append(labelsForAspectCsv)
	allAspects = getAspects(base_review_url,hotelName)
	count = 0
	for i in allAspects:
		reviewAspectsList= [reviewsRating[count]]
		reviewAspectsList.extend(i)
		reviewAspects.append(reviewAspectsList)
		count+=1
	with open(base_aspect_url+hotelName+'.csv', 'w') as fp:
		csv_writer = csv.writer(fp, delimiter=',')
		csv_writer.writerows(reviewAspects)

hotelNames = []
hotelData = pd.read_csv(base_write_url+'master.csv', sep=',', names=u_cols)
for i in range(len(hotelData)):
    hotelNames.append(hotelData.iloc[i,0])
for hotelName in hotelNames:
    writeAspectsForRegression(hotelName)
print('hotel aspects written')

