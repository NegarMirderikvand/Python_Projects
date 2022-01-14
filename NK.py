# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:56:48 2020

@author: Negar
"""


import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import nltk
import tensorflow as tf
from collections import Counter
import csv, re, pickle

import matplotlib as plt

import fasttext
import hazm

df=pd.read_csv('E:/Training/DSA/FinalProject/nk-1/nk.csv')
df.head()
np.sum(df.isna())
df.describe()
df.columns
reviews = df['comment']
adv = df['advantages']
disadv=df['disadvantages']
rate = df['likes']
labels = df['recommend']


labels1 = np.array([1 if each=="recommended" else -1 if each=="not_recommended" else 0 for each in labels])
#cleaning dataset
words=[]
all_text = ''

for t in range (len(reviews)):
	text = reviews[t]
	text = re.sub(r'[^a-zA-Z0-9آ-ی۰-۹ ]', ' ', str(text))
	all_text += text
	all_text += ' '
	wordsInText = text.split()
	for word in wordsInText:
		if word != ' ' or word != '':
			words.append(word)


counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

with open("mySavedDict.txt", "wb") as myFile:
    pickle.dump(vocab_to_int, myFile)


with open("mySavedDict.txt", "rb") as myFile:
    myNewPulledInDictionary = pickle.load(myFile)

print (myNewPulledInDictionary)


reviews_ints = []
for each in reviews:
	print (each)
	#each = each.replace('\u200c',' ')
	each = re.sub(r'[^a-zA-Z0-9آ-ی۰-۹ ]', ' ', str(each))
	reviews_ints.append([vocab_to_int[word] for word in each.split()])


'''
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
'''

seq_len = 400
features = np.zeros((len(reviews), seq_len), dtype=int)
for i, row in enumerate(reviews_ints):
    if(len(row)!=0):
        print (i , row)
       	features[i, -len(row):] = np.array(row)[:seq_len]
    else:
        print (len(row),'****')
split_frac = 0.9
split_idx = int(len(features)*split_frac)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x1, X_test = val_x[:test_idx], val_x[test_idx:]
val_y1, y_test = val_y[:test_idx], val_y[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(X_test.shape))


from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import scale,LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_modelel import LogisticRegression,LinearRegression
#from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dectree = tree.DecisionTreeClassifier(max_depth=10)
bag = BaggingClassifier(n_estimators=100,oob_score=True)
rf = RandomForestClassifier(n_estimators=1000,oob_score=True,max_features='auto')
boost = AdaBoostClassifier(n_estimators=1000)

dectree.fit(train_x,train_y)
dectree.feature_importances_
bag.fit(train_x,train_y)
rf.fit(train_x,train_y)
boost.fit(train_x,train_y)
print('Tree','Bagging','Boosting','Random Forrest\n',np.round_(dectree.score(X_test,y_test),2),np.round_(bag.score(X_test,y_test),2),np.round_(boost.score(X_test,y_test),2),np.round_(rf.score(X_test,y_test),2),'\nTraining error\n',np.round_(dectree.score(train_x,train_y),2),np.round_(bag.score(train_x,train_y),2),np.round_(boost.score(train_x,train_y),2),np.round_(rf.score(train_x,train_y),2))
print('RF cross-val error:\n',1-rf.oob_score_)
print('Bagging cross-val error:\n',1-bag.oob_score_)
print(pd.DataFrame(rf.feature_importances_,index=df.columns[:-1],columns=['Mean Decrease in Gini']))


