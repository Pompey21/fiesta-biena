import csv
import math
import pandas as pd
import re
import collections
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.model_selection import train_test_split

# FILE READING
# each row is of the format:
#                   [ book , verse ]
file_lst = []
with open('train_and_dev.tsv','r') as f:
    for line in f.readlines():
        (corpus_id,text) = line.split("\t",1)
        file_lst.append((corpus_id,text))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                       1. PREPROCESSING
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
    This is the main preprocessing method, which calls all other 
    methods in this sub-component.
"""
def preprocess(text):
    text = stemming(stop_words(tokenisation(text)))
    return text
"""
---------------------    
    TOKENISATION
---------------------    
"""
"""
    Case Folding
"""
def case_folding(sentance):
    sentance = sentance.lower()
    return sentance

"""
    Numbers Handling
"""
def numbers(sentance):
    numbers = list(range(0, 10))
    numbers_strs = [str(x) for x in numbers]

    for number in numbers_strs:
        sentance = sentance.replace(number, '')
    return sentance

"""
    Tokenisation
"""
# splitting at not alphabetic characers
def tokenisation(sentance):
    sentance_list = list(set(re.split('\W+', sentance)))
    sentance_list_new = []
    for word in sentance_list:
        word_new = case_folding(numbers(word))
        sentance_list_new.append(word_new)
    return ' '.join(sentance_list_new)

"""
--------------------------    
    STOPWORD REMOVAL
--------------------------
"""
def stop_words(sentance):
    stop_words = open("stop-words.txt", "r").read()
    stop_words = set(stop_words.split('\n'))

    sentance_lst = sentance.split()
    clean_sentance_lst = []

    for word in sentance_lst:
        if word not in stop_words:
            clean_sentance_lst.append(word)
    sentance = ' '.join(clean_sentance_lst)
    return sentance

"""
------------------
    STEMMING
------------------ 
"""
def stemming(sentance):
    ps = PorterStemmer()
    sentance_lst = sentance.split()
    sentance = ' '.join([ps.stem(x) for x in sentance_lst])
    return sentance

file_lst_preprocessed = [(preprocess(b),a) for (a,b) in file_lst]

"""
    Apply the steps in the text classification lab to this new dataset in order to get your baseline model: 
    extract BOW features and train an SVM classifier with c=1000 to predict the labels (i.e., which corpus a text belongs to). 
    Note that the input data format is slightly different this time, but you will still need to convert to BOW features. 
    You may reuse your code from the lab. 
"""
from gensim.corpora.dictionary import Dictionary
from nltk.stem.porter import *
import scipy
from sklearn.svm import LinearSVC

data_np = np.array(file_lst_preprocessed)
X,y = data_np[:,0],data_np[:,1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)


# 1. Find all the unique terms, and give each of them a unique ID (starting from 0 to the number of terms)
all_docs_train = [sentance.split() for sentance in X_train]

train_vocab = set([word for sentance in all_docs_train for word in sentance])
print(len(train_vocab))


word2id = {}
for word_id,word in enumerate(train_vocab):
    word2id[word] = word_id
print('length word2id')
print(len(word2id))


# and do the same for the categories
cat2id = {}
for cat_id,cat in enumerate(set(y_train)):
    cat2id[cat] = cat_id
print('length cat2id')
print(len(cat2id))

def convert_to_bow_matrix(preprocessed_data, word2id):
    # matrix size is number of docs x vocab size + 1 (for OOV)
    matrix_size = (len(preprocessed_data), len(word2id) + 1)
    oov_index = len(word2id)
    # matrix indexed by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)

    # iterate through all documents in the dataset
    for doc_id, doc in enumerate(preprocessed_data):
        for word in doc:
            # default is 0, so just add to the count for this word in this doc
            # if the word is oov, increment the oov_index
            X[doc_id, word2id.get(word, oov_index)] += 1
    return X

y_train = [cat2id[cat] for cat in y_train]
print('length of y_train')
print(len(y_train))
X_train = convert_to_bow_matrix(all_docs_train,word2id)
print(X_train.shape)
model = LinearSVC(C=1000)
# then train the model!
model.fit(X_train,y_train)


y_train_predictions = model.predict(X_train)

def compute_accuracy(predictions, true_values):
    num_correct = 0
    num_total = len(predictions)
    for predicted,true in zip(predictions,true_values):
        if predicted==true:
            num_correct += 1
    return num_correct / num_total

accuracy = compute_accuracy(y_train_predictions,y_train)
print("Accuracy:",accuracy)











