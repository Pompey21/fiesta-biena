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

# converting the data into numpy format
data_np = np.array(file_lst_preprocessed)

X,y = data_np[:,0],data_np[:,1]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.10, random_state=42)

"""
    Apply the steps in the text classification lab to this new dataset in order to get your baseline model: 
    extract BOW features and train an SVM classifier with c=1000 to predict the labels (i.e., which corpus a text belongs to). 
    Note that the input data format is slightly different this time, but you will still need to convert to BOW features. 
    You may reuse your code from the lab. 
"""
from gensim.corpora.dictionary import Dictionary
from nltk.stem.porter import *

# 1. Find all the unique terms, and give each of them a unique ID (starting from 0 to the number of terms)
all = [sentance.split() for sentance in [a for (a,b) in file_lst_preprocessed]]

# words are numbered -> IDs
id2rowd = Dictionary(all)
print(len(id2rowd))

common_corpus = [id2rowd.doc2bow(text) for text in all]
print(common_corpus)

# creating a count matrix
# 1. make a list of dicts (one dict per doc)
doc_dict = {key: {key: 0 for key in range(len(id2rowd))} for key in range(len(common_corpus))}
lst_dicts = [doc_dict.get(doc).get(word[0])==word[1] for doc in common_corpus for word in doc]














