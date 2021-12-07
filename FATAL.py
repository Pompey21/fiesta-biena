import csv
import math
import pandas as pd
import re
import collections
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from gensim.corpora.dictionary import Dictionary
from nltk.stem.porter import *
import scipy
from sklearn.svm import SVC, LinearSVC

# TIMESTAMP
datetime_beginning = datetime.now()
print(datetime_beginning)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                       1. FILE READING
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# each row is of the format:
#                   [ book , verse ]
file_lst = []
with open('train_and_dev.tsv','r') as f:
    for line in f.readlines():
        (corpus_id,text) = line.split("\t",1)
        file_lst.append((corpus_id,text))

file_test = []
with open('test.tsv','r') as f_test:
    for line in f_test.readlines():
        (corpus_id, text) = line.split("\t", 1)
        file_test.append((corpus_id, text))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                       2. PREPROCESSING
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
    This is the main preprocessing method, which calls all other 
    methods in this sub-component.
"""
def preprocess(text):
    text = tokenisation(text)
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
# def stop_words(sentance):
#     stop_words = open("stop-words.txt", "r").read()
#     stop_words = set(stop_words.split('\n'))
#
#     sentance_lst = sentance.split()
#     clean_sentance_lst = []
#
#     for word in sentance_lst:
#         if word not in stop_words:
#             clean_sentance_lst.append(word)
#     sentance = ' '.join(clean_sentance_lst)
#     return sentance

"""
------------------
    STEMMING
------------------ 
"""
# def stemming(sentance):
#     ps = PorterStemmer()
#     sentance_lst = sentance.split()
#     sentance = ' '.join([ps.stem(x) for x in sentance_lst])
#     return sentance

file_lst_preprocessed = [(preprocess(b),a) for (a,b) in file_lst]
test_file_processed = [(preprocess(b),a) for (a,b) in file_test]
"""
    Apply the steps in the text classification lab to this new dataset in order to get your baseline model: 
    extract BOW features and train an SVM classifier with c=1000 to predict the labels (i.e., which corpus a text belongs to). 
    Note that the input data format is slightly different this time, but you will still need to convert to BOW features. 
    You may reuse your code from the lab. 
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                     3. DOC -> BOW FEATURES
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
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
# TRAIN DEV FILE
data_np = np.array(file_lst_preprocessed)
X,y = data_np[:,0],data_np[:,1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

# 1. Find all the unique terms, and give each of them a unique ID (starting from 0 to the number of terms)
all_docs_train = [sentance.split() for sentance in X_train]
all_docs_test = [sentance.split() for sentance in X_test]

# model can only be trained on the vocabulary from the training set!
train_vocab = set([word for sentance in all_docs_train for word in sentance])
word2id = {}
for word_id,word in enumerate(train_vocab):
    word2id[word] = word_id

# and do the same for the categories
cat2id = {}
for cat_id,cat in enumerate(set(y_train)):
    cat2id[cat] = cat_id

y_train = [cat2id[cat] for cat in y_train]
X_train = convert_to_bow_matrix(all_docs_train,word2id)

y_test = [cat2id[cat] for cat in y_test]
X_test = convert_to_bow_matrix(all_docs_test,word2id)

# TEST FILE
test_data_np = np.array(test_file_processed)
test_file_y = [cat2id[cat] for cat in test_data_np[:,1]]
test_file_X = convert_to_bow_matrix(test_data_np[:,0],word2id)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                     4. TRAIN MODEL
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

model = LinearSVC(C=1000)
# then train the model!
model.fit(X_train,y_train)


y_train_predictions = model.predict(X_train)
y_test_predictions = model.predict(X_test)

test_file_y_predictions = model.predict(test_file_X)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                  5. PERFORMANCE ESTIMATION
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
    Compute the precision, recall, and f1-score for each of the 3 classes, as well as the macro-averaged precision, 
    recall, and f1-score across all three classes. Only train the system on your training split, but evaluate it 
    (compute these metrics) on both the training and development splits that you created 
    (i.e., don't train on documents from the development set). 
"""
from sklearn.metrics import classification_report

y_train = np.array(y_train)
class_rep_train = classification_report(y_train,y_train_predictions,output_dict=True)

print(class_rep_train)

y_test = np.array(y_test)
class_rep_test = classification_report(y_test,y_test_predictions,output_dict=True)

# TEST FILE
test_file_y_predictions = np.array(test_file_y_predictions)
class_rep_test_file = classification_report(test_file_y,test_file_y_predictions,output_dict=True)


print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                     6. GENERATING OUTPUT
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


# query_mean_output = [str(count_1 + 1), 'mean', str(round(p_10_mean / (count_2 + 1), 3)),
#                                  str(round(r_50_mean / (count_2 + 1), 3)),
#                                  str(round(r_precission_mean / (count_2 + 1), 3)),
#                                  str(round(ap_mean / (count_2 + 1), 3)), str(round(ndcg10_mean / (count_2 + 1), 3)),
#                                  str(round(ndcg20_mean / (count_2 + 1), 3))]
#             query_mean_output_str = ','.join(query_mean_output)
#             output.write(query_mean_output_str + '\n')

"""
    Identify 3 instances from the development set that the baseline system labels incorrectly. In your report, 
    start a new section called "Classification" and provide these 3 examples and your hypotheses about why these 
    were classified incorrectly. 
"""

"""
    Based on those 3 examples and any others you want to inspect from the development set, try to improve the results 
    of your classifier (you should have already experimented with ways to do this in the lab). You may change the 
    preprocessing, feature selection (e.g., only using the top N features with the highest MI scores), change the SVM 
    parameters, etc. You should create a system that improves over the baseline when evaluted on the development set. 
    However, remember that your final goal is to build a system that will work well on the test set, which is a randomly 
    held-out set of documents from the original corpora. 
"""

"""
    Six days before the deadline, the test set will be released. Without making any further changes to your baseline or 
    improved models, train on your training set and evaluate on the new test set you just collected. Report all of your 
    results in a file called classification.csv with the following format: 
"""


"""
    In your report on this assignment, in the "Classification" section, please explain how you managed to improve the 
    performance compared to the baseline system, and mention how much gain in the Macro-F1 score you could achieve with 
    your improved method when evaluted on the dev set, and how much gain on the test set. Why did you make the changes 
    that you did?
    - Note: it is okay if your test results are different from the development set results, but if the difference is 
    significant, please discuss why you think that is the case in your report. 
"""

datetime_end = datetime.now()
print(datetime_end)






