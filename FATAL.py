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
from sklearn.metrics import classification_report

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


print()
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

def preprocess_improves(text):
    text = tokenisation(top_mi_words(text))
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
    sentance_list = list((re.split('\W+', sentance)))
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
test_file_processed = [(preprocess(b),a) for (a,b) in file_test]


# MI words - top 5000
def top_mi_words(sentance):
    file_lst_ot = []
    with open('OT_mi.csv', 'r') as f:
        for line in f.readlines():
            (corpus_id, text) = line[:-1].split(',')
            file_lst_ot.append((corpus_id, text))
    file_lst_ot = file_lst_ot[:5000]

    file_lst_nt = []
    with open('NT_mi.csv', 'r') as f:
        for line in f.readlines():
            (corpus_id, text) = line[:-1].split(',')
            file_lst_nt.append((corpus_id, text))
    file_lst_nt = file_lst_nt[:5000]

    file_lst_quran = []
    with open('NT_mi.csv', 'r') as f:
        for line in f.readlines():
            (corpus_id, text) = line[:-1].split(',')
            file_lst_quran.append((corpus_id, text))
    file_lst_quran = file_lst_quran[:5000]

    all = file_lst_nt+file_lst_ot+file_lst_quran
    all = all[:5000]
    #return all

    sentance_lst = sentance.split()
    clean_sentance_lst = []

    for word in sentance_lst:
        if word not in all:
            clean_sentance_lst.append(word)
    sentance = ' '.join(clean_sentance_lst)
    return sentance



# top_mi_words()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                     3. DOC -> BOW FEATURES
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
def convert_to_bow_matrix(preprocessed_data, word2id, tfidf):
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
            X[doc_id, word2id.get(word, oov_index)] += 1#/len(doc)
    if tfidf==False:
        return X

    for doc_id, doc in enumerate(preprocessed_data):
        for word in list(set(doc)):
            # default is 0, so just add to the count for this word in this doc
            # if the word is oov, increment the oov_index
            X[doc_id, word2id.get(word, oov_index)] += 1/len(doc)

    return X

    # df = np.count_nonzero(X.toarray(), axis=0)
    # for doc_id, doc in enumerate(preprocessed_data):
    #     for word in list(set(doc)):
    #         # default is 0, so just add to the count for this word in this doc
    #         # if the word is oov, increment the oov_index
    #         tf = X[doc_id, word2id.get(word, oov_index)]
    #         N = len(X)
    #         result = (1+math.log(tf,10))*(math.log(N/df[word2id.get(word, oov_index)],10))
    #         X[doc_id, word2id.get(word, oov_index)] = result
    # return X

# TRAIN DEV FILE
data_np = np.array(file_lst_preprocessed)
X,y = data_np[:,0],data_np[:,1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.6,random_state=8321)
X_proper_text = X_test.copy()
# 1. Find all the unique terms, and give each of them a unique ID (starting from 0 to the number of terms)
all_docs_train = [sentance.split() for sentance in X_train]
all_docs_test = [sentance.split() for sentance in X_test]

train_vocab = set([word for sentance in all_docs_train for word in sentance])
word2id = {}
for word_id,word in enumerate(train_vocab):
    word2id[word] = word_id

# and do the same for the categories
cat2id = {}
for cat_id,cat in enumerate(set(y_train)):
    cat2id[cat] = cat_id

y_train = [cat2id[cat] for cat in y_train]
X_train = convert_to_bow_matrix(all_docs_train,word2id,False)

y_test = [cat2id[cat] for cat in y_test]
X_test = convert_to_bow_matrix(all_docs_test,word2id,False)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                     4. TRAIN MODEL
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

model = SVC(C=1000)
# then train the model!
model.fit(X_train,y_train)

y_train_predictions = model.predict(X_train)
y_test_predictions = model.predict(X_test)

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
y_train = np.array(y_train)
class_rep_train = classification_report(y_train,y_train_predictions,output_dict=True)

# print(class_rep_train)

y_test = np.array(y_test)
class_rep_test = classification_report(y_test,y_test_predictions,output_dict=True)


print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                        6. TEST FILE
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_data_np = np.array(test_file_processed)
all_docs_test_file = [sentance.split() for sentance in test_data_np[:,0]]

test_file_y = [cat2id[cat] for cat in test_data_np[:,1]]
test_file_X = convert_to_bow_matrix(all_docs_test_file,word2id,False)
test_file_y_predictions = model.predict(test_file_X)


test_file_y_predictions = np.array(test_file_y_predictions)
class_rep_test_file = classification_report(test_file_y,test_file_y_predictions,output_dict=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                        7. IMPROVED
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
file_lst_preprocessed_improved = [(preprocess_improves(b),a) for (a,b) in file_lst]

data_np_improved = np.array(file_lst_preprocessed_improved)
X_improved,y_improved = data_np_improved[:,0],data_np_improved[:,1]
X_train_improved,X_test_improved,y_train_improved,y_test_improved = train_test_split(X_improved,y_improved,test_size=0.6,random_state=8321)
# 1. Find all the unique terms, and give each of them a unique ID (starting from 0 to the number of terms)
all_docs_train_improved = [sentance.split() for sentance in X_train_improved]
all_docs_test_improved = [sentance.split() for sentance in X_test_improved]

train_vocab_improved = set([word for sentance in all_docs_train_improved for word in sentance])
word2id_improved = {}
for word_id,word in enumerate(train_vocab_improved):
    word2id_improved[word] = word_id

cat2id_improved = {}
for cat_id,cat in enumerate(set(y_train_improved)):
    cat2id_improved[cat] = cat_id

y_train_improved = [cat2id_improved[cat] for cat in y_train_improved]
X_train_improved = convert_to_bow_matrix(all_docs_train_improved,word2id_improved,True)

y_test_improved = [cat2id_improved[cat] for cat in y_test_improved]
X_test_improved = convert_to_bow_matrix(all_docs_test_improved,word2id_improved,True)

model_improved = SVC(C=20)
model_improved.fit(X_train_improved,y_train_improved)

y_train_predictions_improved = model_improved.predict(X_train_improved)
y_test_predictions_improved = model_improved.predict(X_test_improved)


y_train_improved = np.array(y_train_improved)
class_rep_train_improved = classification_report(y_train_improved,y_train_predictions_improved,output_dict=True)


y_test_improved = np.array(y_test_improved)
class_rep_test_improved = classification_report(y_test_improved,y_test_predictions_improved,output_dict=True)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #                   Test File Improved
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_file_processed_improved = [(preprocess_improves(b),a) for (a,b) in file_test]
test_data_np_improved = np.array(test_file_processed_improved)

all_docs_test_file_improved = [sentance.split() for sentance in test_data_np_improved[:,0]]

test_file_y_improved = [cat2id_improved[cat] for cat in test_data_np_improved[:,1]]
test_file_X_improved = convert_to_bow_matrix(all_docs_test_file_improved,word2id_improved,True)
test_file_y_predictions_improved = np.array(model_improved.predict(test_file_X_improved))


class_rep_test_file_improved = classification_report(test_file_y_improved,test_file_y_predictions_improved,output_dict=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                     8. GENERATING OUTPUT
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
system=['baseline','baseline','baseline','improved','improved','improved']
split=['train','dev','test','train','dev','test']
# system_and_split = list(set([(sys,spl) for sys in system for spl in split]))
system_and_split=[('baseline','train'),('baseline','dev'),('baseline','test'),
                  ('improved','train'),('improved','dev'),('improved','test')]
output = open("classification_run2.csv", "w+")
first_line = 'system,split,p-quran,r-quran,f-quran,p-ot,r-ot,f-ot,p-nt,r-nt,f-nt,p-macro,r-macro,f-macro\n'
output.write(first_line)
for count,pair in enumerate(system_and_split):
    line_output = pair[0]+','+pair[1]
    if pair[0]=='baseline':
        if pair[1]=='train':
            # Quran
            p_quran = str(round(class_rep_train.get('1').get('precision'),3))
            r_quran = str(round(class_rep_train.get('1').get('recall'),3))
            f1_quran = str(round(class_rep_train.get('1').get('f1-score'),3))
            # OT
            p_ot = str(round(class_rep_train.get('0').get('precision'),3))
            r_ot = str(round(class_rep_train.get('0').get('recall'),3))
            f1_ot = str(round(class_rep_train.get('0').get('f1-score'),3))
            # NT
            p_nt = str(round(class_rep_train.get('2').get('precision'),3))
            r_nt = str(round(class_rep_train.get('2').get('recall'),3))
            f1_nt = str(round(class_rep_train.get('2').get('f1-score'),3))
            # Overall -> Macro
            p_macro = str(round(class_rep_train.get('macro avg').get('precision'),5))
            r_macro = str(round(class_rep_train.get('macro avg').get('recall'),5))
            f1_macro = str(round(class_rep_train.get('macro avg').get('f1-score'),5))

            line_output = line_output+','+p_quran+','+r_quran+','+f1_quran+','+p_ot+','+r_ot+','+f1_ot+','+p_nt+','+r_nt+','+f1_nt+','+p_macro+','+r_macro+','+f1_macro+'\n'

            output.write(line_output)

        elif pair[1] == 'dev':
            # Quran
            p_quran = str(round(class_rep_test.get('1').get('precision'),3))
            r_quran = str(round(class_rep_test.get('1').get('recall'),3))
            f1_quran = str(round(class_rep_test.get('1').get('f1-score'),3))
            # OT
            p_ot = str(round(class_rep_test.get('0').get('precision'),3))
            r_ot = str(round(class_rep_test.get('0').get('recall'),3))
            f1_ot = str(round(class_rep_test.get('0').get('f1-score'),3))
            # NT
            p_nt = str(round(class_rep_test.get('2').get('precision'),3))
            r_nt = str(round(class_rep_test.get('2').get('recall'),3))
            f1_nt = str(round(class_rep_test.get('2').get('f1-score'),3))
            # Overall -> Macro
            p_macro = str(round(class_rep_test.get('macro avg').get('precision'),5))
            r_macro = str(round(class_rep_test.get('macro avg').get('recall'),5))
            f1_macro = str(round(class_rep_test.get('macro avg').get('f1-score'),5))

            line_output = line_output + ',' + p_quran + ',' + r_quran + ',' + f1_quran + ','+p_ot + ',' + r_ot + ',' + f1_ot + ','+p_nt + ',' + r_nt + ',' + f1_nt + ',' + p_macro + ',' + r_macro + ',' + f1_macro + '\n'

            output.write(line_output)

        elif pair[1] == 'test':
            # Quran
            p_quran = str(round(class_rep_test_file.get('1').get('precision'),3))
            r_quran = str(round(class_rep_test_file.get('1').get('recall'),3))
            f1_quran = str(round(class_rep_test_file.get('1').get('f1-score'),3))
            # OT
            p_ot = str(round(class_rep_test_file.get('0').get('precision'),3))
            r_ot = str(round(class_rep_test_file.get('0').get('recall'),3))
            f1_ot = str(round(class_rep_test_file.get('0').get('f1-score'),3))
            # NT
            p_nt = str(round(class_rep_test_file.get('2').get('precision'),3))
            r_nt = str(round(class_rep_test_file.get('2').get('recall'),3))
            f1_nt = str(round(class_rep_test_file.get('2').get('f1-score'),3))
            # Overall -> Macro
            p_macro = str(round(class_rep_test_file.get('macro avg').get('precision'),5))
            r_macro = str(round(class_rep_test_file.get('macro avg').get('recall'),5))
            f1_macro = str(round(class_rep_test_file.get('macro avg').get('f1-score'),5))

            line_output = line_output + ',' + p_quran + ',' + r_quran + ',' + f1_quran + ','+p_ot + ',' + r_ot + ',' + f1_ot + ','+p_nt + ',' + r_nt + ',' + f1_nt + ',' + p_macro + ',' + r_macro + ',' + f1_macro + '\n'

            output.write(line_output)

    elif pair[0]=='improved':
        if pair[1]=='train':
            # Quran
            p_quran = str(round(class_rep_train_improved.get('1').get('precision'),3))
            r_quran = str(round(class_rep_train_improved.get('1').get('recall'),3))
            f1_quran = str(round(class_rep_train_improved.get('1').get('f1-score'),3))
            # OT
            p_ot = str(round(class_rep_train_improved.get('0').get('precision'),3))
            r_ot = str(round(class_rep_train_improved.get('0').get('recall'),3))
            f1_ot = str(round(class_rep_train_improved.get('0').get('f1-score'),3))
            # NT
            p_nt = str(round(class_rep_train_improved.get('2').get('precision'),3))
            r_nt = str(round(class_rep_train_improved.get('2').get('recall'),3))
            f1_nt = str(round(class_rep_train_improved.get('2').get('f1-score'),3))
            # Overall -> Macro
            p_macro = str(round(class_rep_train_improved.get('macro avg').get('precision'),5))
            r_macro = str(round(class_rep_train_improved.get('macro avg').get('recall'),5))
            f1_macro = str(round(class_rep_train_improved.get('macro avg').get('f1-score'),5))

            line_output = line_output+','+p_quran+','+r_quran+','+f1_quran+','+p_ot+','+r_ot+','+f1_ot+','+p_nt+','+r_nt+','+f1_nt+','+p_macro+','+r_macro+','+f1_macro+'\n'

            output.write(line_output)

        elif pair[1] == 'dev':
            # Quran
            p_quran = str(round(class_rep_test_improved.get('1').get('precision'),3))
            r_quran = str(round(class_rep_test_improved.get('1').get('recall'),3))
            f1_quran = str(round(class_rep_test_improved.get('1').get('f1-score'),3))
            # OT
            p_ot = str(round(class_rep_test_improved.get('0').get('precision'),3))
            r_ot = str(round(class_rep_test_improved.get('0').get('recall'),3))
            f1_ot = str(round(class_rep_test_improved.get('0').get('f1-score'),3))
            # NT
            p_nt = str(round(class_rep_test_improved.get('2').get('precision'),3))
            r_nt = str(round(class_rep_test_improved.get('2').get('recall'),3))
            f1_nt = str(round(class_rep_test_improved.get('2').get('f1-score'),3))
            # Overall -> Macro
            p_macro = str(round(class_rep_test_improved.get('macro avg').get('precision'),5))
            r_macro = str(round(class_rep_test_improved.get('macro avg').get('recall'),5))
            f1_macro = str(round(class_rep_test_improved.get('macro avg').get('f1-score'),5))

            line_output = line_output + ',' + p_quran + ',' + r_quran + ',' + f1_quran + ','+p_ot + ',' + r_ot + ',' + f1_ot + ','+p_nt + ',' + r_nt + ',' + f1_nt + ',' + p_macro + ',' + r_macro + ',' + f1_macro + '\n'

            output.write(line_output)

        elif pair[1] == 'test':
            # Quran
            p_quran = str(round(class_rep_test_file_improved.get('1').get('precision'),3))
            r_quran = str(round(class_rep_test_file_improved.get('1').get('recall'),3))
            f1_quran = str(round(class_rep_test_file_improved.get('1').get('f1-score'),3))
            # OT
            p_ot = str(round(class_rep_test_file_improved.get('0').get('precision'),3))
            r_ot = str(round(class_rep_test_file_improved.get('0').get('recall'),3))
            f1_ot = str(round(class_rep_test_file_improved.get('0').get('f1-score'),3))
            # NT
            p_nt = str(round(class_rep_test_file_improved.get('2').get('precision'),3))
            r_nt = str(round(class_rep_test_file_improved.get('2').get('recall'),3))
            f1_nt = str(round(class_rep_test_file_improved.get('2').get('f1-score'),3))
            # Overall -> Macro
            p_macro = str(round(class_rep_test_file_improved.get('macro avg').get('precision'),5))
            r_macro = str(round(class_rep_test_file_improved.get('macro avg').get('recall'),5))
            f1_macro = str(round(class_rep_test_file_improved.get('macro avg').get('f1-score'),5))

            line_output = line_output + ',' + p_quran + ',' + r_quran + ',' + f1_quran + ','+p_ot + ',' + r_ot + ',' + f1_ot + ','+p_nt + ',' + r_nt + ',' + f1_nt + ',' + p_macro + ',' + r_macro + ',' + f1_macro + '\n'

            output.write(line_output)
"""
    Identify 3 instances from the development set that the baseline system labels incorrectly. In your report, 
    start a new section called "Classification" and provide these 3 examples and your hypotheses about why these 
    were classified incorrectly. 
"""
joined = np.concatenate((y_test_predictions,y_test.T),axis=0)
wrong_classified_index = []
for i in range(len(y_test)):
    if y_test.tolist()[i]!=y_test_predictions.tolist()[i]:
        wrong_classified_index.append((y_test.tolist()[i],y_test_predictions.tolist()[i],X_proper_text[i]))


print()




"""
    Based on those 3 examples and any others you want to inspect from the development set, try to improve the results 
    of your classifier (you should have already experimented with ways to do this in the lab). You may change the 
    preprocessing, feature selection (e.g., only using the top N features with the highest MI scores), change the SVM 
    parameters, etc. You should create a system that improves over the baseline when evaluted on the development set. 
    However, remember that your final goal is to build a system that will work well on the test set, which is a randomly 
    held-out set of documents from the original corpora. 
"""

"""
    In your report on this assignment, in the "Classification" section, please explain how you managed to improve the 
    performance compared to the baseline system, and mention how much gain in the Macro-F1 score you could achieve with 
    your improved method when evaluted on the dev set, and how much gain on the test set. Why did you make the changes 
    that you did?
    - Note: it is okay if your test results are different from the development set results, but if the difference is 
    significant, please discuss why you think that is the case in your report. 
"""


"""
    1. change the preprocessing 
    2. feature selection (e.g., only using the top N features with the highest MI scores) 
    3. change the SVM parameters
"""







datetime_end = datetime.now()
print(datetime_end)






