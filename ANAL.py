import csv
import math
import pandas as pd
import re
import collections
from nltk.stem import PorterStemmer

# FILE READING
# each row is of the format:
#                   [ book , verse ]
file = open('train_and_dev.tsv')
file = csv.reader(file, delimiter="\t")
file_lst = [row for row in file]

print(len(file_lst))

file_lst = []
with open('train_and_dev.tsv','r') as f:
    for line in f.readlines():
        (corpus_id,text) = line.split("\t",1)
        file_lst.append((corpus_id,text))
print('Zori')
print(len(file_lst))


# WORD FREQUENCY ANALYSIS
# 1. Preprocess as usual (lowercasing? stemming?...)
# 2. Count words
# 3. Normalize by document length
# 4. Average across all documents

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

len(file_lst)
file_lst_preprocessed = [(a,preprocess(b)) for (a,b) in file_lst]

# print(file_lst_preprocessed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                       2. COUNT WORDS
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dict_OT_words_count = {}
num_ot_docs = 0
for pair in file_lst_preprocessed:
    if pair[0] == 'OT':
        num_ot_docs = num_ot_docs+1
        for word in pair[1].split(' '):
            if word not in dict_OT_words_count.keys():
                dict_OT_words_count[word] = 1
            else:
                dict_OT_words_count[word] = dict_OT_words_count.get(word) + 1
OT_keys = dict_OT_words_count.keys()
size_ot_dict = len(dict_OT_words_count.keys())

dict_NT_words_count = {}
num_nt_docs = 0
for pair in file_lst_preprocessed:
    if pair[0] == 'NT':
        num_nt_docs = num_nt_docs + 1
        for word in pair[1].split(' '):
            if word not in dict_NT_words_count.keys():
                dict_NT_words_count[word] = 1
            else:
                dict_NT_words_count[word] = dict_NT_words_count.get(word) + 1
NT_keys = dict_NT_words_count.keys()
size_nt_dict = len(dict_NT_words_count.keys())

dict_Quran_words_count = {}
num_quran_docs = 0
for pair in file_lst_preprocessed:
    if pair[0] == 'Quran':
        num_quran_docs = num_quran_docs + 1
        for word in pair[1].split(' '):
            if word not in dict_Quran_words_count.keys():
                dict_Quran_words_count[word] = 1
            else:
                dict_Quran_words_count[word] = dict_Quran_words_count.get(word) + 1
quran_keys = dict_Quran_words_count.keys()
size_quran_dict = len(dict_Quran_words_count.keys())


all_keys = []
for key in OT_keys:
    all_keys.append(key)
for key in NT_keys:
    all_keys.append(key)
for key in quran_keys:
    all_keys.append(key)


# Complete dictionary
complete_dict = {}
for word in all_keys:
    if word not in complete_dict.keys():
        if word in dict_OT_words_count.keys():
            complete_dict[word] = {'OT' : dict_OT_words_count.get(word,0)}
        elif word in dict_NT_words_count.keys():
            complete_dict[word] = {'NT' : dict_NT_words_count.get(word,0)}
        elif word in dict_Quran_words_count.keys():
            complete_dict[word] = {'Quran': dict_Quran_words_count.get(word,0)}
    else:
        docs = list(complete_dict.get(word).keys())
        if 'OT' not in docs:
            complete_dict.get(word)['OT'] = dict_OT_words_count.get(word,0)
        if 'NT' not in docs:
            complete_dict.get(word)['NT'] = dict_NT_words_count.get(word,0)
        if 'Quran' not in docs:
            complete_dict.get(word)['Quran'] = dict_Quran_words_count.get(word,0)

# print('Big Dict')
# print(complete_dict.get('histori',0))
# print('OT')
# print(dict_OT_words_count.get('histori',0))
# print('NT')
# print(dict_NT_words_count.get('histori',0))
# print('Quran')
# print(dict_Quran_words_count.get('histori',0))

# word : book
def mutual_information(word,book):
    x = complete_dict.get(word)
    n = len(file_lst_preprocessed)
    n1_ = complete_dict.get(word).get('OT',0) + complete_dict.get(word).get('NT',0) + complete_dict.get(word).get('Quran',0)
    n_1 = return_size_book(book)
    n_0 = n - n_1
    n0_ = n - n1_
    n11 = complete_dict.get(word).get(book,0)
    n01 = n_1 - n11
    n10 = n1_ - n11
    n00 = n0_ - n01

    # CALCULATIONS
    first_log_arg = (n*n11) / (n1_*n_1)
    first_term = (n11/n) * math.log(first_log_arg,2) if n11 != 0 else 0

    second_log_arg = (n*n01) / (n0_*n_1)
    second_term = (n01/n) * math.log(second_log_arg,2) if n01 != 0 else 0

    third_log_arg = (n * n10) / (n1_*n_0)
    third_term = (n10/n) * math.log(third_log_arg,2) if n10 != 0 else 0

    fourth_log_arg = (n * n00) / (n0_*n_0)
    fourth_term = (n00/n) * math.log(fourth_log_arg,2) if n00 != 0 else 0

    result = first_term+second_term+third_term+fourth_term
    return result

def chi_squared(word,book):
    x = complete_dict.get(word)
    n = len(file_lst_preprocessed)
    n1_ = complete_dict.get(word).get('OT', 0) + complete_dict.get(word).get('NT', 0) + complete_dict.get(word).get(
        'Quran', 0)
    n_1 = return_size_book(book)
    n_0 = n - n_1
    n0_ = n - n1_
    n11 = complete_dict.get(word).get(book, 0)
    n01 = n_1 - n11
    n10 = n1_ - n11
    n00 = n0_ - n01

    numerator = n * (n11*n00 - n10*n01)**2
    denominator = n_1 * n1_ * n_0 * n0_ if n_1!=0 or n1_!=0 or n_0!=0 or n0_!=0 else 1
    result = numerator / denominator

    return result

def return_size_book(book):
    if book == 'Quran':
        return num_quran_docs
    elif book == 'NT':
        return num_nt_docs
    elif book == 'OT':
        return num_ot_docs

def return_size_other(book):
    if book == 'Quran':
        return num_nt_docs+num_ot_docs
    elif book == 'NT':
        return num_ot_docs+num_quran_docs
    elif book == 'OT':
        return num_nt_docs+num_quran_docs

def return_size_other_word(word,book):
    if book == 'Quran':
        return complete_dict.get(word).get('NT',0)+complete_dict.get(word).get('OT',0)
    elif book == 'NT':
        return complete_dict.get(word).get('Quran',0)+complete_dict.get(word).get('OT',0)
    elif book == 'OT':
        return complete_dict.get(word).get('NT',0)+complete_dict.get(word).get('Quran',0)


def controller_CS():
    results_CS = {}
    for word in complete_dict.keys():
        word_book_score = {}
        word_book_score['OT'] = chi_squared(word,'OT')
        word_book_score['NT'] = chi_squared(word, 'NT')
        word_book_score['Quran'] = chi_squared(word, 'Quran')
        results_CS[word] = word_book_score
    return results_CS


def controller_MI():
    results_MI = {}
    for word in complete_dict.keys():
        # print(word)
        word_book_score = {}
        word_book_score['OT'] = mutual_information(word,'OT')
        word_book_score['NT'] = mutual_information(word, 'NT')
        word_book_score['Quran'] = mutual_information(word, 'Quran')
        results_MI[word] = word_book_score
    return results_MI


def generate_output(results_MI_or_CS,chi,mi):
    if chi == True and mi == False:
        # Quran
        f = open("quran_chi.csv", "w+")
        quran_tuples = sorted([(word,results_MI_or_CS.get(word).get('Quran')) for word in results_MI_or_CS.keys()], key=lambda x : x[1])
        quran_tuples.reverse()
        [f.write(a+','+str(b)+'\n') for (a,b) in quran_tuples]

        # OT
        f = open("NT_chi.csv", "w+")
        ot_tuples = sorted([(word,results_MI_or_CS.get(word).get('OT')) for word in results_MI_or_CS.keys()], key=lambda x : x[1])
        ot_tuples.reverse()
        [f.write(a+','+str(b)+'\n') for (a,b) in ot_tuples]

        # NT
        f = open("OT_chi.csv", "w+")
        nt_tuples = sorted([(word,results_MI_or_CS.get(word).get('NT')) for word in results_MI_or_CS.keys()], key=lambda x : x[1])
        nt_tuples.reverse()
        [f.write(a+','+str(b)+'\n') for (a,b) in nt_tuples]

    elif chi == False and mi == True:
        # Quran
        f = open("quran_mi.csv", "w+")
        quran_tuples = sorted([(word, results_MI_or_CS.get(word).get('Quran')) for word in results_MI_or_CS.keys()],
                              key=lambda x: x[1])
        quran_tuples.reverse()
        [f.write(a + ',' + str(b) + '\n') for (a, b) in quran_tuples]

        # OT
        f = open("NT_mi.csv", "w+")
        ot_tuples = sorted([(word, results_MI_or_CS.get(word).get('OT')) for word in results_MI_or_CS.keys()],
                           key=lambda x: x[1])
        ot_tuples.reverse()
        [f.write(a + ',' + str(b) + '\n') for (a, b) in ot_tuples]

        # NT
        f = open("OT_mi.csv", "w+")
        nt_tuples = sorted([(word, results_MI_or_CS.get(word).get('NT')) for word in results_MI_or_CS.keys()],
                           key=lambda x: x[1])
        nt_tuples.reverse()
        [f.write(a + ',' + str(b) + '\n') for (a, b) in nt_tuples]


def controller():
    results_CS = controller_CS()
    generate_output(results_CS,True,False)
    results_MI = controller_MI()
    generate_output(results_MI, False, True)


controller()











