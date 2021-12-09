import csv
import math
from scipy import stats
import math
import re
from nltk.stem import PorterStemmer
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
from sklearn.utils import shuffle


# ======================================================================================================================
# ======================================================================================================================
"""
                                                TASK 1: IR EVALUATION
                                            """
# ======================================================================================================================
# ======================================================================================================================

class Eval:
    def __init__(self,file_qrels,file_sys_res):
        # Information Representation
        self.file_qrels = self.create_dict_qrels(file_qrels)
        self.file_sys_res = self.create_dict_sys_res(file_sys_res)

        # Evaluation Mterics
        self.p_10 = self.p_10(self.file_sys_res)
        self.r_50 = self.r_50(self.file_sys_res)
        self.r_precission = self.r_precission(self.file_sys_res)
        self.average_precission = self.average_precission(self.file_sys_res,self.file_qrels)
        self.nDCG_10 = self.nDCG_10(self.file_sys_res,self.file_qrels)
        self.nDCG_20 = self.nDCG_20(self.file_sys_res,self.file_qrels)

    def create_dict_qrels(self,file_qrels_file):
        # we have 10 queries
        # which documents are relevant to which query
        file_qrels = open(file_qrels_file)
        csvreader_qrels = csv.reader(file_qrels)
        rows_qrels = [row for row in csvreader_qrels]
        rows_qrels = rows_qrels[1:]

        # KEYS : Number of Query
        # VALUES : Relevant documents for each query
        qrels_dict = {}
        for elem in rows_qrels:
            if elem[0] not in qrels_dict.keys():
                qrels_dict[elem[0]] = [(elem[1], elem[2])]
            else:
                qrels_dict.get(elem[0]).append((elem[1], elem[2]))
        return qrels_dict
    def create_dict_sys_res(self,file_sys_res_file):
        # we have 6 systems
        # which documents were retrieved by which system
        file_system_results = open(file_sys_res_file)
        csvreader_system_results = csv.reader(file_system_results)
        rows_system_results = [row for row in csvreader_system_results]
        rows_system_results = rows_system_results[1:]

        # create dict
        # KEYS : Number of System
        # VALUES : (query_number,doc_number,rank_of_doc,score)
        system_results_dict = {}
        for elem in rows_system_results:
            if elem[0] not in system_results_dict.keys():
                system_results_dict[elem[0]] = [(elem[1], elem[2], elem[3], elem[4])]
            else:
                # qrels_dict[elem[0]] = qrels_dict.get(elem[0]).append((elem[1],elem[2]))
                system_results_dict.get(elem[0]).append((elem[1], elem[2], elem[3], elem[4]))

        # create dict
        # KEYS_1 : Number of System
        # KEYS_2 : Number of Query
        # VALUES : Documents (doc_number,rank_of_doc,score)
        for system in system_results_dict.keys():
            query_dict = {}
            for query in system_results_dict.get(system):
                if query[0] not in query_dict.keys():
                    query_dict[query[0]] = [(query[1], query[2], query[3])]
                else:
                    query_dict.get(query[0]).append((query[1], query[2], query[3]))
            system_results_dict[system] = query_dict

        return system_results_dict

    # P@10
    # consider only 10 elements for each query for each system
    def p_10(self,system_results_dict):
        system_queries_precision = []
        for system in system_results_dict.keys():
            queries_precission = []
            for query in system_results_dict.get(system).keys():
                precision_counter = 0
                for doc in system_results_dict.get(system).get(query)[:10]:
                    if doc[0] in self.tuples_first(self.file_qrels.get(query)):
                        precision_counter = precision_counter + 1
                query_precission = (query, precision_counter / 10)
                queries_precission.append(query_precission)
            system_queries_precision.append(queries_precission)
        return system_queries_precision

    # R@50
    # consider only 50 elements for each query for each system
    def r_50(self,system_results_dict):
        system_queries_recall = []
        for system in system_results_dict.keys():
            queries_recall = []
            for query in system_results_dict.get(system).keys():
                recall_counter = 0
                for doc in system_results_dict.get(system).get(query)[:50]:
                    if doc[0] in self.tuples_first(self.file_qrels.get(query)):
                        recall_counter = recall_counter + 1
                query_recall = (query, recall_counter / len(self.tuples_first(self.file_qrels.get(query))))
                queries_recall.append(query_recall)
            system_queries_recall.append(queries_recall)
        return system_queries_recall

    # R-Precission
    # consider only 'r' elements for each query for each system -> 'r' is different to each document
    def r_precission(self,system_results_dict):
        system_queries_precision = []
        for system in system_results_dict.keys():
            queries_precission = []
            for query in system_results_dict.get(system).keys():
                precision_counter = 0
                r = len(self.tuples_first(self.file_qrels.get(query)))
                for doc in system_results_dict.get(system).get(query)[:r]:
                    if doc[0] in self.tuples_first(self.file_qrels.get(query)):
                        precision_counter = precision_counter + 1
                query_precission = (query, precision_counter / r)
                queries_precission.append(query_precission)
            system_queries_precision.append(queries_precission)
        return system_queries_precision

    # Average-Precision
    def average_precission(self,system_results_dict,qrels_dict):
        system_queries_average_precission = []
        for system in system_results_dict.keys():
            queries_avg_precision = []
            for query in system_results_dict.get(system).keys():
                number_docs_per_query = len(qrels_dict.get(query))
                correct_query_count = 0
                query_precisions = []
                for count, doc in enumerate(system_results_dict.get(system).get(query)):
                    if number_docs_per_query == 0:
                        break
                    if doc[0] in self.tuples_first(qrels_dict.get(query)):
                        correct_query_count = correct_query_count + 1
                        precission = correct_query_count / (count + 1)
                        query_precisions.append(precission)
                        number_docs_per_query = number_docs_per_query - 1
                if len(query_precisions) != 0:
                    queries_avg_precision.append(sum(query_precisions) / len(qrels_dict.get(query)))
                else:
                    queries_avg_precision.append(0)
            system_queries_average_precission.append(queries_avg_precision)
        return system_queries_average_precission

    # nDCG@10: normalized discount cumulative gain at cutoff 10
    def nDCG_10(self,system_results_dict,qrels_dict):
        system_queries_DG = []
        for system in system_results_dict.keys():
            queries_NDG = []
            for query in system_results_dict.get(system).keys():
                dg_lst = []
                ideal_order_source = self.fill_list([int(a) for a in self.tuple_second(qrels_dict.get(query))], 10)
                for count, doc in enumerate(system_results_dict.get(system).get(query)[:10]):
                    if doc[0] in self.tuples_first(qrels_dict.get(query)):
                        rel = int(self.find_relevance(qrels_dict.get(query), doc[0]))
                        rank = int(doc[1])
                        if rank == 1:
                            dg = rel
                            dg_lst.append(dg)
                        else:
                            dg = rel / math.log(rank, 2)
                            dg_lst.append(dg)
                    else:
                        dg = 0
                        rel = 0
                        dg_lst.append(dg)
                dcg = self.cumulative_dg(dg_lst)
                ideal_order_source.sort()
                ideal_order_source.reverse()
                idcg = self.cumulative_dg(self.ideal_dcg(ideal_order_source))
                ndcg = self.divide_lst_lst(dcg, idcg)
                result = ndcg[len(ndcg) - 1]
                queries_NDG.append(result)
            system_queries_DG.append(queries_NDG)

        return system_queries_DG

    # nDCG@20: normalized discount cumulative gain at cutoff 20
    def nDCG_20(self,system_results_dict,qrels_dict):
        system_queries_DG = []
        for system in system_results_dict.keys():
            queries_NDG = []
            for query in system_results_dict.get(system).keys():
                dg_lst = []
                ideal_order_source = self.fill_list([int(a) for a in self.tuple_second(qrels_dict.get(query))], 20)
                for count, doc in enumerate(system_results_dict.get(system).get(query)[:20]):
                    if doc[0] in self.tuples_first(qrels_dict.get(query)):
                        rel = int(self.find_relevance(qrels_dict.get(query), doc[0]))
                        rank = int(doc[1])
                        if rank == 1:
                            dg = rel
                            dg_lst.append(dg)
                        else:
                            dg = rel / math.log(rank, 2)
                            dg_lst.append(dg)
                    else:
                        dg = 0
                        rel = 0
                        dg_lst.append(dg)
                dcg = self.cumulative_dg(dg_lst)
                ideal_order_source.sort()
                ideal_order_source.reverse()
                idcg = self.cumulative_dg(self.ideal_dcg(ideal_order_source))
                ndcg = self.divide_lst_lst(dcg, idcg)
                result = ndcg[len(ndcg) - 1]
                queries_NDG.append(result)
            system_queries_DG.append(queries_NDG)

        return system_queries_DG


    # HELPER FUNCTIONS
    # returns the first element of a tuple : (a,b) -> a
    def tuples_first(self,lst_tuples):
        return [tpl[0] for tpl in lst_tuples]
    def tuple_second(self,lst_tuples):
        return [tpl[1] for tpl in lst_tuples]
    def fill_list(self,lst, size):
        for i in range(size - len(lst)):
            lst.append(0)
        return lst
    def find_relevance(self,lst_tuples,id):
        dict_id_rel = {}
        for elem in lst_tuples:
            if elem[0] not in dict_id_rel.keys():
                dict_id_rel[elem[0]] = elem[1]
        return dict_id_rel.get(id)
    def ideal_dcg(self,reverse_ordered_lst):
        idcg_lst = []
        for count, elem in enumerate(reverse_ordered_lst):
            if count == 0:
                idcg_lst.append(elem)
            else:
                result = elem / math.log(count + 1, 2)
                idcg_lst.append(result)
        return idcg_lst
    def cumulative_dg(self,lst_dg):
        lst_dcg = []
        for i in range(len(lst_dg)):
            # print(sum(lst_dg[:i]))
            lst_dcg.append(sum(lst_dg[:i + 1]))
        return lst_dcg
    def divide_lst_lst(self,lst_1, lst_2):
        result = []
        for i in range(len(lst_1)):
            if lst_2[i] == 0:
                result.append(0)
            else:
                result.append(lst_1[i] / lst_2[i])
        return result

    # GENERATE OUTPUT
    def generate_output(self,):
        output = open("ir_eval.csv", "w+")
        header = 'system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20'
        output.write(header + '\n')
        for count_1, system in enumerate(self.file_sys_res):
            system_output = []
            p_10_mean = 0
            r_50_mean = 0
            r_precission_mean = 0
            ap_mean = 0
            ndcg10_mean = 0
            ndcg20_mean = 0
            for count_2, query in enumerate(self.file_qrels):
                query_output = [str(count_1 + 1), str(count_2 + 1), str(round(self.p_10[count_1][count_2][1], 3)),
                                str(round(self.r_50[count_1][count_2][1], 3)),
                                str(round(self.r_precission[count_1][count_2][1], 3)),
                                str(round(self.average_precission[count_1][count_2], 3)),
                                str(round(self.nDCG_10[count_1][count_2], 3)),
                                str(round(self.nDCG_20[count_1][count_2], 3))]
                p_10_mean = p_10_mean + self.p_10[count_1][count_2][1]
                r_50_mean = r_50_mean + self.r_50[count_1][count_2][1]
                r_precission_mean = r_precission_mean + self.r_precission[count_1][count_2][1]
                ap_mean = ap_mean + self.average_precission[count_1][count_2]
                ndcg10_mean = ndcg10_mean + self.nDCG_10[count_1][count_2]
                ndcg20_mean = ndcg20_mean + self.nDCG_20[count_1][count_2]

                query_output_str = ','.join(query_output)
                output.write(query_output_str + '\n')
            query_mean_output = [str(count_1 + 1), 'mean', str(round(p_10_mean / (count_2 + 1), 3)),
                                 str(round(r_50_mean / (count_2 + 1), 3)),
                                 str(round(r_precission_mean / (count_2 + 1), 3)),
                                 str(round(ap_mean / (count_2 + 1), 3)), str(round(ndcg10_mean / (count_2 + 1), 3)),
                                 str(round(ndcg20_mean / (count_2 + 1), 3))]
            query_mean_output_str = ','.join(query_mean_output)
            output.write(query_mean_output_str + '\n')

"""
    Calling the Eval module.
"""
eval = Eval('qrels.csv','system_results.csv')
eval.generate_output()

"""
    Statistical analysis of the systems - whether they are statistically significantly better
"""
# # P@10
# sys1 = [0.4,0.3,0.0,0.6,0.2,0.7,0.2,0.6,0.9,0.0]
# sys2 = [0.1,0.0,0.0,0.6,0.1,0.4,0.3,0.1,0.5,0.1]
# sys3 = [0.3,0.6,0.0,0.8,0.3,0.4,0.0,0.8,0.8,0.1]
# sys4 = [0.0,0.1,0.0,0.3,0.1,0.1,0.0,0.0,0.2,0.0]
# sys5 = [0.3,0.6,0.0,0.8,0.3,0.4,0.0,0.8,0.8,0.1]
# sys6 = [0.3,0.6,0.0,0.8,0.3,0.4,0.0,0.8,0.8,0.1]
#
# print('P@10 significance: ')
# x_3_1 = stats.ttest_ind(sys3, sys1)
# print(x_3_1)
#
# x_5_1 = stats.ttest_ind(sys5, sys1)
# print(x_5_1)
#
# print()
# # R@50
# sys1 = [0.667,1.0,1.0,0.875,0.429,1.0,0.667,1.0,0.9,0.8]
# sys2 = [0.667,1.0,1.0,0.875,0.429,1.0,1.0,1.0,0.9,0.8]
#
# print('R@50 significance: ')
# y_2_1 = stats.ttest_ind(sys2, sys1)
# print(y_2_1)
#
#
# print()
# # r - precission
# sys1 = [0.167,0.25,0.0,0.7,0.286,0.75,0.33,0.625,0.9,0.0]
# sys3 = [0.5,0.625,0.0,0.7,0.143,0.417,0.0,1.0,0.9,0.2]
#
# z_3_1 = stats.ttest_ind(sys3,sys1)
#
# print('r-precision significance: ')
# print(z_3_1)
#
# print()
# # average precision
# sys3 = [0.518,0.75,0.056,0.69,0.104,0.465,0.0,1.0,0.756,0.174]
# sys6 = [0.56,0.615,0.056,0.69,0.104,0.465,0.0,1.0,0.784,0.174]
#
# i_3_6 = stats.ttest_ind(sys3,sys6)
#
# print('average-precision significance: ')
# print(i_3_6)
#
# print()
# # nDCG@10
# sys6 = [0.646,0.695,0.0,0.622,0.233,0.132,0.0,0.722,0.533,0.417]
# sys3 = [0.66,0.832,0.0,0.684,0.233,0.132,0.0,0.78,0.464,0.417]
#
# j_3_1 = stats.ttest_ind(sys3,sys6)
#
# print('nDCG@10 signficance: ')
# print(j_3_1)
#
#
# print()
# # nDCG@20
# sys3 = [0.733,0.897,0.24,0.704,0.233,0.449,0.0,0.78,0.584,0.488]
# sys6 = [0.719,0.759,0.24,0.651,0.233,0.449,0.0,0.722,0.641,0.488]
#
# k_3_6 = stats.ttest_ind(sys3,sys6)
#
# print('nDCG@20 significance: ')
# print(k_3_6)





# ======================================================================================================================
# ======================================================================================================================
"""
                                                TASK 2: TEXT ANALYSIS
                                            """
# ======================================================================================================================
# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
#           <><><><><><><><><> <><><><><><><><><> <><><><><><><><><> <><><><><><><><><> <><><><><><><><><>
#                || || ||       || ||  Mutual Information and Chi Squared  || ||        || || ||
#           <><><><><><><><><> <><><><><><><><><> <><><><><><><><><> <><><><><><><><><> <><><><><><><><><>
# ----------------------------------------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                       1. FILE READING
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# each row is of the format:
#                   [ book , verse ]
def read_file(filename):
    file_lst = []
    with open(filename,'r') as f:
        for line in f.readlines():
            (corpus_id,text) = line.split("\t",1)
            file_lst.append((corpus_id,text))
    return file_lst

file_lst = read_file(filename='train_and_dev.tsv')
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
    text = stemming(stop_words(tokenisation(text)))
    return text

def case_folding(sentance):
    sentance = sentance.lower()
    return sentance

def numbers(sentance):
    numbers = list(range(0, 10))
    numbers_strs = [str(x) for x in numbers]

    for number in numbers_strs:
        sentance = sentance.replace(number, '')
    return sentance

# splitting at not alphabetic characers
def tokenisation(sentance):
    sentance_list = list(set(re.split('\W+', sentance)))
    sentance_list_new = []
    for word in sentance_list:
        word_new = case_folding(numbers(word))
        sentance_list_new.append(word_new)
    return ' '.join(sentance_list_new)

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

def stemming(sentance):
    ps = PorterStemmer()
    sentance_lst = sentance.split()
    sentance = ' '.join([ps.stem(x) for x in sentance_lst])
    return sentance


file_lst_preprocessed = [(a,preprocess(b)) for (a,b) in file_lst]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#            3. COUNT WORDS - Generating Dictionaries
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
    Generating dictionaries for each of the corpora as well as the whole corpus:
        1. Old Testimony
        2. New Testimony
        3. Quran
"""
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                   4. MUTUAL INFORMATION
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# word : book
def mutual_information(word,book):
    x = complete_dict.get(word)
    n = len(file_lst_preprocessed)
    n1_ = complete_dict.get(word).get('OT',0) + complete_dict.get(word).get('NT',0) + complete_dict.get(word).get('Quran',0)
    n_1 = return_size_book(book)
    n_0 = n - n_1
    n0_ = n - n1_
    n11 = complete_dict.get(word).get(book,0)
    if n11 == 0:
        return 0
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                       4. CHI SQUARED
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def chi_squared(word,book):
    x = complete_dict.get(word)
    n = len(file_lst_preprocessed)
    n1_ = complete_dict.get(word).get('OT', 0) + complete_dict.get(word).get('NT', 0) + complete_dict.get(word).get(
        'Quran', 0)
    n_1 = return_size_book(book)
    n_0 = n - n_1
    n0_ = n - n1_
    n11 = complete_dict.get(word).get(book, 0)
    if n11 == 0:
        return 0
    n01 = n_1 - n11
    n10 = n1_ - n11
    n00 = n0_ - n01

    numerator = n * (n11*n00 - n10*n01)**2
    denominator = n_1 * n1_ * n_0 * n0_
    result = numerator / denominator if denominator !=0 else 0

    return result

def controller_CS():
    results_CS = {}
    for word in complete_dict.keys():
        # if word == 'jesu':
        #     print('kkkkk')
        word_book_score = {}
        word_book_score['OT'] = chi_squared(word,'OT')
        word_book_score['NT'] = chi_squared(word, 'NT')
        word_book_score['Quran'] = chi_squared(word, 'Quran')
        results_CS[word] = word_book_score
    return results_CS

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                     5. HELPER FUNCTIONS
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                     6. GENERATE OUTPUT
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def generate_output(results_MI_or_CS,chi,mi):
    if chi == True and mi == False:
        # Quran
        f = open("quran_chi.csv", "w+")
        quran_tuples = sorted([(word,round(results_MI_or_CS.get(word).get('Quran'),3)) for word in results_MI_or_CS.keys()],
                              key=lambda x : x[1])
        quran_tuples.reverse()
        [f.write(a+','+str(b)+'\n') for (a,b) in quran_tuples]

        # OT
        f = open("OT_chi.csv", "w+")
        ot_tuples = sorted([(word,round(results_MI_or_CS.get(word).get('OT'),3)) for word in results_MI_or_CS.keys()], key=lambda x : x[1])
        ot_tuples.reverse()
        [f.write(a+','+str(b)+'\n') for (a,b) in ot_tuples]

        # NT
        f = open("NT_chi.csv", "w+")
        nt_tuples = sorted([(word,round(results_MI_or_CS.get(word).get('NT'),3)) for word in results_MI_or_CS.keys()], key=lambda x : x[1])
        nt_tuples.reverse()
        [f.write(a+','+str(b)+'\n') for (a,b) in nt_tuples]

    elif chi == False and mi == True:
        # Quran
        f = open("quran_mi.csv", "w+")
        quran_tuples = sorted([(word, round(results_MI_or_CS.get(word).get('Quran'),3)) for word in results_MI_or_CS.keys()],
                              key=lambda x: x[1])
        quran_tuples.reverse()
        [f.write(a + ',' + str(b) + '\n') for (a, b) in quran_tuples]

        # OT
        f = open("OT_mi.csv", "w+")
        ot_tuples = sorted([(word, round(results_MI_or_CS.get(word).get('OT'),3)) for word in results_MI_or_CS.keys()],
                           key=lambda x: x[1])
        ot_tuples.reverse()
        [f.write(a + ',' + str(b) + '\n') for (a, b) in ot_tuples]

        # NT
        f = open("NT_mi.csv", "w+")
        nt_tuples = sorted([(word, round(results_MI_or_CS.get(word).get('NT'),3)) for word in results_MI_or_CS.keys()],
                           key=lambda x: x[1])
        nt_tuples.reverse()
        [f.write(a + ',' + str(b) + '\n') for (a, b) in nt_tuples]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#            6. CONTROLLER - Calls all other methods
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def controller():
    results_CS = controller_CS()
    generate_output(results_CS,True,False)
    results_MI = controller_MI()
    generate_output(results_MI, False, True)
    print()

controller()



# ----------------------------------------------------------------------------------------------------------------------
#           <><><><><><><><><> <><><><><><><><><> <><><><><><><><><> <><><><><><><><><> <><><><><><><><><>
#                   || || ||       || ||  Latent Dirichlet Allocation || ||        || || ||
#           <><><><><><><><><> <><><><><><><><><> <><><><><><><><><> <><><><><><><><><> <><><><><><><><><>
# ----------------------------------------------------------------------------------------------------------------------
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#          print('*******************************************')
#                            print('LDA')
#          print('*******************************************')
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.stem.porter import *
"""
    Run LDA on the entire set of verses from ALL corpora together.
    Set k=20 topics and inspect the results.
"""

all = [sentance.split() for sentance in [b for (a,b) in file_lst_preprocessed]]
quran = [sentance.split() for sentance in [b for (a,b) in file_lst_preprocessed if a=='Quran']]
nt = [sentance.split() for sentance in [b for (a,b) in file_lst_preprocessed if a=='NT']]
ot = [sentance.split() for sentance in [b for (a,b) in file_lst_preprocessed if a=='OT']]

id2rowd = Dictionary(all)
common_corpus = [id2rowd.doc2bow(text) for text in all]


lda = LdaModel(common_corpus, num_topics=20, passes=30, random_state=53)
"""
    1. For each corpus, compute the average score for each topic by summing the document-topic probability for each
    2. document in that corpus and dividing by the total number of documents in the corpus.
"""
"""
   3. Then, for each corpus, you should identify the topic that has the highest average score (3 topics in total).
   4. For each of those three topics, find the top 10 tokens with highest probability of belonging to that topic.
"""
print()
# Quran
# computing topic probabilities for each document in a Quran corpus
topic_quran = [lda.get_document_topics(id2rowd.doc2bow(text)) for text in quran]

average_probs_quran = {key: 0 for key in range(20)}
num_docs_quran = len(topic_quran)
for doc in topic_quran:
    for topic in doc:
        score = topic[1] / num_docs_quran
        average_probs_quran[topic[0]] = average_probs_quran.get(topic[0])+score

print(average_probs_quran)

highest_topics_quran = [a for (a,b) in sorted(list(average_probs_quran.items()), key=lambda x: x[1], reverse=True)]
words_topic_1_quran = [(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(highest_topics_quran[0],topn=10)]
words_topic_2_quran = [(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(highest_topics_quran[1],topn=10)]
words_topic_3_quran = [(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(highest_topics_quran[2],topn=10)]

print('Quran')
# print(highest_topics_quran[0])
# print(words_topic_1_quran)
# print(id2rowd.get(highest_topics_quran[1]))
# print(words_topic_2_quran)
# print(id2rowd.get(highest_topics_quran[2]))
# print(words_topic_3_quran)

for topic in highest_topics_quran:
    print()
    print(f'topic: {topic}')
    print([(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(topic,topn=10)])

# ----------------------------------------------------------------------------------------------------------------------
print()
# NT
# computing topic probabilities for each document in a NT corpus
topic_nt = [lda.get_document_topics(id2rowd.doc2bow(text)) for text in nt]
# print(topic_nt)

average_probs_nt = {key: 0 for key in range(20)}
num_docs_nt = len(topic_nt)
for doc in topic_nt:
    for topic in doc:
        score = topic[1] / num_docs_nt
        average_probs_nt[topic[0]] = average_probs_nt.get(topic[0])+score

print(average_probs_nt)

highest_topics_nt = [a for (a,b) in sorted(list(average_probs_nt.items()), key=lambda x: x[1], reverse=True)]
words_topic_1_nt = [(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(highest_topics_nt[0],topn=10)]
words_topic_2_nt = [(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(highest_topics_nt[1],topn=10)]
words_topic_3_nt = [(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(highest_topics_nt[2],topn=10)]

print('NT')
# print(id2rowd.get(highest_topics_nt[0]))
# print(words_topic_1_nt)
# print(id2rowd.get(highest_topics_nt[1]))
# print(words_topic_2_nt)
# print(id2rowd.get(highest_topics_nt[2]))
# print(words_topic_3_nt)

for topic in highest_topics_nt:
    print()
    print(f'topic: {topic}')
    print([(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(topic,topn=10)])

# ----------------------------------------------------------------------------------------------------------------------
print()
# OT
# computing topic probabilities for each document in a OT corpus
topic_ot = [lda.get_document_topics(id2rowd.doc2bow(text)) for text in ot]
# print(topic_nt)

average_probs_ot = {key: 0 for key in range(20)}
num_docs_ot = len(topic_ot)
for doc in topic_ot:
    for topic in doc:
        score = topic[1] / num_docs_ot
        average_probs_ot[topic[0]] = average_probs_ot.get(topic[0])+score

print(average_probs_ot)

highest_topics_ot = [a for (a,b) in sorted(list(average_probs_ot.items()), key=lambda x: x[1], reverse=True)]
words_topic_1_ot = [(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(highest_topics_ot[0],topn=10)]
words_topic_2_ot = [(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(highest_topics_ot[1],topn=10)]
words_topic_3_ot = [(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(highest_topics_ot[2],topn=10)]

print('OT')
# print(id2rowd.get(highest_topics_ot[0]))
# print(words_topic_1_ot)
# print(id2rowd.get(highest_topics_ot[1]))
# print(words_topic_2_ot)
# print(id2rowd.get(highest_topics_ot[2]))
# print(words_topic_3_ot)

for topic in highest_topics_ot:
    print()
    print(f'topic: {topic}')
    print([(id2rowd.get(id),prob) for (id,prob) in lda.get_topic_terms(topic,topn=10)])











# ======================================================================================================================
# ======================================================================================================================
"""
                                            TASK 3: TEXT CLASSIFICATION
                                        """
# ======================================================================================================================
# ======================================================================================================================


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
    all.sort(key=lambda x:float(x[1]))
    all = [elem[0] for elem in all[:5000]]
    #return all

    sentance_lst = sentance.split()
    clean_sentance_lst = []

    for word in sentance_lst:
        if word not in all:
            clean_sentance_lst.append(word)
    sentance = ' '.join(clean_sentance_lst)
    return sentance


# CHI SQ words - top 5000
def top_chi_words(sentance):
    file_lst_ot = []
    with open('OT_chi.csv', 'r') as f:
        for line in f.readlines():
            (corpus_id, text) = line[:-1].split(',')
            file_lst_ot.append((corpus_id, text))
    file_lst_ot = file_lst_ot[:5000]

    file_lst_nt = []
    with open('NT_chi.csv', 'r') as f:
        for line in f.readlines():
            (corpus_id, text) = line[:-1].split(',')
            file_lst_nt.append((corpus_id, text))
    file_lst_nt = file_lst_nt[:5000]

    file_lst_quran = []
    with open('NT_chi.csv', 'r') as f:
        for line in f.readlines():
            (corpus_id, text) = line[:-1].split(',')
            file_lst_quran.append((corpus_id, text))
    file_lst_quran = file_lst_quran[:5000]

    all = file_lst_nt+file_lst_ot+file_lst_quran
    all.sort(key=lambda x: float(x[1]))
    all = [elem[0] for elem in all[:5000]]
    #return all

    sentance_lst = sentance.split()
    clean_sentance_lst = []

    for word in sentance_lst:
        if word not in all:
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

# TF-IDF implementation below - since it did not improve it is commented out!
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

"""
    TRAIN DEV FILE
"""

data_np = np.array(file_lst_preprocessed)
X,y = shuffle(data_np[:,0],data_np[:,1])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.9,random_state=8321)
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
"""
    The following improvements were implemented:
    
        1. Data split (training and testing set) was set to 0.9
        
        2. C parameter in SVC model was set to 10
        
        3. Normalised BOW matrix formation
"""


file_lst_preprocessed_improved = [(preprocess_improves(b),a) for (a,b) in file_lst]

data_np_improved = np.array(file_lst_preprocessed_improved)
X_improved,y_improved = shuffle(data_np_improved[:,0],data_np_improved[:,1])
X_train_improved,X_test_improved,y_train_improved,y_test_improved = train_test_split(X_improved,y_improved,test_size=0.9,random_state=8321)
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

model_improved = SVC(C=10)
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
output = open("classification.csv", "w+")
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
wrong_classified_index = []
for i in range(len(y_test)):
    if y_test.tolist()[i]!=y_test_predictions.tolist()[i]:
        wrong_classified_index.append((y_test.tolist()[i],y_test_predictions.tolist()[i],X_proper_text[i]))


print()



datetime_end = datetime.now()
print(datetime_end)


