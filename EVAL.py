import csv
import math

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


eval = Eval('qrels.csv','system_results.csv')
eval.generate_output()

"""
    Statistical analysis of the systems - whether they are statistically significantly better
"""
from scipy import stats
# P@10
sys1 = [0.4,0.3,0.0,0.6,0.2,0.7,0.2,0.6,0.9,0.0]
sys2 = [0.1,0.0,0.0,0.6,0.1,0.4,0.3,0.1,0.5,0.1]
sys3 = [0.3,0.6,0.0,0.8,0.3,0.4,0.0,0.8,0.8,0.1]
sys4 = [0.0,0.1,0.0,0.3,0.1,0.1,0.0,0.0,0.2,0.0]
sys5 = [0.3,0.6,0.0,0.8,0.3,0.4,0.0,0.8,0.8,0.1]
sys6 = [0.3,0.6,0.0,0.8,0.3,0.4,0.0,0.8,0.8,0.1]

print('P@10 significance: ')
x_3_1 = stats.ttest_ind(sys3, sys1)
print(x_3_1)

x_5_1 = stats.ttest_ind(sys5, sys1)
print(x_5_1)

print()
# R@50
sys1 = [0.667,1.0,1.0,0.875,0.429,1.0,0.667,1.0,0.9,0.8]
sys2 = [0.667,1.0,1.0,0.875,0.429,1.0,1.0,1.0,0.9,0.8]

print('R@50 significance: ')
y_2_1 = stats.ttest_ind(sys2, sys1)
print(y_2_1)


print()
# r - precission
sys1 = [0.167,0.25,0.0,0.7,0.286,0.75,0.33,0.625,0.9,0.0]
sys3 = [0.5,0.625,0.0,0.7,0.143,0.417,0.0,1.0,0.9,0.2]

z_3_1 = stats.ttest_ind(sys3,sys1)

print('r-precision significance: ')
print(z_3_1)

print()
# average precision
sys3 = [0.518,0.75,0.056,0.69,0.104,0.465,0.0,1.0,0.756,0.174]
sys6 = [0.56,0.615,0.056,0.69,0.104,0.465,0.0,1.0,0.784,0.174]

i_3_6 = stats.ttest_ind(sys3,sys6)

print('average-precision significance: ')
print(i_3_6)

print()
# nDCG@10
sys6 = [0.646,0.695,0.0,0.622,0.233,0.132,0.0,0.722,0.533,0.417]
sys3 = [0.66,0.832,0.0,0.684,0.233,0.132,0.0,0.78,0.464,0.417]

j_3_1 = stats.ttest_ind(sys3,sys6)

print('nDCG@10 signficance: ')
print(j_3_1)


print()
# nDCG@20
sys3 = [0.733,0.897,0.24,0.704,0.233,0.449,0.0,0.78,0.584,0.488]
sys6 = [0.719,0.759,0.24,0.651,0.233,0.449,0.0,0.722,0.641,0.488]

k_3_6 = stats.ttest_ind(sys3,sys6)

print('nDCG@20 significance: ')
print(k_3_6)
