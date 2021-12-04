import csv
import math

# we have 10 queries
# which documents are relevant to which query
file_qrels = open('qrels.csv')
csvreader_qrels = csv.reader(file_qrels)
rows_qrels = [row for row in csvreader_qrels]
rows_qrels = rows_qrels[1:]

# create dict
# KEYS : Number of Query
# VALUES : Relevant documents for each query
qrels_dict = {}
for elem in rows_qrels:
    if elem[0] not in qrels_dict.keys():
        qrels_dict[elem[0]] = [(elem[1],elem[2])]
    else:
        qrels_dict.get(elem[0]).append((elem[1],elem[2]))


# we have 6 systems
# which documents were retrieved by which system
file_system_results = open('system_results.csv')
csvreader_system_results = csv.reader(file_system_results)
rows_system_results = [row for row in csvreader_system_results]
rows_system_results = rows_system_results[1:]

# create dict
# KEYS : Number of System
# VALUES : (query_number,doc_number,rank_of_doc,score)
system_results_dict = {}
for elem in rows_system_results:
    if elem[0] not in system_results_dict.keys():
        system_results_dict[elem[0]] = [(elem[1],elem[2],elem[3],elem[4])]
    else:
        # qrels_dict[elem[0]] = qrels_dict.get(elem[0]).append((elem[1],elem[2]))
        system_results_dict.get(elem[0]).append((elem[1],elem[2],elem[3],elem[4]))

# print(system_results_dict)

# create dict
# KEYS_1 : Number of System
# KEYS_2 : Number of Query
# VALUES : Documents (doc_number,rank_of_doc,score)
for system in system_results_dict.keys():
    query_dict = {}
    for query in system_results_dict.get(system):
        if query[0] not in query_dict.keys():
            query_dict[query[0]] = [(query[1],query[2],query[3])]
        else:
            query_dict.get(query[0]).append((query[1],query[2],query[3]))
    system_results_dict[system] = query_dict


# P@10
# consider only 10 elements for each query for each system
def p_10():
    system_queries_precision = []
    for system in system_results_dict.keys():
        queries_precission = []
        for query in system_results_dict.get(system).keys():
            # system_results_dict.get(system)[query] = system_results_dict.get(system).get(query)[:10]
            precision_counter = 0
            for doc in system_results_dict.get(system).get(query)[:10]:
                if doc[0] in tuples_first(qrels_dict.get(query)):
                    precision_counter = precision_counter+1
            query_precission = (query,precision_counter/10)
            # print(query_precission)
            queries_precission.append(query_precission)
        system_queries_precision.append(queries_precission)
    return system_queries_precision

# returns the first element of a tuple : (a,b) -> a
def tuples_first(lst_tuples):
    return [tpl[0] for tpl in lst_tuples]

def tuple_second(lst_tuples):
    return [tpl[1] for tpl in lst_tuples]

def fill_list(lst,size):
    for i in range(size-len(lst)):
        lst.append(0)
    return lst

# system_queries_precision = p_10()
# print('P-10')
# print(system_queries_precision)


# R@50
# consider only 50 elements for each query for each system
def r_50():
    system_queries_recall = []
    for system in system_results_dict.keys():
        queries_recall = []
        for query in system_results_dict.get(system).keys():
            # system_results_dict.get(system)[query] = system_results_dict.get(system).get(query)[:10]
            recall_counter = 0
            for doc in system_results_dict.get(system).get(query)[:50]:
                if doc[0] in tuples_first(qrels_dict.get(query)):
                    recall_counter = recall_counter + 1
            query_recall = (query, recall_counter / len(tuples_first(qrels_dict.get(query))))
            # print(query_recall)
            queries_recall.append(query_recall)
        system_queries_recall.append(queries_recall)
    return system_queries_recall

# system_queries_recall = r_50()
# print('R-50')
# print(system_queries_recall)

# R-Precission
# consider only 'r' elements for each query for each system
# 'r' is different to each document
def r_precission():
    system_queries_precision = []
    for system in system_results_dict.keys():
        queries_precission = []
        for query in system_results_dict.get(system).keys():
            # system_results_dict.get(system)[query] = system_results_dict.get(system).get(query)[:10]
            precision_counter = 0
            r = len(tuples_first(qrels_dict.get(query)))
            for doc in system_results_dict.get(system).get(query)[:r]:
                if doc[0] in tuples_first(qrels_dict.get(query)):
                    precision_counter = precision_counter+1
            query_precission = (query,precision_counter/r)
            # print(query_precission)
            queries_precission.append(query_precission)
        system_queries_precision.append(queries_precission)
    return system_queries_precision

# system_queries_r_precission = r_precission()
# print('R-Precission')
# print(system_queries_r_precission[0])


# Average-Precision
def average_precission():
    system_queries_average_precission = []
    for system in system_results_dict.keys():
        # print('*******************************************************************')
        # print(f"System : {system}")
        queries_avg_precision = []
        for query in system_results_dict.get(system).keys():
            # print(f'Query: {query}')
            number_docs_per_query = len(qrels_dict.get(query))
            correct_query_count = 0
            query_precisions = []
            for count,doc in enumerate(system_results_dict.get(system).get(query)):
                # print(doc)
                if number_docs_per_query == 0:
                    # print('eeeeeeoooooo')
                    break
                if doc[0] in tuples_first(qrels_dict.get(query)):

                    correct_query_count = correct_query_count + 1
                    # print(f'correct query count : {correct_query_count}')
                    # print(doc[0])
                    # print(f'count : {count+1}')

                    precission = correct_query_count/(count+1)
                    # print(f'precission : {precission}')
                    query_precisions.append(precission)
                    number_docs_per_query = number_docs_per_query-1
            # print(query_precisions)
            if len(query_precisions) != 0:
                queries_avg_precision.append(sum(query_precisions)/len(qrels_dict.get(query)))
            else:
                queries_avg_precision.append(0)
        # print(queries_avg_precision)
        system_queries_average_precission.append(queries_avg_precision)
    return system_queries_average_precission


# print('Average Precission')
# system_queries_average_precission = average_precission()
# print('###################################################'
#       '###################################################')
# print(system_queries_average_precission)

def find_relevance(lst_tuples,id):
    dict_id_rel = {}
    for elem in lst_tuples:
        if elem[0] not in dict_id_rel.keys():
            dict_id_rel[elem[0]] = elem[1]
    return dict_id_rel.get(id)

def ideal_dcg(reverse_ordered_lst):
    idcg_lst = []
    for count,elem in enumerate(reverse_ordered_lst):
        if count == 0:
            idcg_lst.append(elem)
        else:
            result = elem / math.log(count+1,2)
            idcg_lst.append(result)
    return idcg_lst

def cumulative_dg(lst_dg):
    lst_dcg = []
    for i in range(len(lst_dg)):
        # print(sum(lst_dg[:i]))
        lst_dcg.append(sum(lst_dg[:i+1]))
    return lst_dcg

def divide_lst_lst(lst_1,lst_2):
    result = []
    for i in range(len(lst_1)):
        if lst_2[i] == 0:
            result.append(0)
        else:
            result.append(lst_1[i]/lst_2[i])
    return result

# nDCG@10: normalized discount cumulative gain at cutoff 10
def nDCG_10():
    system_queries_DG = []
    for system in system_results_dict.keys():
        queries_NDG = []
        for query in system_results_dict.get(system).keys():
            dg_lst = []
            ideal_order_source = fill_list([int(a) for a in tuple_second(qrels_dict.get(query))],10)
            # print(ideal_order_source)
            for count,doc in enumerate(system_results_dict.get(system).get(query)[:10]):
                if doc[0] in tuples_first(qrels_dict.get(query)):
                    rel = int(find_relevance(qrels_dict.get(query),doc[0]))
                    rank = int(doc[1])
                    if rank == 1:
                        dg = rel
                        dg_lst.append(dg)
                    else:
                        dg = rel / math.log(rank,2)
                        dg_lst.append(dg)
                else:
                    dg = 0
                    rel = 0
                    dg_lst.append(dg)
            # print(dg_lst)
            dcg = cumulative_dg(dg_lst)
            # print(dcg)
            ideal_order_source.sort()
            ideal_order_source.reverse()
            # print(ideal_order_source)
            idcg = cumulative_dg(ideal_dcg(ideal_order_source))
            # print(idcg)

            ndcg = divide_lst_lst(dcg,idcg)
            # print(ndcg)
            result = ndcg[len(ndcg)-1]

            queries_NDG.append(result)
            # break
        # break
        system_queries_DG.append(queries_NDG)

    return system_queries_DG


# system_queries_DG = nDCG_10()
# print(system_queries_DG)


# nDCG@20: normalized discount cumulative gain at cutoff 20
def nDCG_20():
    system_queries_DG = []
    for system in system_results_dict.keys():
        queries_NDG = []
        for query in system_results_dict.get(system).keys():
            dg_lst = []
            ideal_order_source = fill_list([int(a) for a in tuple_second(qrels_dict.get(query))],20)
            # print(ideal_order_source)
            for count,doc in enumerate(system_results_dict.get(system).get(query)[:20]):
                if doc[0] in tuples_first(qrels_dict.get(query)):
                    rel = int(find_relevance(qrels_dict.get(query),doc[0]))
                    rank = int(doc[1])
                    if rank == 1:
                        dg = rel
                        dg_lst.append(dg)
                    else:
                        dg = rel / math.log(rank,2)
                        dg_lst.append(dg)
                else:
                    dg = 0
                    rel = 0
                    dg_lst.append(dg)
            # print(dg_lst)
            dcg = cumulative_dg(dg_lst)
            # print(dcg)
            ideal_order_source.sort()
            ideal_order_source.reverse()
            # print(ideal_order_source)
            idcg = cumulative_dg(ideal_dcg(ideal_order_source))
            # print(idcg)

            ndcg = divide_lst_lst(dcg,idcg)
            # print(ndcg)
            result = ndcg[len(ndcg)-1]

            queries_NDG.append(result)
            # break
        # break
        system_queries_DG.append(queries_NDG)

    return system_queries_DG


# system_queries_DG = nDCG_20()
# print(system_queries_DG)



p_10_var = p_10()

r_50_var = r_50()

r_precission_var = r_precission()

average_precission_var = average_precission()
print(average_precission_var)
nDCG_10_var = nDCG_10()
nDCG_20_var = nDCG_20()
# system_results_dict
# Generate output!
output = open("ir_eval.csv", "w+")
for count_1,system in enumerate(system_results_dict):
    system_output = []
    p_10_mean= 0
    r_50_mean = 0
    r_precission_mean = 0
    ap_mean = 0
    ndcg10_mean = 0
    ndcg20_mean = 0
    for count_2, query in enumerate(qrels_dict):
        query_output = [str(count_1+1),str(count_2+1),str(round(p_10_var[count_1][count_2][1],3)),str(round(r_50_var[count_1][count_2][1],3)),
                        str(round(r_precission_var[count_1][count_2][1],3)),str(round(average_precission_var[count_1][count_2],3)),
                        str(round(nDCG_10_var[count_1][count_2],1)),str(round(nDCG_20_var[count_1][count_2],3))]
        p_10_mean = p_10_mean + p_10_var[count_1][count_2][1]
        r_50_mean = r_50_mean + r_50_var[count_1][count_2][1]
        r_precission_mean = r_precission_mean + r_precission_var[count_1][count_2][1]
        ap_mean = ap_mean + average_precission_var[count_1][count_2]
        ndcg10_mean = ndcg10_mean + nDCG_10_var[count_1][count_2]
        ndcg20_mean = ndcg20_mean + nDCG_20_var[count_1][count_2]

        query_output_str = ','.join(query_output)
        output.write(query_output_str + '\n')
    query_mean_output = [str(count_1+1),'mean',str(round(p_10_mean/(count_2+1),3)),str(round(r_50_mean/(count_2+1),3)),str(round(r_precission_mean/(count_2+1),3)),
                         str(round(ap_mean/(count_2+1),3)),str(round(ndcg10_mean/(count_2+1),3)),str(round(ndcg20_mean/(count_2+1),3))]
    query_mean_output_str = ','.join(query_mean_output)
    output.write(query_mean_output_str + '\n')










