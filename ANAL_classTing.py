class Analysis:
    def __init__(self,train_and_dev_file):
        self.file_lst_procesed = self.perform_preprocessing(self.read_file(train_and_dev_file))
        self.dict_OT_words_count = self.generate_OT_dict(self.file_lst_procesed)
        self.size_ot_dict = len(self.dict_OT_words_count)
        self.ot_keys = self.dict_OT_words_count.keys()
        self.dict_NT_words_count = self.generate_NT_dict(self.file_lst_procesed)
        self.size_nt_dict = len(self.dict_NT_words_count)
        self.nt_keys = self.dict_NT_words_count.keys()
        self.dict_Quran_words_count = self.generate_Quran_dict(self.file_lst_procesed)
        self.size_quran_dict = len(self.dict_Quran_words_count)
        self.quran_keys = self.dict_Quran_words_count.keys()

    # FILE READING
    # each row is of the format:
    #                   [ book , verse ]
    def read_file(self,train_dev_file):
        file_lst = []
        with open('train_and_dev.tsv', 'r') as f:
            for line in f.readlines():
                (corpus_id, text) = line.split("\t", 1)
                file_lst.append((corpus_id, text))

    # WORD FREQUENCY ANALYSIS
    # 1. Preprocess as usual (lowercasing? stemming?...)
    # 2. Count words
    # 3. Normalize by document length
    # 4. Average across all documents

    def perform_preprocessing(self,file_lst):
        file_lst_preprocessed = [(a, preprocess(b)) for (a, b) in file_lst]
        return file_lst_preprocessed

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #          *******************************************
    #                       1. PREPROCESSING
    #          *******************************************
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def preprocess(self,text):
        text = stemming(stop_words(tokenisation(text)))
        return text
    def case_folding(self,sentance):
        sentance = sentance.lower()
        return sentance
    def numbers(self,sentance):
        numbers = list(range(0, 10))
        numbers_strs = [str(x) for x in numbers]

        for number in numbers_strs:
            sentance = sentance.replace(number, '')
        return sentance
    # splitting at not alphabetic characers
    def tokenisation(self,sentance):
        sentance_list = list(set(re.split('\W+', sentance)))
        sentance_list_new = []
        for word in sentance_list:
            word_new = case_folding(numbers(word))
            sentance_list_new.append(word_new)
        return ' '.join(sentance_list_new)
    def stop_words(self,sentance):
        stop_words = open("stop-words.txt", "r").read()
        stop_words = set(stop_words.split('\n'))

        sentance_lst = sentance.split()
        clean_sentance_lst = []

        for word in sentance_lst:
            if word not in stop_words:
                clean_sentance_lst.append(word)
        sentance = ' '.join(clean_sentance_lst)
        return sentance
    def stemming(self,sentance):
        ps = PorterStemmer()
        sentance_lst = sentance.split()
        sentance = ' '.join([ps.stem(x) for x in sentance_lst])
        return sentance

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #          *******************************************
    #                       2. COUNT WORDS
    #          *******************************************
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def generate_OT_dict(self,file_lst_processed):
        dict_OT_words_count = {}
        num_ot_docs = 0
        for pair in file_lst_processed:
            if pair[0] == 'OT':
                num_ot_docs = num_ot_docs + 1
                for word in pair[1].split(' '):
                    if word not in dict_OT_words_count.keys():
                        dict_OT_words_count[word] = 1
                    else:
                        dict_OT_words_count[word] = dict_OT_words_count.get(word) + 1
        OT_keys = dict_OT_words_count.keys()
        size_ot_dict = len(dict_OT_words_count.keys())
        return dict_OT_words_count

    def generate_NT_dict(self,file_lst_processed):
        dict_NT_words_count = {}
        num_nt_docs = 0
        for pair in file_lst_processed:
            if pair[0] == 'NT':
                num_nt_docs = num_nt_docs + 1
                for word in pair[1].split(' '):
                    if word not in dict_NT_words_count.keys():
                        dict_NT_words_count[word] = 1
                    else:
                        dict_NT_words_count[word] = dict_NT_words_count.get(word) + 1
        NT_keys = dict_NT_words_count.keys()
        size_nt_dict = len(dict_NT_words_count.keys())
        return dict_NT_words_count

    def generate_Quran_dict(self,file_lst_processed):
        dict_Quran_words_count = {}
        num_quran_docs = 0
        for pair in file_lst_processed:
            if pair[0] == 'Quran':
                num_quran_docs = num_quran_docs + 1
                for word in pair[1].split(' '):
                    if word not in dict_Quran_words_count.keys():
                        dict_Quran_words_count[word] = 1
                    else:
                        dict_Quran_words_count[word] = dict_Quran_words_count.get(word) + 1
        quran_keys = dict_Quran_words_count.keys()
        size_quran_dict = len(dict_Quran_words_count.keys())
        return dict_Quran_words_count

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
                complete_dict[word] = {'OT': dict_OT_words_count.get(word, 0)}
            elif word in dict_NT_words_count.keys():
                complete_dict[word] = {'NT': dict_NT_words_count.get(word, 0)}
            elif word in dict_Quran_words_count.keys():
                complete_dict[word] = {'Quran': dict_Quran_words_count.get(word, 0)}
        else:
            docs = list(complete_dict.get(word).keys())
            if 'OT' not in docs:
                complete_dict.get(word)['OT'] = dict_OT_words_count.get(word, 0)
            if 'NT' not in docs:
                complete_dict.get(word)['NT'] = dict_NT_words_count.get(word, 0)
            if 'Quran' not in docs:
                complete_dict.get(word)['Quran'] = dict_Quran_words_count.get(word, 0)
