__all__ = ['get_file', 'get_data', 'create_pairs', 'sort_dict', 'compute_doc_lengths', 
           'descriptive_stats_squad', 'descriptive_stats_subjqa', 'filter_sbj_levels']

import numpy as np
import pandas as pd

import json
import os
import re

def get_file(subdir:str, source:str, subfolder:str, split:str=None):
    subfolder = subfolder if re.search('\w/', subfolder) else subfolder + '/'
    PATH = subdir + source + subfolder
    if isinstance(split, str):
        for file in os.listdir(PATH):
            if re.sub('/', '', split) + '.csv' == file: return PATH + file
    else:
        return PATH + os.listdir(PATH).pop()

def get_data(subdir:str='./data', source:str='/SQuAD/', split:str=None, domain:str=None, compute_lengths:bool=False):
    if source == '/SQuAD/': 
        assert isinstance(split, str), 'split must be one of {train, dev}'
        file = get_file(subdir, source, split)
        qas_pairs = []
        with open(file, 'r') as f:
            f = json.load(f)
            del f['version']
            for line in f['data']:
                for paragraph in line['paragraphs']:
                    for qas in paragraph['qas']:
                        answer = qas['answers'][0]['text'] if len(qas['answers']) == 1 else 'no_answer'
                        qas_pairs.append({'question': qas['question'], 
                                          'answer': answer,
                                          # TODO: figure out, why regex below is not working properly
                                          'context': re.sub(r"\\", "", paragraph['context']), 
                                          'answerable': not qas['is_impossible']})
        if compute_lengths:
            return descriptive_stats_squad(qas_pairs)
        else:
            return qas_pairs
    elif source == '/SubjQA/':
        if compute_lengths:
            domains = ['books', 'electronics', 'grocery', 'movies', 'restaurants', 'tripadvisor', 'all']
            cols = ['question', 'review', 'human_ans_spans']
            desc_stats_subjqa = {}
            for domain in domains:
                file = get_file(subdir, source, domain, split)
                domain_data = pd.read_csv(file)
                desc_stats_subjqa[domain.capitalize()] = descriptive_stats_subjqa(domain_data, cols)
            return desc_stats_subjqa
        else:
            assert isinstance(domain, str), 'domain must be one of {books, electronics, grocery, movies, restaurants, tripadvisor}'
            file = get_file(subdir, source, domain, split)
            return pd.read_csv(file)
    else:
        raise Exception('You did not provide the correct subfolder name')
        
def create_pairs(qas_pairs:list, bert:bool=False):
    questions, answers, support, answerable = [], [], [], []
    for pair in qas_pairs:
        questions.append('[CLS] ' + pair['question'] + ' [SEP]' if bert else pair['question'])
        answers.append(pair['answer'])
        support.append('[CLS] ' + pair['context'] + ' [SEP]' if bert else pair['context'])
        answerable.append(pair['answerable'])
    return questions, answers, support, answerable

def sort_dict(some_dict:dict): return dict(sorted(some_dict.items(), key=lambda kv:kv[1], reverse=True))

def compute_doc_lengths(df:pd.DataFrame, col:str):
    return [len(doc.split()) for doc in df.loc[:, col].values if isinstance(doc, str)]

def descriptive_stats_squad(qas_pairs:list, docs:list=['question', 'answer', 'context']):
    desc_stats_squad = {}
    def get_mean_doc_length_squad(qas_pairs:list, doc:str): return np.mean([len(qas_pair[doc].split()) for qas_pair in qas_pairs])
    for doc in docs:
        desc_stats_squad['avg' + '_' + doc + '_' + 'length'] = get_mean_doc_length_squad(qas_pairs, doc)
    return desc_stats_squad
        
def descriptive_stats_subjqa(df:pd.DataFrame, cols:list, domains:list=None):
    if isinstance(domains, list):
        print('Computing descriptive stats per domain...')
        descript_stats = defaultdict(dict)
        for domain in domains:
            for col in cols:
                doc_lengths = compute_doc_lengths(df, col)
                descript_stats[domain]['avg' + '_' + col + '_' + 'length'] = np.mean(doc_lengths)
    else:
        descript_stats = {}
        for col in cols:
            doc_lengths = compute_doc_lengths(df, col)
            descript_stats['avg' + '_' + col + '_' + 'length'] = np.mean(doc_lengths)
    return descript_stats

def filter_sbj_levels(subj_levels_doc_frq:dict, likert_scale:list):
    return {subj_level: freq for subj_level, freq in subj_levels_doc_frq.items() if subj_level in likert_scale}

if __name__== "__main__":       
    main() 