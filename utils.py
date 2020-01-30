__all__ = ['sort_dict', 'compute_doc_lengths', 'descriptive_stats']

import numpy as np
import pandas as pds

def sort_dict(some_dict:dict): return dict(sorted(some_dict.items(), key=lambda kv:kv[1], reverse=True))

def compute_doc_lengths(df:pd.DataFrame, col:str):
    return [len(doc.split()) for doc in df.loc[:, col].values if isinstance(doc, str)]

def descriptive_stats(df:pd.DataFrame, cols:list, domains:list=None):
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