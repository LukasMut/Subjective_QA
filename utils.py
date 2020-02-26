from __future__ import absolute_import, division, print_function

__all__ = [
           'get_file', 
           'get_data',
           'convert_df_to_dict',
           'create_examples',
           'convert_examples_to_features',
           'create_tensor_dataset',
           'get_class_weights',
           'idx_to_class',
           'class_to_idx',
           'BatchGenerator',
           'split_into_train_and_dev',
           'sort_dict', 
           'compute_doc_lengths', 
           'descriptive_stats_squad', 
           'descriptive_stats_subjqa', 
           'filter_sbj_levels',
           'find_start_end_pos',
            ]


import collections
import numpy as np
import pandas as pd

import json
import os
import random
import re
import string
import torch

from collections import defaultdict, Counter
#from keras.preprocessing.sequence import pad_sequences
from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

# set random seeds to reproduce results
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
"""

def get_file(
             subdir:str,
             source:str,
             subfolder:str,
             split=None,
):
    subfolder = subfolder if re.search('\w/', subfolder) else subfolder + '/'
    PATH = subdir + source + subfolder
    if isinstance(split, str):
        for file in os.listdir(PATH):
            if re.sub('/', '', split) + '.csv' == file: return PATH + file
    else:
        return PATH + os.listdir(PATH).pop()

def get_data(
             subdir:str='./data',
             source:str='/SQuAD/',
             split:str=None,
             domain:str=None,
             compute_lengths:bool=False,
):
    if source == '/SQuAD/': 
        assert isinstance(split, str), 'split must be one of {train, dev}'
        file = get_file(subdir, source, split)
        qas_pairs, paragraphs = [], []
        with open(file, 'r') as f:
            f = json.load(f)
            del f['version']
            for para in f['data']:
                for p in para['paragraphs']:
                    paragraphs.append(p)
            
                if compute_lengths:
                    for qas in p['qas']:
                        answer = qas['answers'][0]['text'] if len(qas['answers']) == 1 else ''
                        #answer_start = qas['answers'][0]['answer_start'] if len(qas['answers']) == 1 else 0
                        #answer_end = re.sub(r"\\", "", p['context']).split().index(answer.split()[-1]) if len(qas['answers']) == 1 else 0
                        qas_pairs.append({'question': qas['question'], 
                                          'answer': answer,
                                          #'answer_span': (answer_start, answer_end),
                                          # TODO: figure out, why regex below is not working properly
                                          'context': re.sub(r"\\", "", p['context']), 
                                          'is_answerable': not qas['is_impossible']})
            
            if compute_lengths:
                return descpritive_stats_squad(qas_pairs)
            else:
                return paragraphs
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
            domains = ['books', 'electronics', 'grocery', 'movies', 'restaurants', 'tripadvisor', 'all']
            assert isinstance(domain, str), 'domain must be one of {}'.format(domains)
            hidden_domain = 'trustyou'
            file = get_file(subdir, source, domain, split)
            subjqa_df = pd.read_csv(file)
            if domain == 'all':
                hidden_domain_indexes = subjqa_df[subjqa_df.name == hidden_domain].index.values.tolist()
                subjqa_df = subjqa_df[subjqa_df.name != hidden_domain]
                return subjqa_df, hidden_domain_indexes
            else:
                return subjqa_df
    else:
        raise Exception('You did not provide the correct subfolder name')
        

# NOTE: this function is only relevant for SubjQA data
def convert_df_to_dict(
                       subjqa:pd.DataFrame,
                       hidden_domain_indexes=None,
                       split:str='train',
):
    splits = ['train', 'dev', 'test']
    if split not in splits:
        # correct type but improper value
        raise ValueError('Split must be one of {}'.format(splits))
    
    columns = [
               'q_review_id',
               'question', 
               'review',
               'human_ans_spans', 
               'human_ans_indices',
               'name',
               'question_subj_level',
               'does_the_answer_span_you_selected_expresses_a_subjective_opinion_or_an_objective_measurable_fact',
    ]
    examples = []
    
    def convert_str_to_int(str_idx:str):
        str_idx = str_idx.translate(str.maketrans('', '', string.punctuation))
        return tuple(int(idx) for idx in str_idx.split())

    for i in range(len(subjqa)):
        if isinstance(hidden_domain_indexes, list):
            if i in hidden_domain_indexes:
                continue
        example = defaultdict(dict)
        example['qa_id'] = subjqa.loc[i, columns[0]]
        example['question'] = subjqa.loc[i, columns[1]]
        example['review'] = subjqa.loc[i, columns[2]]
        example['answer']['answer_text'] = subjqa.loc[i, columns[3]]
        answer_indices = convert_str_to_int(subjqa.loc[i, columns[4]])

        # TODO: figure out, whether we should strip off "ANSWERNOTFOUND" from reviews in SubjQA;
        #       if not, then start and end positions should be second to the last index (i.e., sequence[-2]) instead of 0 (i.e., [CLS]),
        #       since "ANSWERNOTFOUND" is last token in each review text

        example['answer']['answer_start'] = 0 if example['answer']['answer_text'] == 'ANSWERNOTFOUND' else answer_indices[0]
        example['answer']['answer_end'] = 0 if example['answer']['answer_text'] == 'ANSWERNOTFOUND' else answer_indices[1]
        example['domain'] = subjqa.loc[i, columns[5]]
        example['is_impossible'] = True if example['answer']['answer_text'] == 'ANSWERNOTFOUND' else False
        example['question_subj'] = 1 if subjqa.loc[i, columns[6]] > 3 else 0
        example['ans_subj'] = 1 if subjqa.loc[i, columns[7]] > 3 else 0
        examples.append(dict(example))
        
    # save as .json file
    with open('./data/SubjQA/all/' + split + '.json', 'w') as json_file:
        json.dump(examples, json_file)
        
    return examples
    
def create_examples(
                    examples:list,
                    source:str,
                    is_training:bool=True,
):
    """
    Args:
        examples(list): list of examples 
        is_training (bool): whether we want to create examples for training or eval mode
    Return:
        list of examples (each example is an instance)
    """
    sources = ['SQuAD', 'SubjQA']
    if source not in sources:
        raise ValueError('Data source must be one of {}'.format(sources))
    
    if not isinstance(examples, list):
        raise TypeError("Input should be a list of examples.")
        
    def is_whitespace(char:str):
        if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
            return True
        return False
    
    def preproc_context(context:str):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return doc_tokens, char_to_word_offset

    example_instances = []
    
    for example in examples:

        # TODO: figure out, whether we should strip off "ANSWERNOTFOUND" from reviews in SubjQA;
        #       if not, then start and end positions should be second to the last index (i.e., sequence[-2]) instead of 0 (i.e., [CLS]),
        #       since "ANSWERNOTFOUND" is last token in each review text
        
        context = example["context"] if source == 'SQuAD' else example["review"].rstrip('ANSWERNOTFOUND')
        doc_tokens, char_to_word_offset = preproc_context(context)

        if source == 'SQuAD':

            for qa in example["qas"]:

                qas_id = qa["id"]
                q_text = qa["question"]
                dataset = 'SQuAD'
                start_position = None
                end_position = None
                orig_answer_text = qa['answers'][0]['text'] if len(qa['answers']) == 1 else ''
                is_impossible = qa['is_impossible']
                q_sbj = 0
                a_sbj = 0
                domain = 'wikipedia'

                # we don't need start and end positions in eval mode
                if is_training:
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError("For training, each question should have exactly 1 answer.")

                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]

                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.

                        actual_text = " ".join(doc_tokens[start_position : (end_position + 1)])
                        cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))

                        if actual_text.find(cleaned_answer_text) == -1:
                            # logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            # skip example, if answer cannot be recovered from document
                            continue

                    # elif question is NOT answerable, then answer is the empty string and start and end positions are 0 
                    else:
                        start_position = 0
                        end_position = 0
                        orig_answer_text = ""

        elif source == 'SubjQA':

            qas_id = example['qa_id']
            q_text = example['question']
            dataset = 'SubjQA'
            start_position = None
            end_position = None
            is_impossible = example['is_impossible']
            q_sbj = example['question_subj']
            a_sbj = example['ans_subj']
            domain = example['domain']

            assert len(example['answer']) == 3, "Each answer must consist of an answer text, a start and an end index of answer span"

            if not is_impossible:
                orig_answer_text = example['answer']['answer_text']
                answer_offset = example['answer']['answer_start']
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                try:
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                # sometimes orig. answer text has more white spaces between tokens than the same char. sequence in review text,
                # thus we will get an IndexError (i.e., answer_length is too long)
                except IndexError:
                    orig_answer_text = context[answer_offset: example['answer']['answer_end']]
                    answer_length = len(orig_answer_text)
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]

                actual_text = " ".join(doc_tokens[start_position : (end_position + 1)])
                cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))

                if actual_text.find(cleaned_answer_text) == -1:
                    # skip example, if answer cannot be recovered from document
                    continue

            # elif question is NOT answerable, then answer is the empty string and start and end positions are 0 
            else:
                start_position = 0
                end_position = 0
                orig_answer_text = ""

        example_instance = InputExample(
                                        qas_id=qas_id,
                                        q_text=q_text,
                                        doc_tokens=doc_tokens,
                                        orig_answer_text=orig_answer_text,
                                        start_position=start_position,
                                        end_position=end_position,
                                        is_impossible=is_impossible,
                                        q_sbj=q_sbj,
                                        a_sbj=a_sbj,
                                        domain=domain,
                                        dataset=dataset,
        )

        example_instances.append(example_instance)
        
    return example_instances 
    
    
class InputExample(object):
    """
        A single training / test example.
        For examples without an answer, the start and end position are 0.
    """

    def __init__(
        self,
        qas_id,
        q_text,
        doc_tokens,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=None,
        q_sbj=None,
        a_sbj=None,
        domain=None,
        dataset=None,
        ):
        
        self.qas_id = qas_id
        self.q_text = q_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.q_sbj = q_sbj
        self.a_sbj = a_sbj
        self.domain = domain
        self.dataset = dataset
        
        
def convert_examples_to_features(
                                examples,
                                tokenizer,
                                max_seq_length,
                                doc_stride,
                                max_query_length,
                                is_training, # is_training only relevant for SQuAD
                                domain_to_idx,
                                dataset_to_idx,
                                cls_token="[CLS]",
                                sep_token="[SEP]",
                                pad_token=0,
                                sequence_a_segment_id=0,
                                sequence_b_segment_id=1,
                                cls_token_segment_id=0,
                                pad_token_segment_id=0,
                                mask_padding_with_zero=True,
                                sequence_a_is_doc=False,
):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    
    features = []
    for (example_index, example) in enumerate(tqdm(examples)):

        query_tokens = tokenizer.tokenize(example.q_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0: max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        
        # the following step is necessary since WordPiece tokenized docs are longer than white_space tokenized docs
        # want to find the new start and end positions of answer span (different indexes compared to original doc)
        for (i, token) in enumerate(example.doc_tokens):
            # accumulate number of subtokens until current index
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                # append every sub-token (word-piece) to find the correct start and end positions
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        
        if is_training and example.is_impossible:
            # end and start positions of answer span should be zero, if example is NOT answerable
            tok_start_position = 0
            tok_end_position = 0
            
        if is_training and not example.is_impossible:
            # NOTE: new start position is value (i.e., total number of sub-tokens until orig. start_pos) at index start_pos
            tok_start_position = orig_to_tok_index[example.start_position]
            
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.orig_answer_text,
            )

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])  # pylint: disable=invalid-name
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            p_mask = []

            # [CLS] token at the beginning
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = 0  # Index of special classification token [CLS]

            # BERT: CLS Question SEP Context SEP
            if not sequence_a_is_doc:
                # Query
                tokens += query_tokens
                segment_ids += [sequence_a_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            if sequence_a_is_doc:
                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

                tokens += query_tokens
                segment_ids += [sequence_b_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            
            q_sbj = example.q_sbj
            a_sbj = example.a_sbj
            domain = domain_to_idx[example.domain]
            dataset = dataset_to_idx[example.dataset]
            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    if sequence_a_is_doc:
                        doc_offset = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index 
                end_position = cls_index


            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_length=len(tokens),
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    q_sbj=q_sbj,
                    a_sbj=a_sbj,
                    domain=domain,
                    dataset=dataset,
                )
            )
            unique_id += 1

    return features

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        unique_id,
        example_index,
        doc_span_index,
        tokens,
        token_to_orig_map,
        token_is_max_context,
        input_ids,
        input_length,
        input_mask,
        segment_ids,
        cls_index,
        p_mask,
        paragraph_len,
        start_position=None,
        end_position=None,
        is_impossible=None,
        q_sbj=None,
        a_sbj=None,
        domain=None,
        dataset=None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_length = input_length
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.q_sbj = q_sbj
        self.a_sbj = a_sbj
        self.domain = domain
        self.dataset = dataset

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def split_into_train_and_dev(
                             train_examples:list,
                             train_proportion:float=0.8,
):
    n_examples = len(train_examples)
    train_set = train_examples[:int(n_examples * train_proportion)]
    dev_set = train_examples[int(n_examples * train_proportion):]
    return train_set, dev_set

def create_tensor_dataset(
                          features:list,
                          evaluate:bool=False,
):

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_input_lengths = torch.tensor([f.input_length for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)        
        
        # QA labels
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

        # auxiliary task labels
        all_q_sbj = torch.tensor([f.q_sbj for f in features], dtype=torch.long)
        all_a_sbj = torch.tensor([f.a_sbj for f in features], dtype=torch.long)
        all_sbj = torch.stack((all_a_sbj, all_q_sbj), dim=1)
        all_domains = torch.tensor([f.domain for f in features], dtype=torch.long)
        all_datasets = torch.tensor([f.dataset for f in features], dtype=torch.long)

          
        dataset = TensorDataset(
                                all_input_ids,
                                all_input_mask,
                                all_segment_ids,
                                all_input_lengths,
                                all_start_positions,
                                all_end_positions,
                                all_cls_index,
                                all_p_mask,
                                all_sbj,
                                all_domains,
                                all_datasets,
            )
        return dataset

    
def get_class_weights(
                      subjqa_classes:list,
                      idx_to_class:dict,
                      squad_classes=None,
                      binary:bool=False,
                      qa_type:str='questions',

):
    n_total_subjqa = len(subjqa_classes)
    class_distrib_subjqa = {idx_to_class[l]: freq for l, freq in Counter(subjqa_classes).items()}
    
    if isinstance(squad_classes, list):
        n_total_squad = len(squad_classes)
        n_total = n_total_subjqa + n_total_squad

        if len(class_distrib_subjqa) > 2:
            class_distrib_subjqa['wikipedia'] = n_total_squad
            class_distrib = class_distrib_subjqa

        elif len(class_distrib_subjqa) == 2:
            class_distrib_subjqa['obj'] += n_total_squad
            class_distrib = class_distrib_subjqa
    else:
        n_total = n_total_subjqa
        class_distrib = class_distrib_subjqa
    
    if binary:
        # for binary cross-entropy we just need to compute weight for positive class
        class_weight = class_distrib['obj'] / class_distrib['sbj']
        print("Subjective {} will be weighted {} higher than objective {}".format(qa_type, class_weight, qa_type))
        print()
        return torch.tensor(class_weight, dtype=torch.float)
    else:
        class_weights = {c: 1 - (v / n_total) for c, v in class_distrib.items()}
        print("Domain weights: {}".format(class_weights))
        print()
        # sort weights in the correct order (as will be presented to the model)
        if isinstance(squad_classes, type(None)):
            class_weights = [class_weights[c] for _, c in idx_to_class.items() if c != 'wikipedia']
        else:
             class_weights = [class_weights[c] for _, c in idx_to_class.items()]
        return torch.tensor(class_weights, dtype=torch.float)

def idx_to_class(classes:list): return dict(enumerate(classes))
def class_to_idx(classes:list): return {c: i for i, c in enumerate(classes)}


class BatchGenerator(object):
    
    def __init__(
                 self,
                 dataset:torch.Tensor,
                 batch_size:int,
                 sort_batch:bool=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = len(dataset) // batch_size
        self.sort_batch = sort_batch
    
    def __len__(self):
        return self.n_batches
    
    def __iter__(self):
        return create_batches(self.dataset, self.batch_size, self.n_batches, self.sort_batch)

def create_batches(
                   dataset:torch.Tensor,
                   batch_size:int,
                   n_batches:int,
                   sort_batch:bool=False,
):
    n_examples = len(dataset)
    idx = 0
    for _ in range(n_examples // batch_size):
        batch = dataset[idx: idx + batch_size]
        idx += batch_size
        
        if sort_batch:
            seq_length_pos = 3
            sorted_indices, _ = zip(*sorted(enumerate(batch[seq_length_pos]), key=lambda seq_lengths: seq_lengths[1], reverse=True))
            sorted_indices = np.array(sorted_indices)
            batch = tuple(t[sorted_indices] for t in batch)
        
        yield batch

## Helper functions to compute descriptive statistics ##

def sort_dict(some_dict:dict): return dict(sorted(some_dict.items(), key=lambda kv:kv[1], reverse=True))

def compute_doc_lengths(
                        df:pd.DataFrame,
                        col:str,
):
    return [len(doc.split()) for doc in df.loc[:, col].values if isinstance(doc, str)]

def descriptive_stats_squad(
                            qas_pairs:list,
                            docs:list=['question', 'answer', 'context'],
):
    desc_stats_squad = {}
    
    def get_mean_doc_length_squad(
                                  qas_pairs:list,
                                  doc:str,
                                  ):
        return np.mean([len(qas_pair[doc].split()) for qas_pair in qas_pairs])
    
    for doc in docs:
        desc_stats_squad['avg' + '_' + doc + '_' + 'length'] = get_mean_doc_length_squad(qas_pairs, doc)
    return desc_stats_squad
        
def descriptive_stats_subjqa(
                             df:pd.DataFrame,
                             cols:list,
                             domains:list=None,
):
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

def filter_sbj_levels(
                      subj_levels_doc_frq:dict,
                      likert_scale:list,
):
    return {subj_level: freq for subj_level, freq in subj_levels_doc_frq.items() if subj_level in likert_scale}


## Helpers to find start and end positions of annotated answer span in paragraph (i.e., review) due to missing positions in SubjQA ##

def remove_sq_brackets(string:str): return re.sub(r'\]', '', re.sub(r'\[', '', string))

def remove_punct(
                 list_of_strings:list, 
                 lower_case:bool,
): 
    if lower_case:
        return list(map(lambda s: remove_sq_brackets(s.translate(str.maketrans('', '', string.punctuation)).lower()), list_of_strings))
    else:
        return list(map(lambda s: remove_sq_brackets(s.translate(str.maketrans('', '', string.punctuation))), list_of_strings))

def check_remaining_indexes(
                            ans_span:list,
                            review:list,
                            start_idx:int,
                            lower_case:bool,
):
    review_span = review[start_idx + 1: start_idx + 1 + len(ans_span)]
    review_span = remove_punct(review_span, lower_case=lower_case)
    ans_span = remove_punct(ans_span, lower_case=lower_case)
    is_correct_idx = False
    if np.array_equal(ans_span, review_span):
        is_correct_idx = True
    return is_correct_idx

def find_start_end_pos(
                       answer:str,
                       review:str,
                       lower_case:bool,
):
    answer = answer.lower().split()
    review = review.lower().split()
    start = 0
    end = len(review)
    found_start_pos = False
    while not found_start_pos:
        start_token = remove_sq_brackets(answer[0])
        # NOTE: start is always inclusive and end is exclusive
        start_idx = review.index(start_token, start, end)
        if check_remaining_indexes(
                                   answer[1:],
                                   review,
                                   start_idx,
                                   lower_case,
        ):
            found_start_pos = True
        else:
            # update start position to find correct answer span in review
            idx_increment = (start_idx - start) + 1
            start += idx_increment
    start_pos = start_idx
    end_pos = start_idx + len(answer[1:])
    assert len(answer) == (end_pos - start_pos  + 1), 'start and end positions do not match the correct answer span'
    return (start_pos, end_pos)


if __name__== "__main__":       
    main() 