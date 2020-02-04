from __future__ import absolute_import, division, print_function

__all__ = [
           'get_file', 
           'get_data',
           'create_examples',
           'convert_examples_to_features',
           'create_tensor_dataset',
           'create_batches',
           'split_into_train_and_dev',
           'tokenize_qas', 
           'sort_dict', 
           'compute_doc_lengths', 
           'descriptive_stats_squad', 
           'descriptive_stats_subjqa', 
           'filter_sbj_levels',
            ]


import collections
import numpy as np
import pandas as pd

import json
import os
import re
import torch

#from keras.preprocessing.sequence import pad_sequences
from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

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
                        answer_start = qas['answers'][0]['answer_start'] if len(qas['answers']) == 1 else 0
                        answer_end = re.sub(r"\\", "", p['context']).split().index(answer.split()[-1]) if len(qas['answers']) == 1 else 0
                        qas_pairs.append({'question': qas['question'], 
                                          'answer': answer,
                                          'answer_span': (answer_start, answer_end),
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
            assert isinstance(domain, str), 'domain must be one of {books, electronics, grocery, movies, restaurants, tripadvisor}'
            file = get_file(subdir, source, domain, split)
            return pd.read_csv(file)
    else:
        raise Exception('You did not provide the correct subfolder name')
   
    
def create_examples(paragraphs:list, is_training:bool=True):
    """
    Args:
        paragraphs(list): list of examples 
        is_training (bool): whether we want to create examples for training or eval mode
    Return:
        list of examples (each example is an instance)
    """

    if not isinstance(paragraphs, list):
        raise TypeError("Input should be a list of examples.")
        
    def is_whitespace(char:str):
        if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
            return True
        return False
    
    examples = []
    for para in paragraphs:
        
        para_text = re.sub(r"\\", "", para["context"])
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        
        for c in para_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
    
        for qa in para["qas"]:
            qas_id = qa["id"]
            q_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = qa['answers'][0]['text'] if len(qa['answers']) == 1 else ''
            is_impossible = qa['is_impossible']
            
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
               
               
                    
        example = InputExample(
                               qas_id=qas_id,
                               q_text=q_text,
                               doc_tokens=doc_tokens,
                               orig_answer_text=orig_answer_text,
                               start_position=start_position,
                               end_position=end_position,
                               is_impossible=is_impossible,
                               )
    
        examples.append(example)
        
    return examples   
    

        
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
        ):
        
        self.qas_id = qas_id
        self.q_text = q_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        
        
def convert_examples_to_features(
                                examples,
                                tokenizer,
                                max_seq_length,
                                doc_stride,
                                max_query_length,
                                is_training,
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
        
        # the following step is necessary since WordPiece tokenized documents are longer than original documents
        # want to find the new start and end positions of answer span (different indexes compared to original)
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
            cls_index = 0  # Index of classification token [CLS]

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
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        
        if evaluate:
            dataset = TensorDataset(
                                    all_input_ids, 
                                    all_input_mask, 
                                    all_segment_ids, 
                                    all_example_index, 
                                    all_cls_index, 
                                    all_p_mask,
                                    )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            
            dataset = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_input_lengths,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
            )

        return dataset
                 

def create_batches(
                   dataset:torch.Tensor, 
                   batch_size:int,
                   split:str='train', 
                   ):
    """
    Args:
        dataset (torch.Tensor): TensorDataset
        batch_size (int): number of sequences in each mini-batch
        split (str): training or evaluation
        num_samples (int): number of samples to draw (equivalent to number of iterations)
    Return:
        PyTorch data loader (DataLoader object)
    """
    if split == 'train':
       # during training randomly sample examples
        sampler = RandomSampler(dataset, replacement=False)
    elif split == 'eval':
        # during testing sequentially sample elements from the test set (i.e., always sample in the same order)
        sampler = SequentialSampler(dataset)
    dl = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dl

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


### OLD helpers ###

def tokenize_qas(contexts:list, questions:list, max_bert_seq_len:int=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    input_lengths, input_ids, token_type_ids, attention_masks = [], [], [], []
    q_dropped = 0
    for context, question in zip(contexts, questions):
        encoded_seq = tokenizer.encode(context, question)
        seq_length = len(encoded_seq)
        if seq_length > max_bert_seq_len:
            q_dropped += 1
            continue
        else:
            input_lengths.append(seq_length)
            encoded_dict = tokenizer.encode_plus(context, question, max_length=max_bert_seq_len, pad_to_max_length=True)
            input_ids.append(encoded_dict['input_ids'])
            token_type_ids.append(encoded_dict['token_type_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
    if q_dropped == 0:
        raise Exception('The max seq length parameter must be updated to actual max seq length.')
    print("{} questions had to be dropped due to T > 512.".format(q_dropped))
    return torch.tensor(input_lengths), torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_masks)

if __name__== "__main__":       
    main() 