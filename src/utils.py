from __future__ import absolute_import, division, print_function

import csv
from csv import DictReader
import os
import sys
from io import open
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
from tensorboardX import SummaryWriter
import mlflow
import random
from collections import defaultdict

logger = logging.getLogger(__name__)
csv.field_size_limit(2147483647)

class DataSet():
    def __init__(self, name="train", path="data/raw/fnc/"):

        print("*******************************************")
        print("             Reading dataset                 ")
        print("*******************************************\n")
            
        stances = name+"_stances.csv"
        bodies = name+"_bodies.csv"

        self.stances = self.read(stances, path)
        self.bodies = self.read(bodies, path)
       
        print("Total stances: " + str(len(self.stances))+"\n")
        print("Basic information about stances:")
        print(self.stances.info())
        print("\n\nTotal bodies: " + str(len(self.bodies))+"\n")
        print("Basic information about article bodies:")
        print(self.bodies.info())

        print("\n*******************************************")
        print("              Merging dataset                ")
        print("*******************************************\n")

        self.data_merged = pd.merge(self.bodies, self.stances, on="Body ID")
        print("Total examples: " + str(len(self.data_merged))+"\n")
        print("Basic information about merged dataset:")
        print(self.data_merged.info())

    def read(self,filename, path):
        return pd.read_csv(path+filename)

def generate_hold_out_split(data_bodies, data_merged, dataset_name="fnc", training = 0.8, base_dir="data/splits"): 
    """
    fnc introduces train/holdout split with ids of article bodies
    in order to have as reproducible and comparable examples as possible the same split will be used 
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    r = random.Random()
    r.seed(1489215)

    # fnc spefically uses file of article bodies to produce split
    article_ids = list(data_bodies["Body ID"]) 
    # important step: otherwise data is ordered
    r.shuffle(article_ids)  

    training_ids = article_ids[:int(training * len(article_ids))]
    hold_out_ids = article_ids[int(training * len(article_ids)):]

    data_train = data_merged[data_merged["Body ID"].isin(training_ids)]
    data_hold_out = data_merged[data_merged["Body ID"].isin(hold_out_ids)]

    # write the split body ids out to files for future use
    with open(base_dir+ "/"+ "training_ids_"+dataset_name+".txt", "w+") as f:
        f.write("\n".join([str(id) for id in training_ids]))

    with open(base_dir+ "/"+ "hold_out_ids_"+dataset_name+".txt", "w+") as f:
        f.write("\n".join([str(id) for id in hold_out_ids]))

    return data_train, data_hold_out

def labels_to_int(dataframe_column, dictionary):
    """
    takes str inputs of labels and assigns corresponding value to them
    dataframe_column: takes column with labels as str
    dictionary: contains new values for str labels 
    """
    for key, value in list(dictionary.items()):
        dataframe_column[dataframe_column == key] = value

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def read_csv(cls, file):
        """Reads a tab separated value file."""
        data = pd.read_csv(file)
        return data

class MultiProcessor(DataProcessor):
    """Processor for the binary data sets"""

    def get_train_examples(self, data_dir, dataset_name):
        """See base class."""               
        return self._create_examples(
            self.read_csv(os.path.join(data_dir, str(dataset_name+"_train.csv"))), "train")

    def get_dev_examples(self, data_dir, dataset_name):
        """See base class."""
        return self._create_examples(
            self.read_csv(os.path.join(data_dir, str(dataset_name+"_dev.csv"))), "dev")
    
    def get_test_examples(self, data_dir, dataset_name):
        """See base class."""
        return self._create_examples(
            self.read_csv(os.path.join(data_dir, str(dataset_name+"_test.csv"))), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(data)):
            guid = "%s-%s" % (set_type, i)
            text_a = data.text[i]
            label = data.label_multi[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_example_to_feature(example_row, pad_token=0,
sequence_a_segment_id=0, sequence_b_segment_id=1,
cls_token_segment_id=1, pad_token_segment_id=0,
mask_padding_with_zero=True, sep_token_extra=False):
    example, label_map, max_seq_length, tokenizer, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id, sep_token_extra = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[str(example.label)] # sth was wrong here: added str()
   
    return InputFeatures(input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id)
    

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, 
                                 cls_token_at_end=False, sep_token_extra=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 process_count=cpu_count() - 2):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    examples = [(example, label_map, max_seq_length, tokenizer, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id, sep_token_extra) for example in examples]

    with Pool(process_count) as p:
        features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=500), total=len(examples)))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    "multi": MultiProcessor
}

output_modes = {
    "multi": "classification"
}

def score_submission(preds, labels):
    """  
    careful this function assumes that matrix is in form of 
       column = true labels, row = predicted labels
    """
    LABELS_NUM = [0,1,2,3]
    RELATED_NUM = LABELS_NUM[0:3]

    score = 0.0
    conf_matrix = [[0, 0, 0, 0], # pred: 0
                    [0, 0, 0, 0], # pred: 1
                    [0, 0, 0, 0], # pred: 2
                    [0, 0, 0, 0]] # pred: 3

    for i, (p, t) in enumerate(zip(preds, labels)):
        p_stance, t_stance = p, t
        if p_stance == t_stance: # if multi class prediciton in general correct
            score += 0.25
            if p_stance != 0: # if multi class prediction of REL correct
                score += 0.50
        if p_stance in RELATED_NUM and t_stance in RELATED_NUM: # if binary prediction UNREL vs REL correct
            score += 0.25

        conf_matrix[LABELS_NUM.index(p_stance)][LABELS_NUM.index(t_stance)] += 1

    return score, conf_matrix

def print_confusion_matrix(conf_matrix):
    """  
    careful this function assumes that matrix is in form of 
        column = true labels, row = predicted labels
        ! implementation in sklearn.metrics is just the other  way around!
        --> if sklearn.metrics.conf_matrix is fed: column = prediced, row = true label
    """
    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(conf_matrix):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))

def get_matches(conf_matrix, class_value):
    """  
    careful this function assumes that matrix is in form of 
       column = true labels, row = predicted labels
        ! implementation in sklearn.metrics is just the other  way around!
        --> if sklearn.metrics.conf_matrix is fed: fp <--> fn  
    """
    import numpy as np
    conf_matrix = np.asarray(conf_matrix)
    idx = [0,1,2,3]
    idx.remove(class_value)

    tp = conf_matrix[class_value][class_value]
    fp = conf_matrix[class_value][idx].sum()
 
    fn = 0
    for i in range(len(conf_matrix[idx])): 
        fn += conf_matrix[idx][i][class_value]
    tn = conf_matrix.sum() - tp - fn - fp
    matches = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}
    return matches


def get_f1(matches):
    """  
    feed any conf_matrix (sklearn.metrics or self implemented)  
    f1 score not affected by this, see formula
    """
    if (matches['tp'] + matches['fp'] + matches['fn']) != 0:
        f1 = (2*matches['tp']) / (2*matches['tp'] + matches['fp'] + matches['fn'])
    else: f1 = 0

    return f1


def get_f1_overall(labels, conf_matrix):
    """
    input:  list of labels available for multi classification problem
            [0, 1, 2, 3] or ['agree', 'disagree', 'discuss', 'unrelated']
            conf_matrix (sklearn.metrics or self implemented)
    output: float of average f1 score over all f1 scores per class 
    """
    import statistics

    f1_scores = []
    for i in range(len(labels)):
        matches = get_matches(conf_matrix, class_value = i)
        f1_score = get_f1(matches)
        f1_scores.append(f1_score)

    return statistics.mean(f1_scores), f1_scores

def log_scalar(name, value, step, tb_writer):
    """Log a scalar value to both MLflow and TensorBoard"""
    tb_writer.add_scalar(name, value, step)

def log_weights(model, step, tb_writer):
    """Log weights of all layers of a model to TensorBoard"""
    for name, weight in model.named_parameters():     
        tb_writer.add_histogram(name, weight, step)

def freeze(model_type, model):
    if model_type == "bert":
        for param in model.bert.parameters():
            param.requires_grad = False
        model.bert.pooler.dense.weight.requires_grad = True
        model.bert.pooler.dense.bias.requires_grad = True

    if model_type == "roberta" or model_type == "distilbert":
        for param in model.roberta.parameters():
            param.requires_grad = False
        model.roberta.pooler.dense.weight.requires_grad = True
        model.roberta.pooler.dense.bias.requires_grad = True

    if model_type == "albert":
        for param in model.albert.parameters():
            param.requires_grad = False
        model.albert.pooler.dense.weight.requires_grad = True
        model.albert.pooler.dense.bias.requires_grad = True
  
    if model_type == "xlnet":
        for param in model.transformer.parameters():
            param.requires_grad = False

        
def freeze_embed(model_type, model):
    if model_type == "bert":
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
  
    if model_type == "roberta" or model_type == "distilbert":
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
   
    if model_type == "albert":
        for param in model.albert.embeddings.parameters():
            param.requires_grad = False
    
    if model_type == "xlnet":
        for param in model.transformer.word_embedding.parameters():
            param.requires_grad = False
        model.transformer.mask_emb.requires_grad = False

def get_batch_size_seq_length(value):
    if value == 1: 
        batch_size = 4
        max_seq_length = 512

    if value == 2: 
        batch_size = 8
        max_seq_length = 512

    if value == 3: 
        batch_size = 16
        max_seq_length = 256

    if value == 4: 
        batch_size = 32
        max_seq_length = 256

    return batch_size, max_seq_length
