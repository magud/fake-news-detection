# based on:
# https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

from __future__ import absolute_import, division, print_function

import sys
sys.path.append('/home/ubuntu/.local/lib/python3.6/site-packages')
sys.path.append('/home/ubuntu/fnd_implementation')
import argparse

import glob
import logging
import os
import random
import json

import math
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm_notebook, trange
from tensorboardX import SummaryWriter

from sklearn.metrics import accuracy_score

from transformers import (WEIGHTS_NAME, 
                            BertConfig, BertForSequenceClassification, BertTokenizer,
                            RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, # used for Roberta and Distilbert
                            AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
                            XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                            AdamW,
                            get_constant_schedule_with_warmup, 
                            get_linear_schedule_with_warmup,
                            get_cosine_schedule_with_warmup
                        )

from src.utils import (convert_examples_to_features,
                        output_modes, processors, 
                        score_submission, print_confusion_matrix, get_matches, get_f1, get_f1_overall,
                        freeze_embed, 
                        get_batch_size_seq_length
                        )


import ray
from ray import tune
from ray.tune import track
from ray.tune.logger import TBXLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Command-line arguments
parser = argparse.ArgumentParser(description='FN Challenge')
parser.add_argument('--model', type=str, default='bert', metavar='N',
                    help='Model to evaluate, choose bert, roberta, distilbert, albert or xlnet, (default: bert)')
parser.add_argument('--model_type', type=str, default='bert-base-cased', metavar='N',
                    help='Model to evaluata, choose bert-base-cased, roberta-base, distilroberta-base, albert-base-v1 or xlnet-base-cased, (default: bert-base-cased)')
parser.add_argument('--num_epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--dataset_name', type=str, default='fnc', metavar='N',
                    help='Dataset to evaluate, choose fnc or fnc_arc (default: fn)')
args = parser.parse_args()

args_add = {
    'data_dir': '/home/ubuntu/fnd_implementation/data/processed',
    'task_name': 'multi',
    'output_dir': 'outputs/',
        
    'output_mode': 'classification',

    #'gradient_accumulation_steps': 1,
    'weight_decay': 0,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0, 
    'num_logging': 50
}

# inspect availability of GPU power
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():    
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print('The following GPU is used: ', torch.cuda.get_device_name(0))

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model]

#configuration = config_class.from_pretrained(args.model_type, num_labels=4, finetuning_task=args_add['task_name'])
tokenizer = tokenizer_class.from_pretrained(args.model_type, do_lower_case=False)

task = args_add['task_name']

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

output_dir_model = os.path.join('/home/ubuntu/fnd_implementation/', args.model, 'model_pretrained')

# if model was already used in last experiment, use exact same model again
if os.path.exists(output_dir_model):
    model = model_class.from_pretrained(output_dir_model)
    logger.info("Loading initialized pretrained model from %s", output_dir_model)
else:
    os.makedirs(output_dir_model)
    model = model_class.from_pretrained(args.model_type, num_labels=4)
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model.save_pretrained(output_dir_model)
    logger.info("Saving initialized pretrained model to %s", output_dir_model)

print("  ******************************************* ")
print("           Freezing Embedding Layers          ")
print("  *******************************************\n ")
freeze_embed(args.model, model)

search_space = {
    "batch_size_seq_length": tune.grid_search([1,2,3,4]), 
    "lr": tune.grid_search([1e-05]),
    "lr_type": tune.grid_search(["constant"])
}

def load_and_cache_examples(task, tokenizer, max_seq_length=512, evaluate=False, train_eval=True):
    processor = processors[task]()
    
    logger.info("Creating features from dataset file at %s", args_add['data_dir'])
    label_list = processor.get_labels()
    if train_eval: 
        examples = processor.get_dev_examples(args_add['data_dir'], args.dataset_name) if evaluate else processor.get_train_examples(args_add['data_dir'], args.dataset_name)
    else: 
        examples = processor.get_test_examples(args_add['data_dir'], args.dataset_name)
             
    if __name__ == "__main__":
            features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, 
                cls_token_at_end=bool(args.model in ['xlnet']),            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model in ['xlnet'] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.model in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(args.model in ['xlnet']),                 # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model in ['xlnet'] else 0)
        
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
   
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

################################
# HAS TO BE FILLED IN BY HAND  #
################################
# albert lr_3: pipeline_e414339d_2020-02-04_01-14-41iyih8w6c with f1 = 0.7436762778377414
# albert lr_4: pipeline_7cf23c3e_2020-02-04_11-13-29qpg9g36r with f1 = 0.7439490311877925
# albert lr_1: pipeline_0c6bb2e9_2020-02-02_09-19-07jz_todo1 with f1 = 0.7348579749975507
# albert lr_2: pipeline_059ad1ff_2020-02-02_23-58-03gmwy7c8v with f1 = 0.7595712867035815

# bert lr_4: pipeline_711dee64_2020-02-05_16-31-16su_iwvgw with f1 = 0.813827843805033
# bert lr_3: pipeline_5aa271fe_2020-02-06_09-32-39qdqcvxui with f1 = 0.8125985483897753
# bert lr_2: pipeline_b352ca52_2020-02-07_04-23-01vtufcmov with f1 = 0.813421374819208
# bert lr_1: pipeline_42fc7d8e_2020-02-07_10-33-20nlwf2m67 with f1 = 0.8024127516687194

# xlnet lr_2: pipeline_0cacb3c4_2020-02-08_12-57-12m7o44jke with f1 = 0.8546972970862406

# fnc_arc
# roberta lr_1: pipeline_abfb6296_2020-02-14_16-10-26vjyufsvp with f1 = 0.8127813789844165
# roberta lr_4: pipeline_f476b17a_2020-02-20_12-00-00491p74b_ with f1 = 0.7976126255219581

# bert lr_1: pipeline_2e2bd1ee_2020-02-14_21-09-32h520viza with f1 = 0.767972497089149
# bert lr_3: pipeline_78b0bd82_2020-02-18_14-14-01g90xeutb with f1 =  0.7920858045249626

# xlnet lr_3: pipeline_73ac6694_2020-02-16_18-38-09yg6rkgu9 with f1 = 0.8176211589445734
# xlnet lr_4: pipeline_138386e6_2020-02-22_11-13-14ri6mtlpc with f1 = 0.8170590657049875

# albert lr_3: pipeline_c67870f1_2020-02-17_13-26-27guqnjr6c with f1 = 0.725070611475388
# albert lr_4: pipeline_42fc55b2_2020-02-19_16-37-21kqp3l0ll with f1 = 0.7448424323001577

# manually define winner!
winner = 'pipeline_138386e6_2020-02-22_11-13-14ri6mtlpc'

winner_dir = '/home/ubuntu/fnd_implementation/'+args.model+'/grid_search/results/'+ winner+'/'
print(winner_dir+'params.json')

# get best config
with open(winner_dir+'params.json') as f:
    winner_config = json.load(f)

batch_size, max_seq_length = get_batch_size_seq_length(winner_config['batch_size_seq_length'])
winner_model = model_class.from_pretrained(winner_dir+'outputs/checkpoint-'+str(args.num_epochs))
winner_model.to(device)

# load data
test_dataset = load_and_cache_examples(task, tokenizer, max_seq_length, train_eval=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

logger.info("  ******************************************* ")
logger.info("              Running Final Testing           ")
logger.info("  ******************************************* \n ")

logger.info("  ******************************************* ")
logger.info("  Model: %s", args.model)
logger.info("  Length dataset testing: %s", len(test_dataset))
logger.info("  Length dataloader testing: %s", len(test_dataloader))
logger.info("  Number of epochs: %d", args.num_epochs)
logger.info("  Maximal sequence length: %s", max_seq_length)
logger.info("  Batch size: %d", batch_size)
# it's weird that %s is necessary, since lr is not a string but a float!
logger.info("  Learning rate: %s", winner_config["lr"])
logger.info("  Learning rate type: %s", winner_config["lr_type"])
logger.info("  ******************************************* \n ")
      
# reinitialize preds for every checkpoint
preds = None 
out_label_ids = None

for batch in tqdm_notebook(test_dataloader, desc="Testing"):
    winner_model.eval()
    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                        # Distilbert and Roberta don't use segment_ids
                    'token_type_ids': batch[2] if args.model in ['bert', 'xlnet', 'albert'] else None, 
                    'labels':         batch[3]}
        outputs = winner_model(**inputs)
        logits = outputs[1]
    
    if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs['labels'].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                
preds = np.argmax(preds, axis=1)

acc = accuracy_score(out_label_ids, preds)
fnc_score, conf_matrix = score_submission(preds=preds, labels=out_label_ids)
fnc_score_best, _ = score_submission(preds=out_label_ids, labels=out_label_ids)
fnc_score_rel = (fnc_score*100)/fnc_score_best
f1, f1_scores = get_f1_overall(labels=LABELS, conf_matrix=conf_matrix)

#############################################
# info for console
print_confusion_matrix(conf_matrix)
print("Score: " +str(fnc_score) + " out of " + str(fnc_score_best) + "\t("+str(fnc_score*100/fnc_score_best) + "%)")
print("Accuracy: "+str(acc))
print("F1 overall: "+str(f1))
print("F1 per class: "+str(f1_scores))