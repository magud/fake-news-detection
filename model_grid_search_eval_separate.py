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
                        get_batch_size_seq_length,
                        log_scalar
                        )


import ray
from ray import tune
#from ray.tune import track
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

configuration = config_class.from_pretrained(args.model_type, num_labels=4, finetuning_task=args_add['task_name'])
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

""" search_space = {
    "batch_size_seq_length": tune.grid_search([1,2,3,4]), 
    "lr": tune.grid_search([1e-05]),
    "lr_type": tune.grid_search(["constant"])
} """

main_dir = '/home/ubuntu/fnd_implementation/'+args.model+'/grid_search/results/'
sub_dirs = glob.glob(main_dir+"*/")
print("\n*******************************************")
print("Evaluating the following pipelines ")
print(sub_dirs)
print("\n*******************************************")

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


def train_eval(model, tokenizer, pipeline, config, tb_writer=None):

    batch_size, max_seq_length = get_batch_size_seq_length(config['batch_size_seq_length'])
        
    ########################################################################################################
    # EVALUATION

    # init eval structure
    #eval_output_dir = args_add['output_dir']
    #if not os.path.exists(eval_output_dir):
    #    os.makedirs(eval_output_dir)

    eval_dataset = load_and_cache_examples(task, tokenizer, max_seq_length, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)
    
    # start evaluation
    print("  ******************************************* ")
    print("              Running Evaluation                ")
    print("  *******************************************\n ")
    print("  Length dataset evaluation: "+str(len(eval_dataset)))
    print("  Length dataloader evaluation: "+str(len(eval_dataloader)))
    print("  Number of epochs: "+str(args.num_epochs))
    print("  Batch size: "+str(batch_size))
    print("  ******************************************* \n ")
   
    eval_loss = 0.0
    nb_eval_steps = 0
    out_label_ids = None

    checkpoints = [os.path.dirname(pipeline+'outputs/checkpoint-'+str(args.num_epochs)+'/'+ WEIGHTS_NAME)]
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = model_class.from_pretrained(checkpoint)
        model.to(device)
        
        # reinitialize preds for every checkpoint
        preds = None 
        
        for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                             # Distilbert and Roberta don't use segment_ids
                            'token_type_ids': batch[2] if args.model in ['bert', 'xlnet', 'albert'] else None, 
                            'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.item() # removed mean() since functionality is unclear --> for multi gpu use
                print("\r%f" % eval_loss, end='')

            nb_eval_steps += 1

            # calculate average loss
            eval_loss = eval_loss / nb_eval_steps
            log_scalar('loss_eval', eval_loss, nb_eval_steps, tb_writer=tb_writer)
            
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
        f1, f1_scores = get_f1_overall(labels=LABELS, conf_matrix=conf_matrix)

        #############################################
        # info for console
        print("\n*******************************************")
        print("EVALUATING PIPELINE ")
        print("Pipeline directory: " +str(pipeline))
        print("With following parameter combination: "+str(params))
        print("EVALUATION OF CHECKPOINT "+ checkpoint)
        print_confusion_matrix(conf_matrix)
        print("Score: " +str(fnc_score) + " out of " + str(fnc_score_best) + "\t("+str(fnc_score*100/fnc_score_best) + "%)")
        print("Accuracy: "+str(acc))
        print("F1 overall: "+str(f1))
        print("F1 per class: "+str(f1_scores))
        print("*******************************************\n")

comment = f' model={args.model} epochs={args.num_epochs}'
tb_writer = SummaryWriter(comment=comment)

for pipeline in sub_dirs: 
    with open(pipeline+'params.json') as f:
        params = json.load(f)
    print("\n*******************************************")
    print("EVALUATING PIPELINE ")
    print(params)
    train_eval(model, tokenizer, pipeline, config=params, tb_writer=tb_writer)



