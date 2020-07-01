# based on:
# https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

from __future__ import absolute_import, division, print_function

import sys
sys.path.append('/home/ubuntu/.local/lib/python3.6/site-packages')
sys.path.append('/home/ubuntu/maike')
import argparse
import glob
import logging
import os
import random
import math
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm_notebook, trange
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from transformers import (WEIGHTS_NAME, 
                            BertConfig, BertForSequenceClassification, BertTokenizer,
                            RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, 
                            AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
                            XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                            AdamW,
                            get_constant_schedule_with_warmup, 
                            get_linear_schedule_with_warmup,
                            get_cosine_schedule_with_warmup
                        )
from src.utils import (convert_examples_to_features,
                        processors, 
                        score_submission, print_confusion_matrix, get_matches, get_f1, get_f1_overall,
                        get_batch_size_seq_length,
                        freeze, freeze_embed
                        )
import ray
from ray import tune
from ray.tune import track
from ray.tune.logger import TBXLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# command-line arguments
parser = argparse.ArgumentParser(description='FN Challenge')
parser.add_argument('--model', type=str, default='bert', metavar='N',
                    help='Model to evaluate, choose bert, roberta, distilbert, albert or xlnet, (default: bert)')
parser.add_argument('--model_type', type=str, default='bert-base-cased', metavar='N',
                    help='Model to evaluata, choose bert-base-cased, roberta-base, distilroberta-base, albert-base-v1 or xlnet-base-cased, (default: bert-base-cased)')
parser.add_argument('--num_epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 3)')
parser.add_argument('--dataset_name', type=str, default='fnc', metavar='N',
                    help='Dataset to evaluate, choose fnc or fnc_arc (default: fnc)')
parser.add_argument('--seed', type=int, default=3, metavar='N',
                    help='Seed for reproducibility')
parser.add_argument('--num_logging_steps', type=int, default=50, metavar='N',
                    help='Times of logging. For 50 logging steps, values of interest will be stored 50 times.')
parser.add_argument('--warmup_ratio', type=float, default=0.06, metavar='N',
                    help='Set ratio of optimization steps for which learning rate is linearly increased until specified value.')
parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='N',
                    help='Epsilon valjue for Adam-based optimization algorithms')
parser.add_argument('--max_grad_norm', type=float, default=1, metavar='N',
                    help='Value for gradient norm.')
parser.add_argument('--task_name', type=str, default='multi', metavar='N',
                    help='Indicates classification type. Choose "multi", "binary" not yet implemented')
parser.add_argument('--output_mode', type=str, default='classification', metavar='N',
                    help='Indicates classification type. Choose "multi", "binary" not yet implemented')
parser.add_argument('--home_path', type=str, default='/home/ubuntu/maike/', metavar='N',
                    help='General path')
parser.add_argument('--data_path', type=str, default='data/processed', metavar='N',
                    help='Path of processed data')
parser.add_argument('--freeze', type=str, default='freeze_embed', metavar='N',
                    help='Freezing technique for finetuning. Choose between "freeze", "no_freeze" and "freeze_embed".')  

args = parser.parse_args()

# inspect availability of GPU power
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():    
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print('The following GPU is used: ', torch.cuda.get_device_name(0))

# print out seed information
print("The following seed is used: seed=" + str(args.seed))

# fix the seed and make cudnn deterministic
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_classes = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
}

config_class, model_class, tokenizer_class = model_classes[args.model]

tokenizer = tokenizer_class.from_pretrained(args.model_type, do_lower_case=False)

task = args.task_name

labels = ['agree', 'disagree', 'discuss', 'unrelated']

# setting up main directories to save loaded pretrained model and checkpoints
checkpoint_dir = args.model + '_grid_search_' + args.dataset_name
output_dir_model = os.path.join(args.home_path, args.model, 'model_pretrained')

# use same randomly initialized classification layers for all experiments
if os.path.exists(output_dir_model):
    model = model_class.from_pretrained(output_dir_model, num_labels=4)
    logger.info("Loading initialized pretrained model from %s", output_dir_model)
else:
    os.makedirs(output_dir_model)
    model = model_class.from_pretrained(args.model_type, num_labels=4)
    # necessary for distributed/parallel training 
    model = model.module if hasattr(model, 'module') else model 
    model.save_pretrained(output_dir_model)
    logger.info("Saving initialized pretrained model to %s", output_dir_model)

# start freezing technique
if args.freeze == "freeze":
    print("  ******************************************* ")
    print("           Freezing All Layers                ")
    print("  *******************************************\n ")
    freeze(args.model, model)
if args.freeze == "freeze_embed":
    print("  ******************************************* ")
    print("           Freezing Embedding Layers          ")
    print("  *******************************************\n ")
    freeze_embed(args.model, model)
elif args.freeze == "no_freeze": 
    print("  ******************************************* ")
    print("           Freezing No Layers          ")
    print("  *******************************************\n ")
# check if correct freezing technique is used
for name, param in model.named_parameters():
    print('name: ', name)
    print('param.requires_grad: ', param.requires_grad)
    print('=====')

# define fixed search space for experimental set up
# manually change lr to current value
search_space = {
    "batch_size_seq_length": tune.grid_search([1,2,3,4]), 
    "lr": tune.grid_search([4e-05]),
    "lr_type": tune.grid_search(["cosine"])
}

def load_and_cache_examples(task, tokenizer, max_seq_length=512, evaluate=False, train_eval=True):
    processor = processors[task]()
    
    logger.info("Creating features from dataset file at %s", args.home_path + args.data_path)
    label_list = processor.get_labels()
    if train_eval: 
        examples = processor.get_dev_examples(args.home_path + args.data_path, args.dataset_name) if evaluate else processor.get_train_examples(args.home_path + args.data_path, args.dataset_name)
    else: 
        examples = processor.get_test_examples(args.home_path + args.data_path, args.dataset_name)
             
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

def train_eval(model, tokenizer, config, tb_writer=None):    
    
    batch_size, max_seq_length = get_batch_size_seq_length(config['batch_size_seq_length'])
    ########################################################################################################
    # TRAINING

    # init training structure
    train_dataset = load_and_cache_examples(task, tokenizer, max_seq_length)    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    
    model.to(device)
    model.zero_grad()
    
    t_total = len(train_dataloader) * args.num_epochs
    logging_steps = int(t_total/args.num_logging_steps)
    warmup_steps = math.ceil(t_total * args.warmup_ratio)
    
    optimizer = AdamW(model.parameters(), lr=config['lr'], eps=args.adam_epsilon)
    
    lr_types = {
        "constant": get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps),
        "linear": get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps= t_total),
        "cosine": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps= t_total)
    }
    scheduler = lr_types[config['lr_type']]

    # start training      
    print("  ******************************************* ")
    print("              Running Training                  ")
    print("  *******************************************\n ")
    print("  Model: "+str(args.model))
    print("  Length dataset training: "+str(len(train_dataset)))
    print("  Length dataloader training: "+str(len(train_dataloader)))
    print("  Number of epochs: "+str(args.num_epochs))
    print("  Batch size: "+str(batch_size))
    print("  Maximal sequence length: "+str(max_seq_length))
    print("  Learning rate: "+str(config['lr']))
    print("  Learning rate type: "+str(config['lr_type']))
    print("  Total optimization steps: "+str(t_total))
    print("  Steps between logging: "+str(logging_steps))
    print("  Number of logging steps: "+str(args.num_logging_steps))
    print("  *******************************************\n")
   
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    num_epoch = 0
        
    train_iterator = trange(int(args.num_epochs), desc="Epoch")

    for _ in train_iterator:
        
        epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      # Distilbert and Roberta don't use segment_ids
                      'token_type_ids': batch[2] if args.model in ['bert', 'xlnet', 'albert'] else None, 
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  
            # as backup check to see if model is still running
            print("\r%f" % loss, end='')
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            
            optimizer.step()
            scheduler.step()  
            model.zero_grad()
            global_step += 1
            
            # tracking loss has to be AFTER optimization steps were taken 
            if logging_steps > 0 and global_step % logging_steps == 0:
                track.log(lr=scheduler.get_lr()[0])
                track.log(loss_train=(tr_loss - logging_loss)/logging_steps)
                logging_loss = tr_loss

        num_epoch += 1

        if num_epoch == args.num_epochs:
            output_dir = os.path.join(checkpoint_dir, 'checkpoint-{}'.format(num_epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # necessary for distributed/parallel training 
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
    
    ########################################################################################################
    # EVALUATION

    # init eval structure
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
   
    eval_loss, eval_tr_loss = 0.0, 0.0
    nb_eval_steps = 0
    out_label_ids = None

    checkpoints = [checkpoint_dir]
    checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(checkpoint_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
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

                # for multi gpu, use mean() functionality
                eval_loss += tmp_eval_loss.item() 

            nb_eval_steps += 1

           # calculate average loss within epoch 
            eval_tr_loss = eval_loss / nb_eval_steps
            track.log(loss_eval_batch=eval_tr_loss)
            #log_scalar('loss_eval_batch', eval_tr_loss, nb_eval_steps, tb_writer=tb_writer) 
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        
        track.log(loss_eval_epoch=(eval_loss / nb_eval_steps))
                           
        preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(out_label_ids, preds)
        fnc_score, conf_matrix = score_submission(preds=preds, labels=out_label_ids)
        fnc_score_best, _ = score_submission(preds=out_label_ids, labels=out_label_ids)
        fnc_score_rel = (fnc_score*100)/fnc_score_best
        f1, f1_scores = get_f1_overall(labels=labels, conf_matrix=conf_matrix)

        print("\n*******************************************")
        print("EVALUATION OF CHECKPOINT "+ checkpoint)
        print_confusion_matrix(conf_matrix)
        print("Score: " +str(fnc_score) + " out of " + str(fnc_score_best) + "\t("+str(fnc_score*100/fnc_score_best) + "%)")
        print("Accuracy: "+str(acc))
        print("F1 overall: "+str(f1))
        print("F1 per class: "+str(f1_scores))
        print("*******************************************\n")

        track.log(acc=acc)
        track.log(fnc_score=fnc_score)
        track.log(fnc_score_best=fnc_score_best)
        track.log(fnc_score_rel=fnc_score_rel)
        track.log(f1_overall=f1)

def pipeline(config):
    print("Defined search space: ", search_space)

    print("\n*******************************************")
    print("batch_size_seq_length = 1"+"\n  Batch size: 4"+"\n  Maximal sequence length: 512\n\n")
    print("batch_size_seq_length = 2"+"\n  Batch size: 8"+"\n  Maximal sequence length: 512\n\n")
    print("batch_size_seq_length = 3"+"\n  Batch size: 16"+"\n  Maximal sequence length: 256\n\n")
    print("batch_size_seq_length = 4"+"\n  Batch size: 32"+"\n  Maximal sequence length: 256")
    print("*******************************************\n")

    train_eval(model, tokenizer, config)

dir_results = os.path.join('~/maike/',args.model, 'grid_search')

analysis = tune.run(
    pipeline,
    name='results', 
    config=search_space, 
    resources_per_trial={"gpu": 1},
    local_dir=dir_results,
    verbose=2
    ) 

#################################################################################################
######                                        TESTING                                     #######
#################################################################################################

# get best config
winner_config = analysis.get_best_config("f1_overall", mode="max")
logdir = analysis.get_best_logdir("f1_overall", mode="max")
print(winner_config)
print(logdir)

batch_size, max_seq_length = get_batch_size_seq_length(winner_config['batch_size_seq_length'])

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

# load trained model 
logdir = analysis.get_best_logdir("f1_overall", mode="max")
winner_model = model_class.from_pretrained(logdir + '/' + checkpoint_dir + '/checkpoint-'+ str(args.num_epochs))
winner_model.to(device)
        
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
f1, f1_scores = get_f1_overall(labels=labels, conf_matrix=conf_matrix)

#############################################
# info for console
print_confusion_matrix(conf_matrix)
print("Score: " +str(fnc_score) + " out of " + str(fnc_score_best) + "\t("+str(fnc_score*100/fnc_score_best) + "%)")
print("Accuracy: "+str(acc))
print("F1 overall: "+str(f1))
print("F1 per class: "+str(f1_scores))