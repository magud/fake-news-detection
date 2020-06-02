#!/bin/bash

# bert freeze fnc arc
time_var=$(date) 
python3 experiments.py --model="bert" --model_type="bert-base-cased" --freeze="freeze" --dataset_name="fnc_arc" --seed=3 2>&1 | tee -a bert_experiment_freeze_seed_3_fnc_arc.log 
echo $time_var 2>&1 | tee -a bert_experiment_freeze_seed_3_fnc_arc.log 
date 2>&1 | tee -a bert_experiment_freeze_seed_3_fnc_arc.log 

time_var=$(date) 
python3 experiments.py --model="bert" --model_type="bert-base-cased" --freeze="freeze" --dataset_name="fnc_arc" --seed=5 2>&1 | tee -a bert_experiment_freeze_seed_5_fnc_arc.log 
echo $time_var 2>&1 | tee -a bert_experiment_freeze_seed_5_fnc_arc.log 
date 2>&1 | tee -a bert_experiment_freeze_seed_5_fnc_arc.log 

time_var=$(date) 
python3 experiments.py --model="bert" --model_type="bert-base-cased" --freeze="freeze" --dataset_name="fnc_arc" --seed=7 2>&1 | tee -a bert_experiment_freeze_seed_7_fnc_arc.log 
echo $time_var 2>&1 | tee -a bert_experiment_freeze_seed_7_fnc_arc.log 
date 2>&1 | tee -a bert_experiment_freeze_seed_7_fnc_arc.log 


# roberta freeze fnc arc
time_var=$(date) 
python3 experiments.py --model="roberta" --model_type="roberta-base" --freeze="freeze" --dataset_name="fnc_arc" --seed=3 2>&1 | tee -a roberta_experiment_freeze_seed_3_fnc_arc.log 
echo $time_var 2>&1 | tee -a roberta_experiment_freeze_seed_3_fnc_arc.log 
date 2>&1 | tee -a roberta_experiment_freeze_seed_3_fnc_arc.log 

time_var=$(date) 
python3 experiments.py --model="roberta" --model_type="roberta-base" --freeze="freeze" --dataset_name="fnc_arc" --seed=5 2>&1 | tee -a roberta_experiment_freeze_seed_5_fnc_arc.log 
echo $time_var 2>&1 | tee -a roberta_experiment_freeze_seed_5_fnc_arc.log 
date 2>&1 | tee -a roberta_experiment_freeze_seed_5_fnc_arc.log 

time_var=$(date) 
python3 experiments.py --model="roberta" --model_type="roberta-base" --freeze="freeze" --dataset_name="fnc_arc" --seed=7 2>&1 | tee -a roberta_experiment_freeze_seed_7_fnc_arc.log 
echo $time_var 2>&1 | tee -a roberta_experiment_freeze_seed_7_fnc_arc.log 
date 2>&1 | tee -a roberta_experiment_freeze_seed_7_fnc_arc.log 