#!/bin/bash

#time python3 experiments.py --model="bert" --model_type="bert-base-cased" --freeze="freeze" --dataset_name="fnc_arc" --seed=3

#time python3 experiments.py --model="bert" --model_type="bert-base-cased" --freeze="freeze" --dataset_name="fnc_arc" --seed=5

#time python3 experiments.py --model="bert" --model_type="bert-base-cased" --freeze="freeze" --dataset_name="fnc_arc" --seed=7

time_var=$(date) 
#ps -ef &>> test.log
python3 experiments.py --model="distilbert" --model_type="distilroberta-base" --freeze="freeze" --dataset_name="fnc" --seed=3 2>&1 | tee -a distilbert_experiment_freeze_seed_3_fnc.log 
echo $time_var 2>&1 | tee -a distilbert_experiment_freeze_seed_3_fnc.log
date 2>&1 | tee -a distilbert_experiment_freeze_seed_3_fnc.log

time_var=$(date) 
#ps -ef &>> test.log
python3 experiments.py --model="distilbert" --model_type="distilroberta-base" --freeze="freeze" --dataset_name="fnc" --seed=5 2>&1 | tee -a distilbert_experiment_freeze_seed_5_fnc.log 2>&1
echo $time_var 2>&1 | tee -a distilbert_experiment_freeze_seed_5_fnc.log
date 2>&1 | tee -a distilbert_experiment_freeze_seed_5_fnc.log