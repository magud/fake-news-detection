#!/bin/bash

time_var=$(date) 
python3 grid_search.py 2>&1 | tee -a bert_grid_search_fnc_lr1.log 
echo $time_var 2>&1 | tee -a bert_grid_search_fnc_lr1.log 
date 2>&1 | tee -a bert_grid_search_fnc_lr1.log