# Fake News Detection
Master's thesis about Fake News Challenge Stage 1 with BERT, RoBERTa, DistilBERT, ALBERT and XLNet


src/ contains all relevant util functions

data_prep.py 
	concatenates headline and article body, 
	removes stop words and 
	splits the training data into training and evaluation set

the three model_exploration scripts are mainly the same except for the used freezing technique

model_grid_search requires a manual definition of the current learning rate in the search_space dictionary

the main difference between the model_exploration scripts and model_grid_search is that 
	the latter relies on the use of tune to speed up training and to perform grid search

in some cases, the model_grid_search didn't end for one run, in that case, the evaluation and testing step were performed separately in addition

bert/, roberta/, distilbert/, albert/ and xlnet/ contain the used preloaded weights for all experiments and the grid search

all output files can be found under results/
