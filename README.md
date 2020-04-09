# Fake News Detection
Master's thesis about Fake News Challenge Stage 1 with BERT, RoBERTa, DistilBERT, ALBERT and XLNet  

In this master's thesis, the two datasets FNC-1 and FNC-ARC were used to fine-tune large pretrained NLP models to classify the stances of article bodies towards their respective headline. The goal is to systematically analyze how well the models perform in general, how much hyperparameter tuning is necessary and which of the models performs best.  
The background of the thesis is the Fake News Challenge which was held in 2017. More details can be found [here](http://www.fakenewschallenge.org/).

| Dataset | Data Source |
| ------------------- | :------ |
| FNC-1 | |
| FNC-1 ARC | |


| Data Pre-Processing | Details |
| ------------------- | :------ |
| Concatenation | Headline + Article body | 
| Stop word removal   | The, the, A, a, An, an |
| Train-dev split | 80:20 |

* the three model_exploration scripts are mainly the same except for the used freezing technique

* model_grid_search requires a manual definition of the current learning rate in the search_space dictionary

* difference between the model_exploration scripts and model_grid_search:  
 	the latter relies on the use of tune to speed up training and to perform grid search

* in some cases, the model_grid_search didn't end for one run,   
in that case, the evaluation and testing step were performed separately in addition

* bert/, roberta/, distilbert/, albert/ and xlnet/ contain the used preloaded weights for all experiments and the grid search

* all output files can be found under results/
