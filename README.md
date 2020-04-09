# Fake News Detection
Master's thesis about evaluating BERT, RoBERTa, DistilBERT, ALBERT and XLNet for detecting stances of Fake News.

## Goal and Background
In this master's thesis, the two datasets FNC-1 and FNC-ARC were used to finetune large pretrained NLP models to classify the stances of article bodies towards their respective headline.  
The goals are to systematically analyze  
1. how well the models perform in general  
2. how much hyperparameter tuning is necessary and 
3. which of the models performs best.  
The background of the thesis is the Fake News Challenge which was held in 2017. More details can be found [here](http://www.fakenewschallenge.org/). 

## Datasets
In total, two datasets are used to finetune the five models. The first dataset comes from the Fake News Challenge itself, 
while the second dataset is an extesion that was created by [Hanselowski et al](https://arxiv.org/pdf/1806.05180.pdf).

| Dataset | Data Source | Data Type | Instances | AGR | DSG | DSC | UNR | 
| ------------------- | :------ | :------ | :------ | :------ | :------ | :------ | :------ |
| FNC-1 | [Fake News Challenge Stage 1](https://github.com/FakeNewsChallenge/fnc-1/tree/29d473af2d15278f0464d5e41e4cbe7eb58231f2)| News articles | 49,972 | 7.4% | 1.7% | 17.8% | 73.1% |
| FNC-1 ARC | [Review of the Challenge](https://github.com/UKPLab/coling2018_fake-news-challenge/tree/master/data/fnc-1/corpora/FNC_ARC) | + User posts | 7.7% | 3.5% | 15.3% | 73.5% |

## Data Pre-Processing
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
