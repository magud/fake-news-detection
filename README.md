| Hyperparameter  |        |          
| :-------------- | :----- | :-----  |
| Sequence length | 256    | 512   
| Batch size      | 16,32  | 4,8  
| Learning rate  <td colspan=2>1e-5, 2e-5, 3e-5, 4e-5   
| Learning rate schedule  <td colspan=2>constant, linear, cosine    


# Fake News Detection
Master's thesis about evaluating BERT, RoBERTa, DistilBERT, ALBERT and XLNet for detecting stances of Fake News.

## Goal and Background
In this master's thesis, the two datasets FNC-1 and FNC-ARC are used to finetune large pretrained NLP models  
to classify the stances of article bodies towards their respective headline. 

The goal is to systematically analyze the following questions: 
1. How well do the models perform in general?
2. How much hyperparameter tuning is necessary?
3. Which of the models performs best? 

The background of the thesis is the Fake News Challenge which was held in 2017. More details can be found [here](http://www.fakenewschallenge.org/). 

## Datasets
In total, two datasets are used to finetune the five models. The first dataset comes from the Fake News Challenge itself,   
while the second dataset is an extesion that was created by [Hanselowski et al](https://arxiv.org/pdf/1806.05180.pdf).  
Both datasets consist of article bodies, headlines and class labels. The class label expresses the stance of the article body  
towards the headline. The article body can either *Agree* (AGR) or *Disagree* (DSG) with the headline, it can *Discuss* (DSC) it  
or be completely *Unrelated* (UNR). 

| Dataset | Data Source | Data Type | Instances | AGR | DSG | DSC | UNR | 
| :------ | :---------- | :-------- | :-------- | :---| :-- | :-- | :-- |
| FNC-1 | [Fake News Challenge Stage 1](https://github.com/FakeNewsChallenge/fnc-1/tree/29d473af2d15278f0464d5e41e4cbe7eb58231f2)| News articles | 49,972 | 7.4% | 1.7% | 17.8% | 73.1% |
| FNC-1 ARC | [Review of the Challenge](https://github.com/UKPLab/coling2018_fake-news-challenge/tree/master/data/fnc-1/corpora/FNC_ARC) | + User posts | 64,205 | 7.7% | 3.5% | 15.3% | 73.5% |

## Data Pre-Processing
| Step | Details |
| :--- | :------ |
| Concatenation | Headline + Article body | 
| Stop word removal   | The, the, A, a, An, an |
| Train-dev split | 80:20 |

## Models
In total, five models are examined and their implementation of [HuggingFace](https://huggingface.co/transformers/) is used.  

| Model | Publication Date | Published By | Idea in a Nutshell
| :---- | :--------------- | :----------- | :-------------- |
| [BERT](https://arxiv.org/pdf/1810.04805.pdf)  | Oct 2018 | Google AI Language | Bidirectional Encoders from Transformer |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)   | Jul 2019 | Facebook AI &<br>University of Washington | Train BERT excessively |
| [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) | Aug 2019 | HuggingFace | Distill BERT |
| [ALBERT](https://arxiv.org/pdf/1909.11942.pdf) | Sep 2019 | Google Research &<br>Toyota Technological Institute at Chicago | Distill BERT |
| [XLNet](https://arxiv.org/pdf/1906.08237.pdf) | Jun 2019 | Carnegie Mellon University &<br>Google Brain | Permutation Language Model |

## Evaluating Unsupervised Representation Learning
The evaluation is conducted in two steps.  

In the first experimental setup, all models are trained for 2 epochs,  
with a learning rate of 3e-5, a sequence length of 512 tokens, a batch size of 8 and a linear learning rate schedule.  
With this fixed setting of hyperparameters three runs were conducted per model and dataset. The first run freezes all  
layers except for the last two (pooling & classification layer). The second run finetunes all layers. The third run  
freezes all embeddings layers. 

The second step consists of an extensive grid search over the hyperparameters learning rate, batch size, sequence length  
and learning rate schedule and covers the following grid: 

 

## Additional Remarks
* the three model_exploration scripts are mainly the same except for the used freezing technique

* model_grid_search requires a manual definition of the current learning rate in the search_space dictionary

* difference between the model_exploration scripts and model_grid_search:  
 	the latter relies on the use of tune to speed up training and to perform grid search

* in some cases, the model_grid_search didn't end for one run,   
in that case, the evaluation and testing step were performed separately in addition

* bert/, roberta/, distilbert/, albert/ and xlnet/ contain the used preloaded weights for all experiments and the grid search

* all output files can be found under results/
