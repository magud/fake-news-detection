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
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)   | Jul 2019 | Facebook AI &<br>University of Washington | Pretrain BERT excessively |
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

<table>
  <tr>
    <td colspan="5"><b>Hyperparameter</b></td>
  </tr>
  <tr>
    <td>Sequence length</td>
 <td  colspan="2">256</td>
<td  colspan="2">512</td>
  </tr>
  <tr>
    <td>Batch size</td>
<td  colspan="2">16,32</td>
<td  colspan="2">4,8</td>
  </tr>
  <tr>
    <td>Learning rate</td>
<td  colspan="4">1e-5, 2e-5, 3e-5, 4e-5</td>
  </tr>
  <tr>
    <td>Learning rate schedule</td>
    <td  colspan="4">constant, linear, cosine </td>
  </tr>
</table>

## Key Results
1. RoBERTa performs best
2. Encoder-based approach of RoBERTa beats autoregressive approach of XLNet
3. Learning rate is most important hyperparameter

## Remarks Scripts
There are three main scripts:
* data_prep.py
* models_exploration.py
* models_grid_search.py

All three scripts are used via the command line.  

**To execute everything, first install the necessary packages via 
*pip3 install -r requirements.txt***

### Details on Data Pre-Processing script
Executing *python3 data_prep.py* takes the files
- train_bodies.csv
- train_stances.csv
- competition_test_bodies.csv
- competition_test_stances.csv  

for the FNC-1 and FNC-1 ARC dataset and fully processes them. 
The processed files can be found under data/processed.  
For both datasets three files 
are created for training (train), evaluation (dev) and testing (test) respectively.

The main pre-processing steps are
1. assign integer values 0,1,2,3 to the four classes AGR, DSG, DSC, UNR
2. merge headline and article body
3. remove stop words *The*, *the*, *A*, *a*, *An*, *an* by using the word tokenizer of NLTK
4. create split into  *training* and *development* by using the 80:20 split function of the FNC-1

## Details on Initial Experiments script
Executing *python3 model_exploration_freeze.py*, *python3 model_exploration_freeze_embed.py* and *python3 model_exploration_no_freeze.py* yields the evaluation of the three different freezing techniques. All models are trained for two epochs only and evaluation is done 
with respect to the evaluation dataset.  

The *--model* flag defines whether to use bert, roberta, distilbert, albert or xlnet  
The *--model_type* flag takes the specific pretrained model from *HuggingFace*, for exampe *bert-base-cased* for *bert*  
The *--num_epochs* flag is set to a default value of 2 epochs and should not be changed  
The *--dataset_name* flag can be used to switch between the FNC-1 and FNC-1 ARC dataset  

## Details on Grid Search script
Executing *python3 models_grid_search.py* is the script used that conducts the grid search over 48 hyperparameter combinations.  
It uses the *tune* package. In case the code couldn't finish due to for example storage capacity on the virtual machine, the evaluation and testing was redone in a separate script called *models_grid_search_eval_separate.py* and *models_grid_search_test_separate.py* respectively.  

**Important**: the current learning rate has to be set manually within the script in the search_space dictionary. The storage capacity of the virtual machine only allowed for saving 12 model combinations at the same time. Thus for each model and dataset, the script *models_grid_search.py* had to be run 4 times for each of the learning rates separately.  

Go to *Details on Initial Experiments script* for details on the flags that can be set. 

## Additional Remarks
The difference between the model_exploration scripts and model_grid_search is that 
the latter relies on the use of tune to speed up training and to perform grid search.    

In some cases, the model_grid_search didn't end for one run,   
in that case, the evaluation and testing step were performed separately in addition. Check **Details on Grid Search scrip** for more details.  

The folders bert/, roberta/, distilbert/, albert/ and xlnet/ contain the used preloaded weights for all experiments and the grid search.  

All output files can be found under the folder results/  
