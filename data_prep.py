from src.utils import (DataSet, 
                generate_hold_out_split,
                labels_to_int)
import pandas as pd
import nltk
import os

def data_processing(name="train", dataset_name="fnc", output_dir="data/processed/"):
    """ 
    choose dataset_name out of ["fnc", "fnc_arc"]
    choose name out of ["train", "competition_test"]
    """
    print_info = dataset_name+" "+name
    print("\n*******************************************")
    print("        CURRENT DATASET: ", print_info)
    print("*******************************************\n")
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = "data/raw/" + dataset_name + "/"

    # data reading, produces already merged dataset
    #  has three attributes: 
    #   stances, articles and data (merged stances and articles)
    dataset = DataSet(name=name, path=path)
    
    # already merged data of article bodies and stances
    data = dataset.data_merged
    # take their body ids to generate hold out split
    bodies = dataset.bodies

    # create columns for multi and binary label
    data["label_multi"], data["label_bin"] = data["Stance"], data["Stance"]

    # assign integer values to class labels
    label_int_multi = {
        "agree": 0,
        "disagree": 1,
        "discuss": 2, 
        "unrelated": 3
    }
    
    label_int_bin = {
        "agree": 0,
        "disagree": 0,
        "discuss": 0, 
        "unrelated": 1
    }

    labels_to_int(data.label_multi, label_int_multi)
    labels_to_int(data.label_bin, label_int_bin)

    # make sure, values are saved as actual integers
    data.label_bin = data.label_bin.astype("int64")
    data.label_multi = data.label_multi.astype("int64")

    # concatenate headlines with their respective article bodies
    data["text"] = data["Headline"] + ". " + data["articleBody"]

    # stop word removal
    # leave don't/ don\'t since tokenization (of BERT, XLNet etc) seems to be more useful this way 
    text = data.text

    for i in range(len(text)):
        text_current = text[i]
        # tokenize 
        text_tokenized = nltk.word_tokenize(text_current)
        # remove stop words 
        text_subset = [w for w in text_tokenized if w not in ["The", "the", "A", "a", "An", "an"]]
        # text subset contains each token as separate entry: put them back together
        text[i] = " ".join(text_subset)
        # nltk tokenizer uses different quotation marks to indicate start of a quote
        text[i] = text[i].replace("``", "''")

    # data split
    # split is done according to body IDs of article bodies
    if name == "train":
        data_train, data_dev = generate_hold_out_split(bodies, data, dataset_name=dataset_name)

        # create dataframes
        dataframe_train = pd.DataFrame({
            'id':range(len(data_train)),
            'label_multi': data_train.label_multi,
            'label_bin': data_train.label_bin,
            'alpha': ['a']*data_train.shape[0],
            'text': data_train.text
        })

        dataframe_dev = pd.DataFrame({
            'id':range(len(data_dev)),
            'label_multi': data_dev.label_multi,
            'label_bin': data_dev.label_bin,
            'alpha': ['a']*data_dev.shape[0],
            'text': data_dev.text
        })

        dataframe_train.to_csv(output_dir + dataset_name + "_train" + '.csv')
        dataframe_dev.to_csv(output_dir + dataset_name +  "_dev" +'.csv')

    else:
        dataframe_test = pd.DataFrame({
            'id':range(len(data)),
            'label_multi': data.label_multi,
            'label_bin': data.label_bin,
            'alpha': ['a']*data.shape[0],
            'text': data.text
        })

        dataframe_test.to_csv(output_dir + dataset_name + "_test" + '.csv')
        
# process training and test data for fnc and fnc + arc data basis 
data_processing("train", dataset_name="fnc")
data_processing("competition_test", dataset_name="fnc")
data_processing("train", dataset_name="fnc_arc")
data_processing("competition_test", dataset_name="fnc_arc")