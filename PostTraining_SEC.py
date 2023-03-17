from glob import glob
import pandas as pd
import os, time, argparse
from datetime import timedelta
from LazyLineByLineTextDataset import LazyLineByLineTextDataset
from transformers_helper import load_tokenizer_and_model
import post_training_mlm

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='WWM', help='CM: Company name Masking; SM: Subword Masking; WWM: Whole Word Masking')
parser.add_argument('--model', type=str, default='Yang', help='Yang: yiyanghkust/finbert-pretrain; Araci: ProsusAI/finbert; BERT: bert-base-uncased; SECBERT: nlpaueb/sec-bert-base')
args = parser.parse_args()
method_name = args.method
alias_model_name = args.model

model_name_dict = {'Yang': 'yiyanghkust/finbert-pretrain', 'Araci': 'ProsusAI/finbert', 'BERT': 'bert-base-uncased', 'SECBERT': 'nlpaueb/sec-bert-base'}

root_dir = '/home/jihyeparkk/DATA/ComBERT'
post_filepath = os.path.join(root_dir, 'data_postTraining', 'post_item1_converted_subnames_to_fullname_sentences_with_fullname.txt') 

save_dir_format = os.path.join(root_dir, 'models_post-trained', '{}_{}') # model_name, method_name
    
def record_elasped_time(start, save_filepath):
    end = time.time()
    content = "Time elapsed: {}".format(timedelta(seconds=end-start))
    print(content)
    with open(save_filepath, "w") as f:
        f.write(content)    
    
def start_post_train(model_name_or_dir, post_filepath, save_dir, method_name):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, mode='masking')
    dataset = LazyLineByLineTextDataset(tokenizer=tokenizer, file_path=post_filepath)
    post_training_mlm.train(tokenizer, model, dataset, save_dir, method_name)

if __name__ == '__main__':

    start = time.time()
    model_name_or_dir = model_name_dict[alias_model_name]
    save_dir = save_dir_format.format(alias_model_name, method_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    start_post_train(model_name_or_dir, post_filepath, save_dir, method_name)
    record_elasped_time(start, os.path.join(save_dir, 'elapsed-time.log'))