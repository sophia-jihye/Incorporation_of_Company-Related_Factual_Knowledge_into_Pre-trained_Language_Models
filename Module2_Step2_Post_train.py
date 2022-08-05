from glob import glob
import pandas as pd
import os, time
from datetime import timedelta
from LazyLineByLineTextDataset import LazyLineByLineTextDataset
from transformers_helper import load_tokenizer_and_model
import post_training_mlm

root_dir = '/data/jihye_data/ComBERT'
company_filepath = os.path.join(root_dir, 'data', 'company_info_sec_cik_mapper_12057_20220802.csv')
post_filepath = os.path.join(root_dir, 'pt_data', 'post_item1_converted_subnames_to_fullname_sentences_with_fullname.txt') 

save_dir_format = os.path.join(root_dir, 'pt_{}_{}_company_masking_first={}', '{}')
    
def record_elasped_time(start, save_filepath):
    end = time.time()
    content = "Time elapsed: {}".format(timedelta(seconds=end-start))
    print(content)
    with open(save_filepath, "w") as f:
        f.write(content)    
    
def start_post_train(model_name_or_dir, post_filepath, save_dir, company_names, is_masking_company_name_first):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, mode='masking')
    dataset = LazyLineByLineTextDataset(tokenizer=tokenizer, file_path=post_filepath)
    post_training_mlm.train(tokenizer, model, dataset, save_dir, company_names, is_masking_company_name_first)

if __name__ == '__main__':
    
    for masking_mode, is_masking_company_name_first in [('with', True), ('wo', None)]:
        for model_name_or_dir in ['nlpaueb/sec-bert-base', 'yiyanghkust/finbert-pretrain', 'ProsusAI/finbert', 'bert-base-uncased']:
            start = time.time()
            data_mode = os.path.basename(post_filepath).replace('post_', '').replace('.txt', '')
            save_dir = save_dir_format.format(os.path.basename(model_name_or_dir), masking_mode, is_masking_company_name_first, data_mode)
            if not os.path.exists(save_dir): os.makedirs(save_dir)

            if masking_mode == 'with':
                company_names = [item.lower() for item in list(pd.read_csv(company_filepath).Name.unique())]
            elif masking_mode == 'wo':
                company_names = None
            start_post_train(model_name_or_dir, post_filepath, save_dir, company_names, is_masking_company_name_first)
            record_elasped_time(start, os.path.join(save_dir, 'elapsed-time.log'))