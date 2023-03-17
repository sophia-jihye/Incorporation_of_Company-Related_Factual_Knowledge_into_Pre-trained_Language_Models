import os
import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForWholeWordMask

from custom_data_collator import DataCollatorForWholeWordMask as DataCollatorForCompanyNameMask

company_filepath = os.path.join('/home/jihyeparkk/DATA/ComBERT', 'data', 'company_info_sec_cik_mapper_12057_20220802.csv')

def train(tokenizer, model, dataset, save_dir, method_name='CM'):
    mlm_prob=0.15
    model.train()
    
    training_args = TrainingArguments(
        output_dir = save_dir,
        num_train_epochs = 5,   
        per_device_train_batch_size = 8,
        save_steps = 10000,
        save_total_limit = 1,
    )
    
    if method_name == 'CM':
        company_names = [item.lower() for item in list(pd.read_csv(company_filepath).Name.unique())]
        print('[Company name Masking] Number of company names:', len(company_names))
        data_collator = DataCollatorForCompanyNameMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob, company_names=company_names)
    
    elif method_name == 'SM':
        print('[Subword Masking]')
        data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = mlm_prob)
    
    elif method_name == 'WWM':
        print('[Whole Word Masking]')
        data_collator = DataCollatorForWholeWordMask(tokenizer = tokenizer, mlm = True, mlm_probability = mlm_prob)
    
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset,
    )
    
    print('[Post-training (Company name - MLM)] Training..')
    trainer.train()
    
    tokenizer.save_pretrained(save_dir)
    trainer.save_model(save_dir)
    print('[Post-training (Company name - MLM)] Saved trained model at {}'.format(save_dir))