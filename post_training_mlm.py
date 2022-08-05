from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

from custom_data_collator import DataCollatorForWholeWordMask

def train(tokenizer, model, dataset, save_dir, company_names=None, is_masking_company_name_first=None):
    mlm_prob=0.15
    model.train()
    
    training_args = TrainingArguments(
        output_dir = save_dir,
        num_train_epochs = 5,   
        per_device_train_batch_size = 4,
        save_steps = 10000,
        save_total_limit = 1,
    )
    
    if company_names is None:
        print('company_names is None')
        data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = mlm_prob)
    else:
        print('Number of company names:', len(company_names))
        data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob, \
                             company_names=company_names, is_masking_company_name_first=is_masking_company_name_first)

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