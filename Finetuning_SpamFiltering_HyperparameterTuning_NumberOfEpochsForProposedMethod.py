from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm 
tqdm.pandas()
import pandas as pd
import os, torch, copy, shutil, re, argparse
import numpy as np

from transformers_helper import load_tokenizer_and_model
from CustomDataset import CustomDataset, encode_for_inference
import finetuning_classification, reports

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='')
args = parser.parse_args()
num_train_epochs = args.epoch

root_dir = '/home/jihyeparkk/DATA/ComBERT' 
model_save_dir = os.path.join(root_dir, 'temp{}'.format(num_train_epochs))

model_name_or_dirs = []
model_name_or_dirs.extend(sorted(glob(os.path.join(root_dir, 'models_post-trained', '*_CM'))))

test_filepath = os.path.join(root_dir, 'data_finetuning_spamFiltering', 'test_10000.csv')

def glob_re(pattern, strings):
    return list(filter(re.compile(pattern).match, strings))

def do_prepare_data(relabel_dict, filepath):
    df = pd.read_csv(filepath)[['text', 'label']]
    print('Loaded {}'.format(filepath))
    df['label'] = df['label'].apply(lambda x: relabel_dict[x])
    return df
    
def start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir, num_train_epochs):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, num_classes=num_classes, mode='classification')
    
    print('Getting data..\n')
    train_dataset = CustomDataset(tokenizer, train_texts, train_labels)
    val_dataset = CustomDataset(tokenizer, val_texts, val_labels)
    
    finetuning_classification.train(model, train_dataset, val_dataset, save_dir, num_train_epochs)
    tokenizer.save_pretrained(save_dir)

def start_test(device, model_name_or_dir, df, save_dir, postfix=''):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, mode='classification')
    model = model.to(device)
    
    print('Inferencing..\n')
    df['predicted_label'] = df['text'].progress_apply(lambda x: finetuning_classification.inference(model, *encode_for_inference(device, tokenizer, x)))
    
    # Save results
    df['correct'] = df.apply(lambda x: x.true_label==x.predicted_label, axis=1)
    labels, preds = df.true_label, df.predicted_label
    accuracy = len(df[df['correct']==True]) / len(df)

    csv_filepath = os.path.join(save_dir, 'results_{}.csv'.format(postfix))
    df.to_csv(csv_filepath, index=False)
    
    report_filepath = os.path.join(save_dir, 'classification_report_{}.csv'.format(postfix))
    reports.create_classification_report(labels, preds, accuracy, report_filepath)
    
    confusion_filepath = os.path.join(save_dir, 'confusion_matrix_{}.csv'.format(postfix))
    reports.create_confusion_matrix(labels, preds, confusion_filepath)
    
if __name__ == '__main__':        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relabel_dict = {'human':0, 'bot':1}
    num_classes = len(relabel_dict)    
    
    save_dir_format = os.path.join(root_dir, 'results_spamFiltering', '{}_epoch{}_seed{}')
    
    train_seed_nums = [0,1,2]
    for train_seed_num in train_seed_nums:
        train_filepaths_ = sorted(glob(os.path.join(root_dir, 'data_finetuning_spamFiltering', 'train_seed_{}'.format(train_seed_num), 'train_*.csv')))
        train_filepaths = glob_re(r'.*(_400.csv|_500.csv|_600.csv|_700.csv|_800.csv|_900.csv|_1000.csv|_2000.csv)', train_filepaths_)
        for train_filepath in train_filepaths:
            source_df = do_prepare_data(relabel_dict, train_filepath)
            train_df = source_df.iloc[:int(len(source_df)*0.8)]
            val_df = source_df.iloc[int(len(source_df)*0.8):]
            train_texts, val_texts = train_df['text'].values, val_df['text'].values
            train_labels, val_labels = train_df['label'].values, val_df['label'].values

            for model_name_or_dir in model_name_or_dirs:
                save_dir = save_dir_format.format(os.path.basename(model_name_or_dir), num_train_epochs, train_seed_num)
                if not os.path.exists(save_dir): os.makedirs(save_dir)

                start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, model_save_dir, num_train_epochs)

                ##### test #####
                test_df = do_prepare_data(relabel_dict, test_filepath)
                test_df.rename(columns = {'label' : 'true_label'}, inplace = True)
                start_test(device, model_save_dir, test_df, save_dir, \
                           postfix='train_{}'.format(os.path.basename(train_filepath.split('_')[-1].replace('.csv',''))))

                # To save memory, delete the finetuned model in `temp` directory once model training is finished
                try: shutil.rmtree(model_save_dir)
                except OSError as e: print("Error: %s - %s." % (e.filename, e.strerror))
