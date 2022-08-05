import os, re, html
import pandas as pd
from nltk import sent_tokenize
from tqdm import tqdm

root_dir = '/media/dmlab/My Passport/DATA/ComBERT'
item1s_filepath = os.path.join(root_dir, 'data', 'Item1s_2016_by_sec_api_ExtractorApi_with_cik_of_sec_cik_mapper_3990.csv')
company_filepath = os.path.join(root_dir, 'data, 'company_info_sec_cik_mapper_12057_20220802.csv')

save_dir = os.path.join(root_dir, 'pt_data')
if not os.path.exists(save_dir): os.makedirs(save_dir)
save_filepath = os.path.join(save_dir, 'post_item1_converted_subnames_to_fullname_sentences_with_fullname.txt')

removal_list =  "‘, ’, ◇, ‘, ”,  ’, ', ·, \“, ·, △, ●,  , ■, (, ), \", >>, `, /, -,∼,=,ㆍ<,>, .,?, !,【,】, …, ◆,%"
def clean(sent):
    sent = html.unescape(sent)
    sent = sent.translate(str.maketrans(removal_list, ' '*len(removal_list)))
    sent = re.sub("\s+", " ", sent)
    sent = sent.lower()
    return sent

def trim(sent):
    sent = re.sub("\s+", " ", sent)
    sent = sent.strip()
    return sent

def get_fullname(cik):
    try:
        return company_df[company_df['CIK']==cik].iloc[0]['Name']
    except:
        print('No full name for cik {}'.format(cik))
        return None
    
def subnames_of_company_name(fullname): 
    strings = fullname.split(' ')
    subnames = [' '.join(strings[:i+1]) for i in range(len(strings))] # ['amazon', 'amazon com', 'amazon com inc' ]
    subnames.reverse() # ['amazon com inc', 'amazon com', 'amazon']
    subnames.extend(['we', 'the company'])    
    subnames = [item.strip() for item in subnames if item.strip() != '']
    return subnames

def replace_subnames_to_target(sent, subnames, target):
    for sub in subnames:
        if sub in sent:
            sent = sent.replace(sub, '[TO-BE-REPLACED]')
    sent = sent.replace('[TO-BE-REPLACED]', target)
    return sent

if __name__ == '__main__': 
    company_df = pd.read_csv(company_filepath)
    company_df = company_df.astype({"CIK": int}, errors='raise')
    
    item_df = pd.read_csv(item1s_filepath)
    item_df = item_df.astype({"cik": int}, errors='raise')
    
    cnt = 0
    with open(save_filepath, 'w') as output_file:
        for cik, doc in tqdm(item_df[['cik', 'item_1']].values):
            fullname = get_fullname(cik)
            if fullname is None: 
                continue

            fullname = fullname.lower()
            for sent in sent_tokenize(doc):
                sent = clean(sent)
                sent = replace_subnames_to_target(sent, subnames_of_company_name(fullname), fullname)

                # 원 텍스트에 기업명이 등장한 케이스만 post-training에 사용
                if fullname in sent: 
                    sent = trim(sent)
                    output_file.write('{}\n\n'.format(sent))
                    cnt += 1
        output_file.write('[EOD]')
    print('Number of total sentences including company fullnames: {}'.format(cnt))
    print('Created {}'.format(save_filepath))
    