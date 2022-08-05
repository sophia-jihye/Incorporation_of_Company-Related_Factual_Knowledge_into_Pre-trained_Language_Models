import os, copy
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from sec_api import ExtractorApi

root_dir = '/media/dmlab/My Passport/DATA/ComBERT/data'
url_filepath =  os.path.join(root_dir, 'urls_2016_by_sec_api_FullTextSearchApi_with_cik_of_sec_cik_mapper_3999.csv')

save_filepath_format = os.path.join(root_dir, 'Item1s_2016_by_sec_api_ExtractorApi_with_cik_of_sec_cik_mapper_{}.csv')

api_key_filepath = 'API_Key.txt'
with open(api_key_filepath, "r") as f:
    api_key = f.read()
    
if __name__ == '__main__':
    extractorApi = ExtractorApi(api_key)
    
    df = pd.read_csv(url_filepath)
    print('Number of urls: {}'.format(len(df)))
    
    return_type = 'text'
    for item_num in ['1']:
        appended_df = copy.copy(df)
        colname = 'item_{}'.format(item_num)
        appended_df[colname] = appended_df['filingUrl'].progress_apply(lambda x: extractorApi.get_section(x, item_num, return_type))    
        
        result_df = appended_df[appended_df[colname]!='']        
        save_filepath = save_filepath_format.format(len(result_df))
        result_df.to_csv(save_filepath, index=False)
        print('Created {}'.format(save_filepath))
        
        save_undefined_filepath_format = os.path.join(root_dir, 'Item1s_2016_by_sec_api_ExtractorApi_with_cik_of_sec_cik_mapper_undefined_{}.csv')
        undefined_result_df = appended_df[appended_df[colname]=='']
        if len(undefined_result_df) > 0:
            save_undefined_filepath = save_undefined_filepath_format.format(len(undefined_result_df))
            undefined_result_df.to_csv(save_undefined_filepath, index=False)
            print('Created {}'.format(save_undefined_filepath))