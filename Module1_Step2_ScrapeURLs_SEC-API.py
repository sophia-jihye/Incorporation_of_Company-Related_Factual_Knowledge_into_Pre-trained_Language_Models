import os
import pandas as pd
from tqdm import tqdm
from sec_api import FullTextSearchApi

root_dir = '/media/dmlab/My Passport/DATA/ComBERT/data'
company_info_filepath = os.path.join(root_dir, 'company_info_sec_cik_mapper_12057_20220802.csv')
save_filepath_format = os.path.join(root_dir, 'urls_2016_by_sec_api_FullTextSearchApi_with_cik_of_sec_cik_mapper_{}.csv')

api_key_filepath = 'API_Key.txt'
with open(api_key_filepath, "r") as f:
    api_key = f.read()

start_date, end_date = '2016-01-01', '2016-12-31'
    
if __name__ == '__main__':
    fullTextSearchApi = FullTextSearchApi(api_key=api_key)

    companies_df = pd.read_csv(company_info_filepath)
    companies_df = companies_df.astype({'CIK':'str'})
    print('Number of CIKs: {}'.format(len(companies_df['CIK'].unique())))
    
    records, err_records = [], []
    for cik in tqdm(list(companies_df['CIK'].unique())):
        query = {
            "query": '',
            'ciks': [cik],
            "formTypes": ['10-K'],
            "startDate": start_date,
            "endDate": end_date,
        }
        
        try:
            filings = fullTextSearchApi.get_filings(query)
        except Exception as e:
            err_records.append((cik, start_date, end_date, str(e)))
            print('[{}] {}'.format(cik, str(e)))
            continue
            
        for item in filings['filings']:
            if item['type'] == '10-K':
                records.append(item)

    url_df = pd.DataFrame(records)
    print('Number of documents: {}'.format(len(url_df)))
    
    save_filepath = save_filepath_format.format(len(url_df))
    url_df.to_csv(save_filepath, index=False)
    print('Created {}'.format(save_filepath))
    
    err_filepath_format = os.path.join(root_dir, 'error_{}.csv')
    if len(err_df) > 0:
        err_df = pd.DataFrame(err_records, columns=['CIK', 'start_date', 'end_date', 'Exception'])
        err_filepath = err_filepath_format.format(len(err_df))
        err_df.to_csv(err_filepath, index=False)
        print('Created {}'.format(err_filepath))