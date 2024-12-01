# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Cache libraries
import joblib
import os
import time

from sklearn.model_selection import train_test_split

CSV_PATH = './data/financial_transactions.csv'
DF_CACHE = './data/df_cache.pkl'

def exploratory_data_analysis(dataset_file_path:str):
    '''
    Perform EDA on the dataset. Rename columns, create graphs, then return 
    dataframe.
    '''

    # Check if the dataframe is cached
    if os.path.exists(DF_CACHE):

        # Cache exists, load dataframe from cache
        df = joblib.load(DF_CACHE)
    else:

        df = pd.read_csv(dataset_file_path)

        # Better fitting column names, removes some of the typos in the original 
        # dataset column names
        df.rename(columns={'type':'paymentType', 'nameOrig':'accSender',
                           'oldbalanceOrg':'oldBalanceSender', 'newbalanceOrig':
                           'newBalanceSender', 'nameDest':'accRecipient',
                           'oldbalanceDest':'oldBalanceRecipient',
                           'newbalanceDest':'newBalanceRecipient'})

        # Cache the dataframe for quick repeated script execution
        joblib.dump(df, DF_CACHE)

    return df

def data_preprocessing():
    pass


if __name__ == "__main__":

    start = time.time()
    df = exploratory_data_analysis(CSV_PATH)


    end = time.time() - start
    print (f'Time: {end:.2f}s')

