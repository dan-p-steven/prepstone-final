# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Cache libraries
import joblib
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

CSV_PATH = './data/financial_transactions.csv'
DF_CACHE = './data/df_cache.pkl'

XY_CACHE = './data/xy_cache.pkl'
SPLIT_CACHE = './data/split.pkl'


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

    # Print class imbalance
    print (df['isFraud'].value_counts())

    return df

def data_preprocessing(df: pd.DataFrame):

    # Define features and target variable
    if os.path.exists(XY_CACHE):
        data = joblib.load(XY_CACHE)
    else:

        data = {
                'X': df.drop('isFraud', axis=1),
                'y': df['isFraud']
                }

        joblib.dump(data, XY_CACHE)

    # We need to split first then scale
    if os.path.exists(SPLIT_CACHE):
        split = joblib.load(SPLIT_CACHE)
    else:
        split = {
                 'X_train': None,
                 'X_test': None,
                 'y_train': None,
                 'y_test': None
                 }

        split['X_train'], split['X_test'],\
        split['y_train'], split['y_test'] = train_test_split(data['X'], data['y'], 
                                                        test_size=0.3,
                                                        stratify=data['y'],
                                                        random_state=777)


        joblib.dump(split, SPLIT_CACHE)

    print (f'X {split["X_train"].shape} {split["X_test"].shape}')
    print (f'y {split["y_train"].shape} {split["y_test"].shape}')

    return split

if __name__ == "__main__":

    start = time.time()
    df = exploratory_data_analysis(CSV_PATH)
    end = time.time() - start
    print (f'EDA time: {end:.2f}s')



    start = time.time()
    #split = data_preprocessing(df)
    end = time.time() - start
    print (f'Data preprocessing time: {end:.2f}s')


