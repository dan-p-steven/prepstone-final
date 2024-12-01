# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Cache libraries
import joblib
import os
import time

CSV_PATH = './data/financial_transactions.csv'
CSV_RESIZED_PATH = './data/financial_transactions_resized.csv'
DF_CACHE = './data/df_cache.pkl'


def read_dataset(dataset_file_path:str):
    '''
    Rename columns then return dataframe. Our dataset was too large (around
    6,500,000 rows) to run on our machines. So caching is used to speed up 
    execution.
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

def resize_dataset(df: pd.DataFrame):
    ''' 
    Resize the dataset to have 500,000 entries.
    '''

    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    X_resized, X_test, y_resized, y_test = train_test_split(X, y,
                                                            test_size=0.92,
                                                            stratify=y,
                                                            random_state=777)
    X_resized['isFraud'] = y_resized

    # Write to file
    X_resized.to_csv(CSV_RESIZED_PATH, index=False)

if __name__ == "__main__":

    df = read_dataset(CSV_PATH)
    resize_dataset(df)
