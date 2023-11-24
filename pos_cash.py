import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from functions import *

# Load data
pos_cash = pd.read_csv('raw-data/dseb63_POS_CASH_balance.csv')
print('Initial shape: {}'.format(pos_cash.shape))

# One-hot encoding for categorical columns with get_dummies
pos_cash, cat_cols = one_hot_encoder(pos_cash, nan_as_category= True)

# General aggregations
pos_cash.drop(['SK_ID_PREV'], axis= 1, inplace = True)
pos_cash_agg = pos_cash.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
pos_cash_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_cash_agg.columns.tolist()])
pos_cash_agg['POS_COUNT'] = pos_cash.groupby('SK_ID_CURR').size()

# Print shape after one-hot encoding
print('After one-hot encoding: {}'.format(pos_cash_agg.shape))
print('Null values: {}'.format(pos_cash_agg.isnull().values.sum()))

# Add missing indicator
missing_cols = [col for col in pos_cash_agg.columns if pos_cash_agg[col].isnull().any()]
new_cols = [col + '_MISSING' for col in missing_cols]
mi = MissingIndicator()
mi.fit(pos_cash_agg[missing_cols])
missing_df = pd.DataFrame(mi.transform(pos_cash_agg[missing_cols]), columns=new_cols, index=pos_cash_agg.index)
pos_cash_agg = pd.concat([pos_cash_agg, missing_df], axis=1)
print('After missing indicator: {}'.format(pos_cash_agg.shape))

# Merge with target
pos_cash_copy = pos_cash_agg.copy()
target = pd.read_csv('processed-data/target.csv')
pos_cash_agg = target.merge(pos_cash_agg, how='left', on='SK_ID_CURR')
pos_cash_agg.set_index('SK_ID_CURR', inplace=True)

# Fill missing values
imputer = SimpleImputer(strategy='mean')
pos_cash_agg = pd.DataFrame(imputer.fit_transform(pos_cash_agg), columns=pos_cash_agg.columns,
                      index=pos_cash_agg.index)
print('Null values: {}'.format(pos_cash_agg.isnull().values.sum()))

# Select features
selected_features = select_features_rf(pos_cash_agg.drop(['TARGET'], axis=1), 
                                       pos_cash_agg['TARGET'], threshold=0.001)
print('Number of selected features: {}'.format(len(selected_features.index.tolist())))
pos_cash = pos_cash_copy[selected_features.index.tolist()]

# Save
pos_cash.to_csv('processed-data/processed_pos_cash.csv')

