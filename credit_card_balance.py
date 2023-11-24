import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from functions import *

# Load data
cc = pd.read_csv('raw-data/dseb63_credit_card_balance.csv')
cc.set_index('SK_ID_CURR', inplace=True)
print('Initial shape: {}'.format(cc.shape))

# One-hot encoding for categorical columns with get_dummies
cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

# General aggregations
cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

# Count credit card lines
cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

# Print shape after one-hot encoding
print('After one-hot encoding: {}'.format(cc_agg.shape))
print('Null values: {}'.format(cc_agg.isnull().values.sum()))

# Add missing indicator
missing_cols = [col for col in cc_agg.columns if cc_agg[col].isnull().any()]
new_cols = [col + '_MISSING' for col in missing_cols]
mi = MissingIndicator()
mi.fit(cc_agg[missing_cols])
missing_df = pd.DataFrame(mi.transform(cc_agg[missing_cols]), columns=new_cols, index=cc_agg.index)
cc_agg = pd.concat([cc_agg, missing_df], axis=1)
print('After missing indicator: {}'.format(cc_agg.shape))

# Merge with target
cc_copy = cc_agg.copy()
target = pd.read_csv('processed-data/target.csv')
cc_agg = target.merge(cc_agg, how='left', on='SK_ID_CURR')
cc_agg.set_index('SK_ID_CURR', inplace=True)

# Fill missing values
imputer = SimpleImputer(strategy='mean')
cc_agg = pd.DataFrame(imputer.fit_transform(cc_agg), columns=cc_agg.columns,
                      index=cc_agg.index)
print('Null values: {}'.format(cc_agg.isnull().values.sum()))

# Select features
selected_features = select_features_rf(cc_agg.drop(['TARGET'], axis=1), 
                                       cc_agg['TARGET'], threshold=0.005)
print('Selected features: {}'.format(selected_features.index.tolist()))
cc = cc_copy[selected_features.index.tolist()]

# Save
cc.to_csv('processed-data/processed_credit_card_balance.csv')
print(cc.head())
