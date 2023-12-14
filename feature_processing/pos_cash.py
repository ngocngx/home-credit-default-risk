import numpy as np
import pandas as pd
from optbinning import BinningProcess

import os
import sys
sys.path.append(os.getcwd())
from functions.functions import *

def create_feature(df):
    new_features = {
        'CNT_INSTALMENT_DIFF': df['CNT_INSTALMENT'] - df['CNT_INSTALMENT_FUTURE'],
        'DPD': df['SK_DPD'] - df['SK_DPD_DEF'],
        'DPD_FLAG': df['SK_DPD'] > 0,
        'DPD_DEF_FLAG': df['SK_DPD_DEF'] > 0,
        'OVERDUE_RATIO': df['SK_DPD'] / df['CNT_INSTALMENT'],
        'OVERDUE_DEF_RATIO': df['SK_DPD_DEF'] / df['CNT_INSTALMENT'],
    }

    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df

# Load data
pos_cash = pd.read_csv('raw-data/dseb63_POS_CASH_balance.csv')
pos_cash.set_index('SK_ID_CURR', inplace=True)
print('Initial shape: {}'.format(pos_cash.shape))

# Create features
pos_cash = create_feature(pos_cash)
print('After creating features: {}'.format(pos_cash.shape))

# Replace positive inf with max, negative inf with min
pos_cash.replace([np.inf, -np.inf], [pos_cash.max(), pos_cash.min()], inplace=True)

# Filter last month
pos_cash_filter = pos_cash.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE']).groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
pos_cash_filter['COMPLETED_FLAG'] = pos_cash_filter['NAME_CONTRACT_STATUS'] == 'Completed'
pos_cash_filter['COMPLETED_COUNT'] = pos_cash_filter.groupby('SK_ID_CURR')['COMPLETED_FLAG'].transform('sum')
pos_cash_filter['COMPLETED_COUNT'] = pos_cash_filter['COMPLETED_COUNT'].fillna(0)
pos_cash_filter['OVERDUE_FLAG'] = pos_cash_filter.apply(lambda x: x['SK_DPD_DEF'] > 0 and x['CNT_INSTALMENT_FUTURE'] > 0, axis=1)
pos_cash_filter['OVERDUE_COUNT'] = pos_cash_filter.groupby('SK_ID_CURR')['OVERDUE_FLAG'].transform('sum')

# One-hot encoding for categorical columns with get_dummies
pos_cash_filter, cat_cols = one_hot_encoder(pos_cash_filter, nan_as_category= True)
print('After one-hot encoding: {}'.format(pos_cash_filter.shape))

# Aggregate
pos_cash_filter_agg = pos_cash_filter.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'var','sum'])
pos_cash_filter_agg.columns = pd.Index(['POS_FILTER_' + e[0] + "_" + e[1].upper() for e in pos_cash_filter_agg.columns.tolist()])
print('After aggregation: {}'.format(pos_cash_filter_agg.shape))

# One-hot encoding for categorical columns with get_dummies
pos_cash, cat_cols = one_hot_encoder(pos_cash, nan_as_category= True)
print('After one-hot encoding: {}'.format(pos_cash.shape))

# Aggregate
pos_cash.drop('SK_ID_PREV', axis=1, inplace=True)
pos_cash_agg = pos_cash.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'var','sum'])
pos_cash_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_cash_agg.columns.tolist()])
print('After aggregation: {}'.format(pos_cash_agg.shape))
print('Null values: {}'.format(pos_cash_agg.isnull().values.sum()))

# Merge with pos_cash_filter
pos_cash_agg = pos_cash_agg.merge(pos_cash_filter_agg, how='left', on='SK_ID_CURR')

# Check duplicated columns
pos_cash_agg = pos_cash_agg.loc[:, ~pos_cash_agg.columns.duplicated()]
print('After removing duplicated columns: {}'.format(pos_cash_agg.shape))

# Target
target = pd.read_csv('processed-data/target.csv')
target.set_index('SK_ID_CURR', inplace=True)
y_train = target[target.index.isin(pos_cash_agg.index)]['TARGET']

# Split train and test
pos_cash_train = pos_cash_agg[pos_cash_agg.index.isin(target.index)]
pos_cash_test = pos_cash_agg[~pos_cash_agg.index.isin(target.index)]
print('POS_CASH train shape: {}'.format(pos_cash_train.shape))

# Drop columns with 1 unique value
cols_to_drop = [col for col in pos_cash_train.columns if pos_cash_train[col].nunique() <= 1]
pos_cash_train.drop(cols_to_drop, axis=1, inplace=True)
pos_cash_test.drop(cols_to_drop, axis=1, inplace=True)
print('After removing columns with 1 unique value: {}'.format(pos_cash_train.shape))

# Binning process
variable_names = pos_cash_train.columns.tolist()
binning_process = BinningProcess(variable_names)
binning_process.fit(pos_cash_train, y_train)

# Transform train and test
pos_cash_train_binned = binning_process.transform(pos_cash_train, metric_missing=0.05)
pos_cash_train_binned.columns = [pos_cash_train.columns.tolist()[i] + '_BINNED' for i in range(len(pos_cash_train.columns.tolist()))]
pos_cash_train_binned.index = pos_cash_train.index
pos_cash_test_binned = binning_process.transform(pos_cash_test, metric_missing=0.05)
pos_cash_test_binned.columns = [pos_cash_test.columns.tolist()[i] + '_BINNED' for i in range(len(pos_cash_test.columns.tolist()))]
pos_cash_test_binned.index = pos_cash_test.index

# Merge original and binned
pos_cash_train = pd.concat([pos_cash_train, pos_cash_train_binned], axis=1)
pos_cash_test = pd.concat([pos_cash_test, pos_cash_test_binned], axis=1)

# Select columns
print(f'Selecting features...')
selected_features = select_features_iv(pos_cash_train, y_train, threshold=0.02)
print(f'Number of selected features: {len(selected_features)}')
pos_cash_train = pos_cash_train[selected_features]
pos_cash_test = pos_cash_test[selected_features]

# Concatenate train and test
pos_cash = pd.concat([pos_cash_train, pos_cash_test], axis=0)

# Drop sk_id_prev
cols_to_drop = [col for col in pos_cash.columns if 'SK_ID_PREV' in col]
pos_cash.drop(cols_to_drop, axis=1, inplace=True)

# Save data
pos_cash.to_csv('processed-data/processed_pos_cash.csv')
