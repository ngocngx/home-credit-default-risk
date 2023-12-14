import numpy as np
import pandas as pd
from optbinning import BinningProcess

import os
import sys
sys.path.append(os.getcwd())
from functions.functions import *

def create_feature(df):
    new_features = {
        'APP_CREDIT_PERC': df['AMT_APPLICATION'] / df['AMT_CREDIT'],
        'APP_CREDIT_RATIO': df.apply(lambda x: x['AMT_APPLICATION'] / x['AMT_CREDIT'] if x['AMT_CREDIT'] != 0 else np.nan, axis=1),
        'ANNUITY_PAYMENT_PRODUCT': df['AMT_ANNUITY'] * df['CNT_PAYMENT'],
        'INTEREST_SHARE': df['INTEREST']/df['AMT_CREDIT'],
        # Time-based Features
        'DAYS_DECISION_YEAR': df['DAYS_DECISION'] // 365,
        # Categorical Combinations
        'CONTRACT_CLIENT_TYPE': df['NAME_CONTRACT_TYPE'] + "_" + df['NAME_CLIENT_TYPE'],
        # Binary Indicators
        'HIGH_DOWN_PAYMENT': df['AMT_DOWN_PAYMENT'] > 10_000,
        # Missing Value Indicators
        'MISSING_AMT_GOODS_PRICE': df['AMT_GOODS_PRICE'].isnull().astype(int),
        # Ratio
        'RATIO_DOWN_PAYMENT_TO_CREDIT': df['AMT_DOWN_PAYMENT'] / df['AMT_CREDIT'],
        'RATIO_CREDIT_TO_GOODS': df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'],
        'RATIO_ANNUITY_TO_CREDIT': df['AMT_ANNUITY'] / df['AMT_CREDIT'],
        'RATIO_APPLICATION_TO_CREDIT': df['AMT_APPLICATION'] / df['AMT_CREDIT'],
        'RATIO_APPLICATION_TO_GOODS': df['AMT_APPLICATION'] / df['AMT_GOODS_PRICE'],
        'RATIO_DOWN_PAYMENT_TO_GOODS': df['AMT_DOWN_PAYMENT'] / df['AMT_GOODS_PRICE'],
        'RATIO_DOWN_PAYMENT_TO_ANNUITY': df['AMT_DOWN_PAYMENT'] / df['AMT_ANNUITY'],
        'RATIO_APPLICATION_TO_ANNUITY': df['AMT_APPLICATION'] / df['AMT_ANNUITY'],
        'RATIO_CREDIT_TO_ANNUITY': df['AMT_CREDIT'] / df['AMT_ANNUITY'],
        'RATIO_GOODS_TO_ANNUITY': df['AMT_GOODS_PRICE'] / df['AMT_ANNUITY'],
        'RATIO_APPLICATION_TO_DOWN_PAYMENT': df['AMT_APPLICATION'] / df['AMT_DOWN_PAYMENT'],
        'RATIO_CREDIT_TO_DOWN_PAYMENT': df['AMT_CREDIT'] / df['AMT_DOWN_PAYMENT'],
        'AMT_DIFF_PERCENT' : ((df['AMT_CREDIT'] - df['AMT_APPLICATION']) / df['AMT_APPLICATION']) * 100

    }

    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df

# Load data
previous = pd.read_csv('raw-data/dseb63_previous_application.csv')
previous.sort_values(['SK_ID_PREV', 'DAYS_DECISION'], inplace=True)
previous.set_index('SK_ID_CURR', inplace=True)
print('Initial shape: {}'.format(previous.shape))

# Replace XNA and XAP with nan
previous.replace('XNA', np.nan, inplace=True)

# Create features
previous = create_feature(previous)
print('After feature creation: {}'.format(previous.shape))

# One-hot encoding
previous, cat_cols = one_hot_encoder(previous, nan_as_category=True)
print('After one-hot encoding: {}'.format(previous.shape))

# Replace positive inf with max, negative inf with min
previous.replace([np.inf, -np.inf], [np.nan, np.nan], inplace=True)

print('Number of inf values: {}'.format(previous.select_dtypes(include=np.number).isin([np.inf, -np.inf]).sum().sum()))

# Aggregate
previous.drop(['SK_ID_PREV'], axis=1, inplace=True)
grouped_previous = previous.groupby('SK_ID_CURR')
previous_agg = grouped_previous.agg(['min', 'max', 'mean', 'var', 'sum'])
previous_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in previous_agg.columns.tolist()])
previous_agg['PREV_COUNT'] = grouped_previous.size()
previous_agg['PREV_LASTEST_Approved'] = grouped_previous['NAME_CONTRACT_STATUS_Approved'].last()
previous_agg['PREV_LASTEST_Refused'] = grouped_previous['NAME_CONTRACT_STATUS_Refused'].last()
previous_agg['PREV_LASTEST_Canceled'] = grouped_previous['NAME_CONTRACT_STATUS_Canceled'].last()
previous_agg['PREV_LASTEST_Unused'] = grouped_previous['NAME_CONTRACT_STATUS_Unused offer'].last()
previous_agg['PREV_APPROVED_PERC'] = grouped_previous['NAME_CONTRACT_STATUS_Approved'].mean()
previous_agg['PREV_REFUSED_PERC'] = grouped_previous['NAME_CONTRACT_STATUS_Refused'].mean()

# Target
target = pd.read_csv('processed-data/target.csv')
target.set_index('SK_ID_CURR', inplace=True)
y_train = target[target.index.isin(previous_agg.index)]['TARGET']

# Split train and test
previous_train = previous_agg[previous_agg.index.isin(target.index)]
previous_test = previous_agg[~previous_agg.index.isin(target.index)]
print('Previous train shape: {}'.format(previous_train.shape))

# Drop columns with 1 unique value
print('Dropping columns with 1 unique value...')
cols_to_drop = [col for col in previous_train.columns if previous_train[col].nunique() <= 1]
previous_train.drop(cols_to_drop, axis=1, inplace=True)
previous_test.drop(cols_to_drop, axis=1, inplace=True)
print('After dropping: {}'.format(previous_train.shape))

# Binning process
variable_names = previous_train.columns.tolist()
binning_process = BinningProcess(variable_names)
binning_process.fit(previous_train, y_train)

# Transform train and test
previous_train_binned = binning_process.transform(previous_train, metric_missing=0.05)
previous_train_binned.columns = [previous_train_binned.columns[i] + '_BINNED' for i in range(len(previous_train_binned.columns))]
previous_train_binned.index = previous_train.index
previous_test_binned = binning_process.transform(previous_test, metric_missing=0.05)
previous_test_binned.columns = [previous_test_binned.columns[i] + '_BINNED' for i in range(len(previous_test_binned.columns))]
previous_test_binned.index = previous_test.index

# Merge original and binned
previous_train_binned = pd.concat([previous_train, previous_train_binned], axis=1)
previous_test_binned = pd.concat([previous_test, previous_test_binned], axis=1)

# Sanitize columns
previous_train_binned = sanitize_columns(previous_train_binned)
previous_test_binned = sanitize_columns(previous_test_binned)

# Select features by IV
print('Selecting features...')
selected_features = select_features_iv(previous_train_binned, y_train, threshold=0.02)
previous_train_binned = previous_train_binned[selected_features]
previous_test_binned = previous_test_binned[selected_features]
print('Number of selected features: {}'.format(len(selected_features)))

# Concatenate train and test
previous = pd.concat([previous_train_binned, previous_test_binned], axis=0)

# Save data
previous.to_csv('processed-data/processed_previous_application.csv')
