import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from functions import *
from optbinning import OptimalBinning, Scorecard, BinningProcess

def create_feature(df):
    new_features = {
        'CNT_INSTALMENT_FUTURE': df['CNT_INSTALMENT'] - df['CNT_INSTALMENT_FUTURE'],
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

# Replace positive inf with nan
pos_cash = pos_cash.replace([np.inf, -np.inf], np.nan)

# One-hot encoding for categorical columns with get_dummies
pos_cash, cat_cols = one_hot_encoder(pos_cash, nan_as_category= True)
print('After one-hot encoding: {}'.format(pos_cash.shape))

# Aggregate
pos_cash.drop('SK_ID_PREV', axis=1, inplace=True)
pos_cash_agg = pos_cash.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
pos_cash_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_cash_agg.columns.tolist()])
print('After aggregation: {}'.format(pos_cash_agg.shape))
print('Null values: {}'.format(pos_cash_agg.isnull().values.sum()))

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
pos_cash_train_binned.index = pos_cash_train.index
pos_cash_test_binned = binning_process.transform(pos_cash_test, metric_missing=0.05)
pos_cash_test_binned.index = pos_cash_test.index

# Sanitize columns
pos_cash_train_binned = sanitize_columns(pos_cash_train_binned)
pos_cash_test_binned = sanitize_columns(pos_cash_test_binned)

# Select features
selected_features = select_features_lightgbm(pos_cash_train_binned, y_train, threshold=1)
print('Number of selected features: {}'.format(len(selected_features)))
print('Top 10 features:', selected_features.sort_values(ascending=False)[:10].index.tolist())
pos_cash_train_binned = pos_cash_train_binned[selected_features.index]
pos_cash_test_binned = pos_cash_test_binned[selected_features.index]

# Concatenate train and test
pos_cash = pd.concat([pos_cash_train_binned, pos_cash_test_binned], axis=0)

# Drop sk_id_prev
cols_to_drop = [col for col in pos_cash.columns if 'SK_ID_PREV' in col]
pos_cash.drop(cols_to_drop, axis=1, inplace=True)

# Save data
pos_cash.to_csv('processed-data/processed_pos_cash.csv')


