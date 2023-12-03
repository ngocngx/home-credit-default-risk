import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import MissingIndicator

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from functions import *
from optbinning import OptimalBinning, Scorecard, BinningProcess

def create_feature(df):
    new_features = {
        'APP_CREDIT_PERC': df['AMT_APPLICATION'] / df['AMT_CREDIT'],
        'APP_CREDIT_RATIO': df.apply(lambda x: x['AMT_APPLICATION'] / x['AMT_CREDIT'] if x['AMT_CREDIT'] != 0 else np.nan, axis=1),
        'ANNUITY_PAYMENT_PRODUCT': df['AMT_ANNUITY'] * df['CNT_PAYMENT'],
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
    }

    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df

# Load data
previous = pd.read_csv('raw-data/dseb63_previous_application.csv')
previous.sort_values(['SK_ID_PREV', 'DAYS_DECISION'], inplace=True)
previous.set_index('SK_ID_CURR', inplace=True)

# Create features
previous = create_feature(previous)

# One-hot encoding
previous, cat_cols = one_hot_encoder(previous, nan_as_category=True)
print('After one-hot encoding: {}'.format(previous.shape))

# Replace positive inf with nan
previous = previous.replace([np.inf, -np.inf], np.nan)

# Aggregate
previous_agg = previous.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
previous_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in previous_agg.columns.tolist()])
previous_agg['PREV_COUNT'] = previous.groupby('SK_ID_CURR').size()

# Target
target = pd.read_csv('processed-data/target.csv')
target.set_index('SK_ID_CURR', inplace=True)
y_train = target[target.index.isin(previous_agg.index)]['TARGET']

# Split train and test
previous_train = previous_agg[previous_agg.index.isin(target.index)]
previous_test = previous_agg[~previous_agg.index.isin(target.index)]
print('Previous train shape: {}'.format(previous_train.shape))

# Drop columns with 1 unique value
cols_to_drop = [col for col in previous_train.columns if previous_train[col].nunique() <= 1]
previous_train.drop(cols_to_drop, axis=1, inplace=True)
previous_test.drop(cols_to_drop, axis=1, inplace=True)

# Binning process
variable_names = previous_train.columns.tolist()
binning_process = BinningProcess(variable_names, categorical_variables=cat_cols, 
                                 max_n_prebins=30)
binning_process.fit(previous_train, y_train)

# Transform train and test
previous_train_binned = binning_process.transform(previous_train, metric_missing=0.05)
previous_train_binned.index = previous_train.index
previous_test_binned = binning_process.transform(previous_test, metric_missing=0.05)
previous_test_binned.index = previous_test.index

# Sanitize columns
previous_train_binned = sanitize_columns(previous_train_binned)
previous_test_binned = sanitize_columns(previous_test_binned)

# Select features
selected_features = select_features_lightgbm(previous_train_binned, y_train, threshold=1)
print('Number of selected features: {}'.format(len(selected_features)))
print('Top 10 features:', selected_features.sort_values(ascending=False)[:10].index.tolist())
previous_train = previous_train_binned[selected_features.index]
previous_test = previous_test_binned[selected_features.index]

# Concatenate train and test
previous = pd.concat([previous_train, previous_test], axis=0)

# Drop sk_id_prev
cols_to_drop = [col for col in previous.columns if 'SK_ID_PREV' in col]
previous.drop(cols_to_drop, axis=1, inplace=True)

# Save data
previous.to_csv('processed-data/processed_previous_application.csv')
