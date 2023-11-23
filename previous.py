import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def select_feature_var(df, threshold=0.001):
    print('Initial shape: {}'.format(df.shape))
    for col in df.select_dtypes('object').columns:
        value_counts = df[col].value_counts(normalize=True)
        if value_counts.var() < threshold:
            df.drop(col, axis=1, inplace=True)

    for col in df.select_dtypes('number').columns:
        if df[col].var() < threshold:
            df.drop(col, axis=1, inplace=True)
    print('Final shape: {}'.format(df.shape))
    return df



# Load data
previous = pd.read_csv('raw-data/dseb63_previous_application.csv')

# Select features
previous = select_feature_var(previous)

# Missing indicator
# Identify columns with missing values
missing_columns = previous.columns[previous.isnull().any()].tolist()
new_cols = [col + '_MISSING' for col in missing_columns]

# Create missing indicator
indicator = MissingIndicator(features='missing-only')
indicator.fit(previous[missing_columns])
indicator = pd.DataFrame(indicator.transform(previous[missing_columns]),
                         columns=new_cols, index=previous.index)

# Concatenate with original data
previous = pd.concat([previous, indicator], axis=1)

# Encode categorical features
previous = pd.get_dummies(previous, drop_first=True)

# Aggregate features by SK_ID_CURR
previous_agg = previous.groupby('SK_ID_CURR').mean()

# Flatten column names
# previous_agg.columns = ['_'.join(col).strip() for col in previous_agg.columns.values]

# Drop SK_ID_PREV
previous_agg.drop(['SK_ID_PREV'], axis=1, inplace=True)
previous_agg

previous_agg.to_csv('processed_previous_2111.csv', index=True)