import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import MissingIndicator

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from functions import *

# Load data
previous = pd.read_csv('raw-data/dseb63_previous_application.csv')
previous.set_index('SK_ID_CURR', inplace=True)

# Create features

# Drop SK_ID_PREV
previous.drop(['SK_ID_PREV'], axis= 1, inplace = True)

# Merge with target
print('Merge with target')
target = pd.read_csv('raw-data/dseb63_application_train.csv', usecols=['SK_ID_CURR', 'TARGET'])
target.set_index('SK_ID_CURR', inplace=True)

previous = previous.merge(target, left_index=True, right_index=True, how='left')
print('After merge: {}'.format(previous.shape))

# Split train and test
prev_train = previous[previous['TARGET'].notnull()]
y_train = prev_train['TARGET']
prev_train.drop('TARGET', axis=1, inplace=True)

prev_test = previous[previous['TARGET'].isnull()]
prev_test.drop('TARGET', axis=1, inplace=True)

# Astype to category
print('Astype to category')
cat_cols = prev_train.select_dtypes('object').columns
prev_train[cat_cols] = prev_train[cat_cols].astype('category')
prev_test[cat_cols] = prev_test[cat_cols].astype('category')

# WoETransformer
woe_transformer = WoETransformer(bins=20)
woe_transformer.fit(prev_train, y_train)

# Transform
prev_train = woe_transformer.transform(prev_train)
prev_test = woe_transformer.transform(prev_test)

# Concat
prev = pd.concat([prev_train, prev_test], axis=0)

# Astype to float
print('Astype to float')
prev = prev.astype(float)

# Aggregate
print('Aggregate')
prev_agg = prev.groupby('SK_ID_CURR').agg(['mean', 'var'])
prev_agg.columns = ['_'.join(col).strip() for col in prev_agg.columns.values]

# Save
prev_agg.to_csv('processed-data/processed_previous_application.csv')