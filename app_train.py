import numpy as np
import pandas as pd
from datetime import date

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import MissingIndicator

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from functions import *


def create_features(df):
    # Calculate new features
    new_columns = {
        'DAYS_EMPLOYED_PERC': df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'],
        'INCOME_CREDIT_PERC': df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'],
        'INCOME_PER_PERSON': df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'],
        'ANNUITY_INCOME_PERC': df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'],
        'PAYMENT_RATE': df['AMT_ANNUITY'] / df['AMT_CREDIT'],
        'CHILDREN_RATIO': df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS'],
        'CREDIT_TO_ANNUITY_RATIO': df['AMT_CREDIT'] / df['AMT_ANNUITY'],
        'CREDIT_TO_GOODS_RATIO': df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'],
        'ANNUITY_TO_INCOME_RATIO': df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'],
        'CREDIT_TO_INCOME_RATIO': df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'],
        'INCOME_TO_EMPLOYED_RATIO': df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED'],
        'INCOME_TO_BIRTH_RATIO': df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH'],
        'EMPLOYED_TO_BIRTH_RATIO': df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'],
        'ID_TO_BIRTH_RATIO': df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH'],
        'CAR_TO_BIRTH_RATIO': df['OWN_CAR_AGE'] / df['DAYS_BIRTH'],
        'CAR_TO_EMPLOYED_RATIO': df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED'],
        'PHONE_TO_BIRTH_RATIO': df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    }

    # Add new columns to the DataFrame all at once
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return df

# Load data
train = pd.read_csv('raw-data/dseb63_application_train.csv')
train.drop('Unnamed: 0', axis=1, inplace=True)
train.set_index('SK_ID_CURR', inplace=True)

test = pd.read_csv('raw-data/dseb63_application_test.csv')
test.drop('Unnamed: 0', axis=1, inplace=True)
test.set_index('SK_ID_CURR', inplace=True)

# Merge train and test
train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], axis=0)

# Replace positive if DAYS feature with nan
days_cols = [col for col in df.columns if 'DAYS' in col]
for col in days_cols:
    posive_mask = df[col] >= 0
    df.loc[posive_mask, col] = np.nan

df = df.replace(['XNA', 'Unknown', 'not specified'], np.nan)
print(f'df shape: {df.shape}')

# Missing Imputer
missing_cols = [col for col in df.columns if df[col].isnull().any()]
new_cols = [col + '_missing' for col in missing_cols]
print(f'Number of missing columns: {len(missing_cols)}')

# MissingIndicator
mi = MissingIndicator()
mi.fit(df)
new_df = pd.DataFrame(mi.transform(df), columns=new_cols, 
                      index=df.index)
print(f'new_df shape: {new_df.shape}')

# Concat
df = pd.concat([df, new_df], axis=1)
print(f'df shape: {df.shape}')

# Create features
df = create_features(df)

# Split train and test
train = df[df['is_train'] == 1]
test = df[df['is_train'] == 0]
train = train.drop('is_train', axis=1)
test = test.drop('is_train', axis=1)

# Target
y = train['TARGET']
train = train.drop('TARGET', axis=1)
test = test.drop('TARGET', axis=1)

# Fill nan by mode for categorical features and mean for numerical features
cat_cols = train.select_dtypes(include=['object']).columns
num_cols = train.select_dtypes(exclude=['object']).columns

train_mode = train[cat_cols].mode().iloc[0]
train_mean = train[num_cols].mean()

for col in cat_cols:
    train[col] = train[col].fillna(train_mode[col])
    test[col] = test[col].fillna(train_mode[col])

for col in num_cols:
    train[col] = train[col].fillna(train_mean[col])
    test[col] = test[col].fillna(train_mean[col])

# OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(train[cat_cols])
train_cat = ohe.transform(train[cat_cols]).toarray()
test_cat = ohe.transform(test[cat_cols]).toarray()

train_cat_df = pd.DataFrame(train_cat, columns=ohe.get_feature_names_out(cat_cols),
                            index=train.index)
test_cat_df = pd.DataFrame(test_cat, columns=ohe.get_feature_names_out(cat_cols),
                           index=test.index)

# Drop and concat
train = train.drop(cat_cols, axis=1)
test = test.drop(cat_cols, axis=1)

train = pd.concat([train, train_cat_df], axis=1)
test = pd.concat([test, test_cat_df], axis=1)

# Merge with previous application
previous = pd.read_csv('drafts/processed_previous_2111.csv')
train = pd.merge(train, previous, on='SK_ID_CURR', how='left')
test = pd.merge(test, previous, on='SK_ID_CURR', how='left')
print('Number of nulls in train: ', train.isnull().sum().sum())
print('Number of nulls in test: ', test.isnull().sum().sum())

# Merge with bureau
bureau = pd.read_csv('processed-data/processed_bureau.csv')
train = pd.merge(train, bureau, on='SK_ID_CURR', how='left')
test = pd.merge(test, bureau, on='SK_ID_CURR', how='left')
print('Number of nulls in train: ', train.isnull().sum().sum())
print('Number of nulls in test: ', test.isnull().sum().sum())

# Replace inf
train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)

# Fill nan
train = train.fillna(train.mean())
test = test.fillna(train.mean())
print('Number of nulls in train: ', train.isnull().sum().sum())
print('Number of nulls in test: ', test.isnull().sum().sum())

# Set index
train.set_index('SK_ID_CURR', inplace=True)
test.set_index('SK_ID_CURR', inplace=True)


# Scale numerical features
robust_scaler = RobustScaler(quantile_range=(1, 99))
train = pd.DataFrame(robust_scaler.fit_transform(train), columns=train.columns, index=train.index)
test = pd.DataFrame(robust_scaler.transform(test), columns=test.columns, index=test.index)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
train = pd.DataFrame(minmax_scaler.fit_transform(train), columns=train.columns, index=train.index)
test = pd.DataFrame(minmax_scaler.transform(test), columns=test.columns, index=test.index)

print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')
print(f'y shape: {y.shape}')

# Select features
selected_features = select_features_rf(train, y, threshold=0.0005)
train = train[selected_features.index]
test = test[selected_features.index]

print("Final train shape: ", train.shape)

# Train
log_reg = LogisticRegression(class_weight='balanced', solver='newton-cholesky', max_iter=750, C=100)

# Cross validate
scores = cross_val_score(log_reg, train, y, cv=5, scoring='roc_auc')
print(f'ROC AUC scores: {scores}')
print(f'ROC AUC mean: {scores.mean()}, GINI: {2*scores.mean() - 1}')

# Fit
log_reg.fit(train, y)

# Predict
y_pred = log_reg.predict_proba(test)[:, 1]
submission = pd.DataFrame(index=test.index, data={'TARGET': y_pred})
submission.sort_index(inplace=True)

# Save submission with date
today = date.today()
submission.to_csv(f'submissions/submission_{today}.csv')
