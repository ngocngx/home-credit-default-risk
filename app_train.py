import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import MissingIndicator, SimpleImputer

from sklearn.decomposition import PCA
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
        'PHONE_TO_BIRTH_RATIO': df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH'],
        # Change in income
        'CHANGE_INCOME': df.groupby('SK_ID_CURR')['AMT_INCOME_TOTAL'].diff().fillna(0),
        # Change in credit
        'CHANGE_CREDIT': df.groupby('SK_ID_CURR')['AMT_CREDIT'].diff().fillna(0),
        # Change in annuity
        'CHANGE_ANNUITY': df.groupby('SK_ID_CURR')['AMT_ANNUITY'].diff().fillna(0),
        # Loan Utilization Ratio
        'LOAN_UR': df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'],
        # Age
        'AGE': df['DAYS_BIRTH'].apply(lambda x: -int(x / 365)),
        # Debt Burden Ratio
        'DEBT_BURDEN': df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'],
        # External Source Product:
        'EXT_SOURCE_PROD': df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
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

# # Missing Imputer
# missing_cols = [col for col in df.columns if df[col].isnull().any()]
# new_cols = [col + '_missing' for col in missing_cols]
# print(f'Number of missing columns: {len(missing_cols)}')

# # MissingIndicator
# mi = MissingIndicator()
# mi.fit(df)
# new_df = pd.DataFrame(mi.transform(df), columns=new_cols, 
#                       index=df.index)
# print(f'new_df shape: {new_df.shape}')

# # Concat
# df = pd.concat([df, new_df], axis=1)
# print(f'df shape: {df.shape}')

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

# Astype into category
cat_cols = train.select_dtypes('object').columns
train[cat_cols] = train[cat_cols].astype('category')
test[cat_cols] = test[cat_cols].astype('category')

# Drop columns with only one unique value
print('Drop columns with only one unique value')
cols_to_drop = [col for col in train.columns if train[col].nunique() == 1]
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)
print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')

# WoETransformer
print('WoETransformer')
woe_transformer = WoETransformer(bins=40)
woe_transformer.fit(train, y)

train = woe_transformer.transform(train)
test = woe_transformer.transform(test)
print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')

# Impute missing values
print('Impute missing values')
imputer = SimpleImputer(strategy='most_frequent')
train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns, index=train.index)
test = pd.DataFrame(imputer.transform(test), columns=test.columns, index=test.index)
print("Number of nulls in train: ", np.isnan(train).sum().sum())
print("Number of nulls in test: ", np.isnan(test).sum().sum())

# # Select features
# selected_features = select_features_lightgbm(train, y, threshold=0.001)
# train = train[selected_features.index]
# test = test[selected_features.index]

print("Final train shape: ", train.shape)
print("Final test shape: ", test.shape)

# Save train and test
print('Saving...')
train.to_csv('processed-data/application_train.csv')
test.to_csv('processed-data/application_test.csv')
print('Done!')