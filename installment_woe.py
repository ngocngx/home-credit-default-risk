import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import MissingIndicator, SimpleImputer

from sklearn.decomposition import PCA
from functions import *

def create_features(df):
    new_features = {
        'VERSION_CHANGE': df.groupby('SK_ID_PREV')['NUM_INSTALMENT_VERSION'].diff().fillna(0),
        'TIMING_DIFF': df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT'],
        'PAYMENT_RATIO': df['AMT_PAYMENT'] / df['AMT_INSTALMENT'],
        'PAYMENT_DIFF': df['AMT_INSTALMENT'] - df['AMT_PAYMENT'],
        'DUE_FLAG': df['DAYS_ENTRY_PAYMENT'] > df['DAYS_INSTALMENT'],
        'DPD_RATIO': df['DAYS_ENTRY_PAYMENT'] / df['DAYS_INSTALMENT'],
        'DPD_DIFF': df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT'],
        'MOVING_AVG_PAYMENT': df.groupby('SK_ID_PREV')['AMT_PAYMENT'].rolling(3).mean().fillna(0).reset_index(0, drop=True),
        'MOVING_AVG_INSTALMENT': df.groupby('SK_ID_PREV')['AMT_INSTALMENT'].rolling(3).mean().fillna(0).reset_index(0, drop=True),
        'TOTAL_PAID_SO_FAR': df.groupby('SK_ID_PREV')['AMT_PAYMENT'].cumsum().fillna(0),
        'TOTAL_INSTALMENT_SO_FAR': df.groupby('SK_ID_PREV')['AMT_INSTALMENT'].cumsum().fillna(0),
        'PAYMENT_REGULARITY': df.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].diff().fillna(0),
        'DELAYED_PAYMENT_COUNT': df.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].apply(lambda x: x > 0).sum(),
        'VERSION_PAYMENT_INTERACTION': df.groupby('SK_ID_PREV')['NUM_INSTALMENT_VERSION'].apply(lambda x: x > 1).sum(),
        # 'SUM_LAST_180_DAYS': df.groupby('SK_ID_PREV')['AMT_PAYMENT'].rolling(180).sum().fillna(0).reset_index(0, drop=True)
    }


    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df

# Load data
installments = pd.read_csv('raw-data/dseb63_installments_payments.csv')
installments.sort_values(['SK_ID_PREV', 'DAYS_INSTALMENT'], inplace=True)
print('Initial shape: {}'.format(installments.shape))

# Replace positive if DAYS feature with nan
days_cols = [col for col in installments.columns if 'DAYS' in col]
for col in days_cols:
    posive_mask = installments[col] >= 0
    installments.loc[posive_mask, col] = np.nan

# Create features
installments = create_features(installments)
print('After feature creation: {}'.format(installments.shape))

# Groupby SK_ID_CURR
installments = installments.groupby('SK_ID_CURR').mean()
print('After groupby: {}'.format(installments.shape))

# Astype into category
cat_cols = installments.select_dtypes('object').columns
installments[cat_cols] = installments[cat_cols].astype('category')

# Drop SK_ID_PREV
installments.drop('SK_ID_PREV', axis=1, inplace=True)

# Drop columns with only one unique value
print('Drop columns with only one unique value')
cols_to_drop = [col for col in installments.columns if installments[col].nunique() == 1]
installments.drop(cols_to_drop, axis=1, inplace=True)

# Merge with target
print('Merge with target')
target = pd.read_csv('raw-data/dseb63_application_train.csv', usecols=['SK_ID_CURR', 'TARGET'])
target.set_index('SK_ID_CURR', inplace=True)

installments = installments.merge(target, how='left', on='SK_ID_CURR')
print('After merge: {}'.format(installments.shape))

# Filter installments with not null target
print('Filter installments with not null target')
installments_target = installments[installments['TARGET'].notnull()]
y = installments_target['TARGET']
installments_target.drop('TARGET', axis=1, inplace=True)

installments_non_target = installments[installments['TARGET'].isnull()]
installments_non_target.drop('TARGET', axis=1, inplace=True)

print('installments_target: {}'.format(installments_target.shape))

# WoETransformer
print('WoETransformer')
woe_transformer = WoETransformer(bins=40)
woe_transformer.fit(installments_target, y)
installments_target = woe_transformer.transform(installments_target)
installments_non_target = woe_transformer.transform(installments_non_target)

# Concat
print('Concat')
installments = pd.concat([installments_target, installments_non_target], axis=0)
print('After concat: {}'.format(installments.shape))
print('Null values: {}'.format(installments.isnull().values.sum()))

# Impute missing values
print('Impute missing values')
imputer = SimpleImputer(strategy='most_frequent')
installments = pd.DataFrame(imputer.fit_transform(installments), columns=installments.columns, index=installments.index)

print("Final shape: {}".format(installments.shape))

# Save train and test
print('Saving...')
installments.to_csv('processed-data/processed_installments.csv')
print('Done.')
