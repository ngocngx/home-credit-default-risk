import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
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

# Create features
installments = create_features(installments)

# One-hot encoding
installments, cat_cols = one_hot_encoder(installments, nan_as_category= True)
print('After one-hot encoding: {}'.format(installments.shape))

# Replace positive if DAYS feature with nan
days_cols = [col for col in installments.columns if 'DAYS' in col]
for col in days_cols:
    posive_mask = installments[col] >= 0
    installments.loc[posive_mask, col] = np.nan

# Replace XNA, Unknown, not specified with nan
installments = installments.replace(['XNA', 'Unknown', 'not specified'], np.nan)

# Replace inf with nan
installments = installments.replace([np.inf, -np.inf], np.nan)
print('Null values: {}'.format(installments.isnull().values.sum()))

# Missing indicator
missing_cols = [col for col in installments.columns if installments[col].isnull().any()]
new_cols = [col + '_MISSING' for col in missing_cols]
mi = MissingIndicator()
mi.fit(installments[missing_cols])
missing_df = pd.DataFrame(mi.transform(installments[missing_cols]), columns=new_cols, index=installments.index)
installments = pd.concat([installments, missing_df], axis=1)

# Fill missing values
imputer = SimpleImputer(strategy='median')
installments = pd.DataFrame(imputer.fit_transform(installments), columns=installments.columns,
                      index=installments.index)
print('Null values: {}'.format(installments.isnull().values.sum()))

# Agrregate
installments.drop(['SK_ID_PREV'], axis=1, inplace=True) 
installments_agg = installments.groupby('SK_ID_CURR').mean()

# Merge with target
installments_copy = installments_agg.copy()
target = pd.read_csv('processed-data/target.csv')
installments_agg = target.merge(installments_agg, how='left', on='SK_ID_CURR')
installments_agg.set_index('SK_ID_CURR', inplace=True)
print('Null values: {}'.format(installments_agg.isnull().values.sum()))

# Fill missing values
imputer = SimpleImputer(strategy='mean')
installments_agg = pd.DataFrame(imputer.fit_transform(installments_agg), columns=installments_agg.columns,
                      index=installments_agg.index)
print('Null values: {}'.format(installments_agg.isnull().values.sum()))

# Select features
selected_features = select_features_rf(installments_agg.drop(['TARGET'], axis=1), installments_agg['TARGET'])
print('Selected features: {}'.format(selected_features.index.tolist()))
installments_agg = installments_copy[selected_features.index.tolist()]

# Save
installments_agg.to_csv('processed-data/processed_installments.csv')