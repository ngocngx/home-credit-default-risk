import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from functions import *

def create_features(df):
    new_features = {
        'CREDIT_DURATION': df['DAYS_CREDIT'] - df['DAYS_CREDIT_ENDDATE'],
        'ENDDATE_DIF': df['DAYS_CREDIT_ENDDATE'] - df['DAYS_ENDDATE_FACT'],
        'DEBT_PERCENTAGE': df['AMT_CREDIT_SUM'] / df['AMT_CREDIT_SUM_DEBT'],
        'DEBT_CREDIT_DIFF': df['AMT_CREDIT_SUM'] - df['AMT_CREDIT_SUM_DEBT'],
        'CREDIT_TO_ANNUITY_RATIO': df['AMT_CREDIT_SUM'] / df['AMT_ANNUITY'],
        'BUREAU_CREDIT_FACT_DIFF': df['DAYS_CREDIT'] - df['DAYS_ENDDATE_FACT'],
        'BUREAU_CREDIT_ENDDATE_DIFF': df['DAYS_CREDIT'] - df['DAYS_CREDIT_ENDDATE'],
        'BUREAU_CREDIT_DEBT_RATIO': df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM'],
        'BUREAU_IS_DPD': df['CREDIT_DAY_OVERDUE'] > 0,
        'BUREAU_IS_DPD_OVER60': df['CREDIT_DAY_OVERDUE'] > 60,
        'BUREAU_IS_DPD_OVER120': df['CREDIT_DAY_OVERDUE'] > 120,
    }

    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df


# Load data
bureau = pd.read_csv('raw-data/dseb63_bureau.csv')
bureau_balance = pd.read_csv('raw-data/dseb63_bureau_balance.csv')
print('Initial shape of bureau: {}'.format(bureau.shape))
print('Initial shape of bureau_balance: {}'.format(bureau_balance.shape))

# Merge bureau and bureau_balance
bureau = bureau.merge(bureau_balance, how='left', on='SK_ID_BUREAU')
print('Shape after merging: {}'.format(bureau.shape))

# Create features
bureau = create_features(bureau)
print('Shape after creating features: {}'.format(bureau.shape))

# One-hot encoding
bureau, cat_cols = one_hot_encoder(bureau, nan_as_category= True)
print('Shape after one-hot encoding: {}'.format(bureau.shape))

# Replace inf with nan
bureau = bureau.replace([np.inf, -np.inf], np.nan)

# Missing indicator
missing_cols = [col for col in bureau.columns if bureau[col].isnull().any()]
new_cols = [col + '_MISSING' for col in missing_cols]
mi = MissingIndicator()
mi.fit(bureau[missing_cols])
missing_df = pd.DataFrame(mi.transform(bureau[missing_cols]), columns=new_cols, index=bureau.index)
bureau = pd.concat([bureau, missing_df], axis=1)
print('Shape after missing indicator: {}'.format(bureau.shape))

# Fill missing values
imputer = SimpleImputer(strategy='median')
bureau = pd.DataFrame(imputer.fit_transform(bureau), columns=bureau.columns,
                      index=bureau.index)
print('Null values: {}'.format(bureau.isnull().values.sum()))

# Agrregate
bureau_agg = bureau.groupby('SK_ID_CURR').agg(['mean', 'max', 'min', 'sum', 'var'])
print('Shape after aggregation: {}'.format(bureau_agg.shape))

# Merge with target
bureau_copy = bureau_agg.copy()
target = pd.read_csv('processed-data/target.csv')
bureau_agg = target.merge(bureau_agg, how='left', on='SK_ID_CURR')
bureau_agg.set_index('SK_ID_CURR', inplace=True)
print('Shape after merging with target: {}'.format(bureau_agg.shape))

# Fill missing values
imputer = SimpleImputer(strategy='median')
bureau_agg = pd.DataFrame(imputer.fit_transform(bureau_agg), columns=bureau_agg.columns,
                      index=bureau_agg.index)
print('Null values: {}'.format(bureau_agg.isnull().values.sum()))

# Select features
selected_features = select_features_rf(bureau_agg.drop(['TARGET'], axis=1), 
                                       bureau_agg['TARGET'], threshold=0.01)
print('Number of selected features: {}'.format(len(selected_features)))
bureau_agg = bureau_copy[selected_features]

# Save
bureau_agg.to_csv('processed-data/processed_bureau.csv')
