import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from functions import *

def create_features(df):
    new_features = {
        # % LOADING OF CREDIT LIMIT PER CUSTOMER
        'PERCENTAGE_LOADING_OF_CREDIT_LIMIT': df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL'],
        # RATE OF PAYBACK OF LOANS
        'RATE_OF_PAYBACK': df['AMT_PAYMENT_CURRENT'] / df['AMT_INST_MIN_REGULARITY'],
        # DAY PAST DUE FLAG
        'DPD_FLAG': df['SK_DPD'] > 0,
        # % of MINIMUM PAYMENTS MISSED
        'PERCENTAGE_OF_MINIMUM_PAYMENTS_MISSED': df['AMT_PAYMENT_CURRENT'] / df['AMT_INST_MIN_REGULARITY'],
        #  RATIO OF CASH VS CARD SWIPES
        'RATIO_OF_CASH_VS_CARD_SWIPES': df['CNT_DRAWINGS_ATM_CURRENT'] / df['CNT_DRAWINGS_CURRENT'],
        # Minimum Payments Only
        'MINIMUM_PAYMENTS_ONLY': df['AMT_PAYMENT_CURRENT'] == df['AMT_INST_MIN_REGULARITY'],
        # Utilization Rate
        'UTILIZATION_RATE': df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL'],
        # Increasing Debt Load
        'INCREASING_DEBT_LOAD': df['AMT_BALANCE'] > df['AMT_BALANCE'].shift(1),
        # Changes in Spending Patterns
        'CHANGES_IN_SPENDING_PATTERNS': df['AMT_DRAWINGS_CURRENT'] > df['AMT_DRAWINGS_CURRENT'].shift(1),
        # Overlimit Flag
        'OVERLIMIT_FLAG': df['AMT_BALANCE'] > df['AMT_CREDIT_LIMIT_ACTUAL'],
        # Rapid Account Turnover
        'RAPID_ACCOUNT_TURNOVER': df['CNT_DRAWINGS_CURRENT'] > df['CNT_DRAWINGS_CURRENT'].shift(1),
    }
    
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df

# Load data
cc = pd.read_csv('raw-data/dseb63_credit_card_balance.csv')
cc.set_index('SK_ID_CURR', inplace=True)
print('Initial shape: {}'.format(cc.shape))

# Create features
cc = create_features(cc)
print('After feature creation: {}'.format(cc.shape))

# One-hot encoding for categorical columns with get_dummies
cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

# General aggregations
cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

# Count credit card lines
cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

# Replace inf with nan
cc_agg = cc_agg.replace([np.inf, -np.inf], np.nan)

# Print shape after one-hot encoding
print('After one-hot encoding: {}'.format(cc_agg.shape))
print('Null values: {}'.format(cc_agg.isnull().values.sum()))

# Add missing indicator
missing_cols = [col for col in cc_agg.columns if cc_agg[col].isnull().any()]
new_cols = [col + '_MISSING' for col in missing_cols]
mi = MissingIndicator()
mi.fit(cc_agg[missing_cols])
missing_df = pd.DataFrame(mi.transform(cc_agg[missing_cols]), columns=new_cols, index=cc_agg.index)
cc_agg = pd.concat([cc_agg, missing_df], axis=1)
print('After missing indicator: {}'.format(cc_agg.shape))

# Merge with target
cc_copy = cc_agg.copy()
target = pd.read_csv('processed-data/target.csv')
cc_agg = target.merge(cc_agg, how='left', on='SK_ID_CURR')
cc_agg.set_index('SK_ID_CURR', inplace=True)

# Fill missing values
imputer = SimpleImputer(strategy='mean')
cc_agg = pd.DataFrame(imputer.fit_transform(cc_agg), columns=cc_agg.columns,
                      index=cc_agg.index)
print('Null values: {}'.format(cc_agg.isnull().values.sum()))

# Select features
selected_features = select_features_rf(cc_agg.drop(['TARGET'], axis=1), 
                                       cc_agg['TARGET'], threshold=0.005)
print('Number of selected features: {}'.format(len(selected_features.index.tolist())))
cc = cc_copy[selected_features.index.tolist()]

# Save
cc.to_csv('processed-data/processed_credit_card_balance.csv')
print(cc.head())
