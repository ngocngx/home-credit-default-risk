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

# Groupby SK_ID_CURR
cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
cc_object_groupby = cc.select_dtypes('object').groupby('SK_ID_CURR').agg([mode, 'nunique'])
cc_grouped = cc.select_dtypes(exclude='object').groupby('SK_ID_CURR').agg(['min', 'var'])

# Flatten columns
cc_object_groupby.columns = ['_'.join(col).strip() for col in cc_object_groupby.columns.values]
cc_grouped.columns = ['_'.join(col).strip() for col in cc_grouped.columns.values]

# Merge
cc = pd.concat([cc_object_groupby, cc_grouped], axis=1)

# Merge with target
print('Merge with target')
target = pd.read_csv('raw-data/dseb63_application_train.csv', usecols=['SK_ID_CURR', 'TARGET'])
target.set_index('SK_ID_CURR', inplace=True)

cc = cc.merge(target, left_index=True, right_index=True, how='left')
print('After merge: {}'.format(cc.shape))

# Astype to float
print('Astype to float')
num_cols = cc.select_dtypes(exclude='object').columns
cc[num_cols] = cc[num_cols].astype(float)

# Astype into category
print('Astype into category')
cat_cols = cc.select_dtypes('object').columns
cc[cat_cols] = cc[cat_cols].astype('category')

# Replace inf with max, -inf with min
print('Replace inf with max, -inf with min')
for col in num_cols:
    # Replace positive infinity with the mean of the column excluding infinite values
    cc[col].replace(np.inf, np.nan, inplace=True)  # Convert inf to NaN for mean calculation
    cc[col].replace(np.nan, cc[col].max(), inplace=True)  # Replace NaN with mean value

    # Replace negative infinity with the median of the column excluding infinite values
    cc[col].replace(-np.inf, np.nan, inplace=True)  # Convert -inf to NaN for median calculation
    cc[col].replace(np.nan, cc[col].min(), inplace=True)  # Replace NaN with median value

# Filter cc with not null target
print('Filter cc with not null target')
cc_train = cc[cc['TARGET'].notnull()]
y_train = cc_train['TARGET']
cc_train.drop('TARGET', axis=1, inplace=True)

cc_test = cc[cc['TARGET'].isnull()]
cc_test.drop('TARGET', axis=1, inplace=True)

# WoETransformer
print('WoETransformer')
woe_transformer = WoETransformer(bins=20)
woe_transformer.fit(cc_train, y_train)
cc_train =  woe_transformer.transform(cc_train)
cc_test = woe_transformer.transform(cc_test)

# Concat
cc = pd.concat([cc_train, cc_test], axis=0)
print('After concat: {}'.format(cc.shape))
print('Null values: {}'.format(cc.isnull().values.sum()))

# Astype to float
print('Astype to float')
cc = cc.astype(float)

# Impute missing values
print('Impute missing values')
imputer = SimpleImputer(strategy='most_frequent')
cc = pd.DataFrame(imputer.fit_transform(cc), columns=cc.columns, index=cc.index)

print("Final shape: {}".format(cc.shape))

# Save train and test
print('Saving...')
cc.to_csv('processed-data/processed_credit_card_balance.csv')
print('Done.')
