import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from functions import *
from optbinning import BinningProcess

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
        # 'RATIO_OF_CASH_VS_CARD_SWIPES': df['CNT_DRAWINGS_ATM_CURRENT'] / df['CNT_DRAWINGS_CURRENT'],
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
cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)

# General aggregations
cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

# Count credit card lines
cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

# Replace inf with nan
cc_agg = cc_agg.replace([np.inf, -np.inf], np.nan)

# Target
target = pd.read_csv('processed-data/target.csv')
target.set_index('SK_ID_CURR', inplace=True)

cc_train = cc_agg[cc_agg.index.isin(target.index)]
y_train = target[target.index.isin(cc_agg.index)]['TARGET']

cc_test = cc_agg[~cc_agg.index.isin(target.index)]

# Binning process
variable_names = cc_train.columns.tolist()
binning_process = BinningProcess(variable_names)

# Define pipeline
pipeline = Pipeline([
    ('binning_process', binning_process),
    ('imputer', SimpleImputer(strategy='median')),
    ('duplicate_remover', DuplicateColumnRemover()),
    ('feature_selector', FeatureSelectorIV(threshold=0.02)),
])

# Fit and transform train data
cc_train_processed = pipeline.fit_transform(cc_train, y_train)
cc_train_processed.columns = [f'{col}_BINNED' for col in cc_train_processed.columns]
cc_train_processed.index = cc_train.index

# Transform test data
cc_test_processed = pipeline.transform(cc_test)
cc_test_processed.columns = [f'{col}_BINNED' for col in cc_test_processed.columns]
cc_test_processed.index = cc_test.index

# Concatenate original and processed data
cc_train = pd.concat([cc_train, cc_train_processed], axis=1)
cc_test = pd.concat([cc_test, cc_test_processed], axis=1)

# Save
print('Saving...')
cc.to_csv('processed-data/processed_credit_card_balance.csv')
print('Done.')
