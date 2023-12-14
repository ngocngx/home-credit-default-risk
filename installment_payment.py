import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from functions.functions import *
from optbinning import OptimalBinning, Scorecard, BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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
        # Volatility
        'PAYMENT_VOLATILITY': df.groupby('SK_ID_PREV')['AMT_PAYMENT'].std().fillna(0),
        'INSTALMENT_VOLATILITY': df.groupby('SK_ID_PREV')['AMT_INSTALMENT'].std().fillna(0),
        # Time-based features
        'INSTALLMENT_COUNT': df.groupby('SK_ID_PREV').size(),
        'INSTALLMENT_FIRST_DUE_DIFF': df.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].first() - df.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].first(),
        'INSTALLMENT_LAST_DUE_DIFF': df.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].last() - df.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].last(),
        'DUE_DIFF_MEAN': df.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].mean() - df.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].mean(),
        'DUE_DIFF_MAX': df.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].max() - df.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].max(),
        'DUE_DIFF_MIN': df.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].min() - df.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].min(),
        'DUE_DIFF_VAR': df.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].var() - df.groupby('SK_ID_PREV')['DAYS_INSTALMENT'].var(),
        
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
installments, cat_cols = one_hot_encoder(installments, nan_as_category=True)
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

# Agrregate
installments.drop(['SK_ID_PREV'], axis=1, inplace=True) 
installments_agg = installments.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
installments_agg.columns = pd.Index(['INSTALL_' + e[0] + "_" + e[1].upper() for e in installments_agg.columns.tolist()])

# Count installments
installments_agg['INSTALL_COUNT'] = installments.groupby('SK_ID_CURR').size()

# Target 
target = pd.read_csv('processed-data/target.csv')
target.set_index('SK_ID_CURR', inplace=True)

installments_train = installments_agg[installments_agg.index.isin(target.index)]
y_train = target[target.index.isin(installments_agg.index)]['TARGET']

installments_test = installments_agg[~installments_agg.index.isin(target.index)]

# Binning process
variable_names = installments_train.columns.tolist()
binning_process = BinningProcess(variable_names)
binning_process.fit(installments_train, y_train)

# Transform train and test
installments_train_binned = binning_process.transform(installments_train, metric_missing=0.05)
installments_train_binned.index = installments_train.index
installments_test_binned = binning_process.transform(installments_test, metric_missing=0.05)
installments_test_binned.index = installments_test.index

# Select features IV
print('Selecting features...')
selected_features = select_features_iv(installments_train_binned, y_train, threshold=0.02)
print(f'Number of selected features: {len(selected_features)}')
installments_train = installments_train[selected_features]
installments_test = installments_test[selected_features]

# Concat train and test
installments_agg = pd.concat([installments_train, installments_test], axis=0)
installments_agg.index = installments_agg.index.astype('int64')

# Save
installments_agg.to_csv('processed-data/processed_installments.csv')