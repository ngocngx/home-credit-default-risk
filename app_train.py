import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import MissingIndicator, SimpleImputer

from sklearn.decomposition import PCA
from functions import *
from optbinning import OptimalBinning, Scorecard, BinningProcess


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

# Replace inf values
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Binning
cat_cols = train.select_dtypes(include='object').columns.tolist()
num_cols = train.select_dtypes(exclude='object').columns.tolist()

variable_names = train.columns.tolist()
binning_process = BinningProcess(variable_names, categorical_variables=cat_cols, 
                                 max_n_prebins=30)

binning_process.fit(train, y)

# Transform train and test
train_binned = binning_process.transform(train, metric_missing=0.05)
train_binned.index = train.index
test_binned = binning_process.transform(test, metric_missing=0.05)
test_binned.index = test.index

# Select features
selected_features = select_features_lightgbm(train_binned, y, threshold=0.001)
train = train_binned[selected_features.index]
test = test_binned[selected_features.index]
print(f'Number of selected features: {len(selected_features)}')
print(f'Top 10 selected features: {selected_features.index[:10].tolist()}')

print("Final train shape: ", train.shape)
print("Final test shape: ", test.shape)

# Save train and test
print('Saving...')
train.to_csv('processed-data/application_train.csv')
test.to_csv('processed-data/application_test.csv')
print('Done!')