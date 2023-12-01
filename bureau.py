import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
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
        'DEBT_TO_INCOME_RATIO': df['AMT_CREDIT_SUM_DEBT'] / df['AMT_INCOME_TOTAL']
    
    }

    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df

def bureau_bb(bureau, bb):

    bureau = create_features(bureau)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category=True)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=True)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size', 'mean']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']

    #Status of Credit Bureau loan during the month
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean', 'min'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'SK_ID_BUREAU': ['count'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'ENDDATE_DIF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
        'DEBT_CREDIT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_IS_DPD': ['mean', 'sum'],
        'BUREAU_IS_DPD_OVER120': ['mean', 'sum'],
        'DEBT_TO_INCOME_RATIO': ['min', 'max', 'mean']
        }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: 
        cat_aggregations[cat] = ['mean']

    for cat in bb_cat: 
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    print('"Bureau/Bureau Balance" final shape:', bureau_agg.shape)
    return bureau_agg

# Load data
bureau = pd.read_csv('raw-data/dseb63_bureau.csv')
bb = pd.read_csv('raw-data/dseb63_bureau_balance.csv')
print('Initial shape of bureau: {}'.format(bureau.shape))
print('Initial shape of bureau_balance: {}'.format(bb.shape))

# Aggregations for bureau_balance
bureau_agg = bureau_bb(bureau, bb)

# Replace all inf values with nan values
bureau_agg.replace([np.inf, -np.inf], np.nan, inplace=True)

# Merge with target
bureau_agg_copy = bureau_agg.copy()
target = pd.read_csv('processed-data/target.csv')
bureau_agg = target.merge(bureau_agg, how='left', on='SK_ID_CURR')
# bureau_agg.set_index('SK_ID_CURR', inplace=True)

# Select features
selected_features = select_features_xgboost(bureau_agg.drop(['SK_ID_CURR', 'TARGET'], axis=1), bureau_agg['TARGET'])
selected_features = selected_features.index.tolist()
bureau_agg = bureau_agg_copy[selected_features]
print('Number of features selected: {}'.format(len(selected_features) - 2))

# Fill missing values
imputer = SimpleImputer(strategy='mean')
bureau_agg = pd.DataFrame(imputer.fit_transform(bureau_agg), columns=bureau_agg.columns, index=bureau_agg.index)

# Save data
bureau_agg.to_csv('processed-data/processed_bureau_2511.csv', index=True)
print('Final shape of bureau: {}'.format(bureau_agg.shape))
