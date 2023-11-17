import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

def drop_missing(df, threshold):
    cols_to_drop = []
    for col in df.columns:
        if df[col].isnull().sum() / len(df) > threshold:
            cols_to_drop.append(col)
    print('Columns to drop: {}'.format(cols_to_drop))
    df = df.drop(cols_to_drop, axis=1)
    return df

def drop_low_variance_cat(df, threshold):
    low_var_cat = []
    for col in df.select_dtypes(include='object').columns:
        if df[col].value_counts(normalize=True).var() < threshold:
            low_var_cat.append(col)
    print('Low variance categorical columns: {}'.format(low_var_cat))
    df = df.drop(low_var_cat, axis=1)
    return df

def drop_low_variance_num(df, threshold):
    low_var_num = []
    for col in df.select_dtypes(include='number').columns:
        if df[col].var() < threshold:
            low_var_num.append(col)
    print('Low variance numerical columns: {}'.format(low_var_num))
    df = df.drop(low_var_num, axis=1)
    return df

def drop_correlated_col(df, threshold):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    correlated_col = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(correlated_col, axis=1)
    print('Correlated columns: {}'.format(correlated_col))
    return df

def gini(y_test, y_pred):
    return 2 * roc_auc_score(y_test, y_pred) - 1

def evaluate(y_test, y_pred):
    print('Gini: {}'.format(gini(y_test, y_pred)))
    print(classification_report(y_test, y_pred))


def mode(x):
    return x.value_counts().index[0]

def aggregate_prev(df):
    id_curr = df[['SK_ID_PREV', 'SK_ID_CURR']]
    id_curr_agg = id_curr.groupby('SK_ID_PREV').first()

    df = df.drop(columns=['SK_ID_CURR'])
    
    num_df = df[df.select_dtypes(include='number').columns.tolist()]
    cat_df = df[df.select_dtypes(include='object').columns.tolist() + ['SK_ID_PREV']]


    num_df = num_df.groupby('SK_ID_PREV').agg(['min', 'max', 'mean', 'sum'])
    cat_df = cat_df.groupby('SK_ID_PREV').agg([mode, 'nunique', 'count'])

    # flatten name
    num_df.columns = ['_'.join(col).strip() for col in num_df.columns.values]
    cat_df.columns = ['_'.join(col).strip() for col in cat_df.columns.values]
    
    # concat
    df = pd.concat([num_df, cat_df], axis=1)
    df = pd.concat([id_curr_agg, df], axis=1)

    return df

def aggregate_curr(df):
    id_curr = df[['SK_ID_CURR']]
    id_curr_agg = id_curr.groupby('SK_ID_CURR').first()
    
    num_df = df[df.select_dtypes(include='number').columns.tolist()]
    cat_df = df[df.select_dtypes(include='object').columns.tolist() + ['SK_ID_CURR']]


    num_df = num_df.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum'])
    cat_df = cat_df.groupby('SK_ID_CURR').agg([mode, 'nunique', 'count'])

    # flatten name
    num_df.columns = ['_'.join(col).strip() for col in num_df.columns.values]
    cat_df.columns = ['_'.join(col).strip() for col in cat_df.columns.values]
    
    # concat
    df = pd.concat([num_df, cat_df], axis=1)
    df = pd.concat([id_curr_agg, df], axis=1)

    return df
