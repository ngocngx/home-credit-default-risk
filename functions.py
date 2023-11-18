import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2

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

# run functions and pre_settings
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def group(df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()

def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)

def do_sum(dataframe, group_cols, counted, agg_name):
    gp = dataframe[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(columns={counted: agg_name})
    dataframe = dataframe.merge(gp, on=group_cols, how='left')
    return dataframe

def reduce_mem_usage(dataframe):
    m_start = dataframe.memory_usage().sum() / 1024 ** 2
    for col in dataframe.columns:
        col_type = dataframe[col].dtype
        if col_type != object:
            c_min = dataframe[col].min()
            c_max = dataframe[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dataframe[col] = dataframe[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dataframe[col] = dataframe[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dataframe[col] = dataframe[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dataframe[col] = dataframe[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dataframe[col] = dataframe[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dataframe[col] = dataframe[col].astype(np.float32)
                else:
                    dataframe[col] = dataframe[col].astype(np.float64)

    m_end = dataframe.memory_usage().sum() / 1024 ** 2
    return dataframe

nan_as_category = True


def risk_groupanizer(dataframe, column_names, target_val=1, upper_limit_ratio=8.2, lower_limit_ratio=8.2):
    # one-hot encoder killer :-)
    all_cols = dataframe.columns
    for col in column_names:

        temp_df = dataframe.groupby([col] + ['TARGET'])[['SK_ID_CURR']].count().reset_index()
        temp_df['ratio%'] = round(temp_df['SK_ID_CURR']*100/temp_df.groupby([col])['SK_ID_CURR'].transform('sum'), 1)
        col_groups_high_risk = temp_df[(temp_df['TARGET'] == target_val) &
                                       (temp_df['ratio%'] >= upper_limit_ratio)][col].tolist()
        col_groups_low_risk = temp_df[(temp_df['TARGET'] == target_val) &
                                      (lower_limit_ratio >= temp_df['ratio%'])][col].tolist()
        if upper_limit_ratio != lower_limit_ratio:
            col_groups_medium_risk = temp_df[(temp_df['TARGET'] == target_val) &
                (upper_limit_ratio > temp_df['ratio%']) & (temp_df['ratio%'] > lower_limit_ratio)][col].tolist()

            for risk, col_groups in zip(['_high_risk', '_medium_risk', '_low_risk'],
                                        [col_groups_high_risk, col_groups_medium_risk, col_groups_low_risk]):
                dataframe[col + risk] = [1 if val in col_groups else 0 for val in dataframe[col].values]
        else:
            for risk, col_groups in zip(['_high_risk', '_low_risk'], [col_groups_high_risk, col_groups_low_risk]):
                dataframe[col + risk] = [1 if val in col_groups else 0 for val in dataframe[col].values]
        if dataframe[col].dtype == 'O' or dataframe[col].dtype == 'object':
            dataframe.drop(col, axis=1, inplace=True)
    return dataframe, list(set(dataframe.columns).difference(set(all_cols)))

def select_feature(X, y, k=10):
    importances = mutual_info_classif(X, y)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
    return X.columns[indices[:k]].tolist()

def select_feature_var_threshold(X, y, threshold=0.01):
    """
    Select features with variance threshold of value count
    """
    value_counts = X.apply(lambda x: x.value_counts(normalize=True).var(), axis=0)
    return X.columns[value_counts > threshold].tolist()
