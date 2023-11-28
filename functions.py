import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm

class WoETransformer:
    def __init__(self, smoothing=0.5, default_woe=0.5):
        self.woe_dict = {}
        self.smoothing = smoothing
        self.default_woe = default_woe

    def fit(self, X, y):
        """
        Fit the transformer to the data.

        :param X: DataFrame, feature data (only categorical columns)
        :param y: Series, target variable
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Combine X and y for easier groupby operations
        data = X.copy()
        data['y'] = y

        total_events = y.sum() + self.smoothing
        total_non_events = y.count() - total_events + self.smoothing

        for column in X.columns:
            if data[column].dtype.name == 'category' or isinstance(data[column].iloc[0], str):
                # Group by each category and calculate sums and counts
                grouped = data.groupby(column)['y'].agg(['sum', 'count'])
                grouped['event'] = grouped['sum'] + self.smoothing
                grouped['non_event'] = grouped['count'] - grouped['event'] + self.smoothing

                # Calculate WoE
                grouped['woe'] = np.log((grouped['event'] / total_events) / (grouped['non_event'] / total_non_events))
                self.woe_dict[column] = grouped['woe'].to_dict()

    def transform(self, X):
        """
        Transform the data using the fitted WoE values.

        :param X: DataFrame, feature data to be transformed
        :return: Transformed DataFrame
        """
        # Astype into category
        X = X.astype('category')

        X_transformed = X.copy()
            
        for column in self.woe_dict:
            if column in X_transformed.columns:
                # Handle categorical data
                if X_transformed[column].dtype.name == 'category' or isinstance(X_transformed[column].iloc[0], str):
                    # Add default WoE category if needed
                    if X_transformed[column].dtype.name == 'category':
                        X_transformed[column] = X_transformed[column].cat.add_categories([self.default_woe])

                    X_transformed[column] = X_transformed[column].map(self.woe_dict[column]).fillna(self.default_woe)
            else:
                # If column is not in test data, create it with default WoE value
                X_transformed[column] = self.default_woe

        return X_transformed

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
    categorical_columns = df.select_dtypes(include='object').columns.tolist()
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

def select_feature_var_threshold(X, y, threshold=0.01):
    """
    Select features with variance threshold of value count
    """
    value_counts = X.apply(lambda x: x.value_counts(normalize=True).var(), axis=0)
    return X.columns[value_counts > threshold].tolist()

def feature_importance_rf(X, y):
    """
    Select features using Random Forest.

    Parameters:
    X (DataFrame): The input DataFrame.
    y (Series): The target variable.

    Returns:
    DataFrame: The DataFrame containing feature importances.
    """
    cols = X.columns
    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=cols)
    return importances

def select_features_rf(X, y, threshold=0.001):
    """
    Select features using Random Forest.

    Parameters:
    X (DataFrame): The input DataFrame.
    y (Series): The target variable.
    threshold (float): The threshold for feature selection.

    Returns:
    DataFrame: The DataFrame containing feature importances.
    """
    importances = feature_importance_rf(X, y)
    return importances[importances >= threshold]

def select_features_xgboost(X, y, threshold=0.001):
    """
    Select features using XGBoost.

    Parameters:
    X (DataFrame): The input DataFrame.
    y (Series): The target variable.
    threshold (float): The threshold for feature selection.

    Returns:
    DataFrame: The DataFrame containing feature importances.
    """
    cols = X.columns
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X, y)
    importances = pd.Series(xgb_model.feature_importances_, index=cols)
    return importances[importances >= threshold]

def sanitize_columns(df):
    """
    Sanitize column names.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The DataFrame with sanitized column names.
    """
    json_char = ['{', '}', ':', '"', "'", ',', '[', ']']
    df.columns = [''.join(c for c in str(x) if c not in json_char) for x in df.columns]
    return df

def select_features_lightgbm(X, y, threshold=0.001):
    """
    Select features using LightGBM.

    Parameters:
    X (DataFrame): The input DataFrame.
    y (Series): The target variable.
    threshold (float): The threshold for feature selection.

    Returns:
    DataFrame: The DataFrame containing feature importances.
    """
    cols = X.columns
    lgbm_model = lgbm.LGBMClassifier()
    lgbm_model.fit(X, y)
    importances = pd.Series(lgbm_model.feature_importances_, index=cols)
    print('Max importance: {}'.format(importances.max()))
    importances = importances / 10
    return importances[importances >= threshold]

import pandas as pd
import numpy as np

def calculate_iv(X, y, bins=10, missing=False):
    """
    Calculate the Information Value (IV) of each feature in X relative to the binary target y.

    :param X: DataFrame, feature data
    :param y: Series, binary target variable
    :param bins: Number of bins to use for numerical features
    :param missing: Whether to include missing values as a separate category
    :return: DataFrame with IV values for each feature
    """
    iv_dict = {}
    for column in X.columns:
        if X[column].dtype.kind in 'fi':  # Numeric features
            X[column] = pd.qcut(X[column], q=bins, duplicates='drop').cat.add_categories(['MISSING'])
        elif missing:
            X[column] = X[column].astype('category').cat.add_categories(['MISSING'])

        if missing:
            X[column].fillna('MISSING', inplace=True)

        # Calculate WoE and IV
        grouped = X.groupby(column)[y.name].agg(['sum', 'count'])
        grouped['event'] = grouped['sum']
        grouped['non_event'] = grouped['count'] - grouped['event']
        grouped['event_dist'] = grouped['event'] / grouped['event'].sum()
        grouped['non_event_dist'] = grouped['non_event'] / grouped['non_event'].sum()
        grouped['woe'] = np.log(grouped['event_dist'] / grouped['non_event_dist'])
        grouped['iv'] = (grouped['event_dist'] - grouped['non_event_dist']) * grouped['woe']
        
        iv_value = grouped['iv'].sum()
        iv_dict[column] = iv_value

    iv_df = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['IV'])
    return iv_df

def select_features_by_iv(iv_df, threshold=0.1):
    """
    Select features based on a threshold IV value.

    :param iv_df: DataFrame with IV values for each feature
    :param threshold: IV threshold for feature selection
    :return: List of selected features
    """
    selected_features = iv_df[iv_df['IV'] >= threshold].index.tolist()
    return selected_features

# Usage Example
# iv_df = calculate_iv(X_train, y_train, bins=10, missing=True)
# selected_features = select_features_by_iv(iv_df, threshold=0.1)
# X_train_selected = X_train[selected_features]
# X_test_selected = X_test[selected_features]
