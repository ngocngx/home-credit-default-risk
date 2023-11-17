import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from functions import *

    
def gini(y_test, y_pred):
    return 2 * roc_auc_score(y_test, y_pred) - 1


def evaluate(y_test, y_pred):
    print('Gini: {}'.format(gini(y_test, y_pred)))
    print(classification_report(y_test, y_pred))

class ProcessingPipeline:
    def __init__(self) -> None:
        self.df = None
        self.cols_with_missing_values = None
        self.low_variance_cols_cat = None

        self.num_cols = None
        self.cat_cols = None

        self.onehot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')

    def replace_nan(self, df):
        nan_values = ['Unknown', 'XNA', 'not specified']
        df = df.replace(nan_values, np.nan)
        return df
        
    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        self.df = self.replace_nan(df)

        # Identify columns with more than 50% missing values
        self.cols_with_missing_values = df.columns[df.isnull().mean() > 0.5]

        # Identify low variance categorical columns
        object_cols_df = df.select_dtypes(include='object')
        self.low_variance_cols_cat = object_cols_df.columns[object_cols_df.apply(lambda x: x.value_counts(normalize=True).var()) < 0.01]

        # Update numerical and categorical column references
        self.num_cols = df.select_dtypes(include='number').columns
        self.cat_cols = object_cols_df.columns

    def transform(self, df):
        if self.df is None:
            raise RuntimeError("The fit method must be called before transform")
        
        df = self.replace_nan(df)

        # Drop identified columns
        cols_to_drop = set(self.cols_with_missing_values) | set(self.low_variance_cols_cat)
        df.drop(columns=list(cols_to_drop), inplace=True)
        print(f'Dropped {len(cols_to_drop)} columns: {cols_to_drop}')

        # Fill missing values
        for col in df.select_dtypes('object').columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        for col in df.select_dtypes('number').columns:
            df[col].fillna(df[col].mean(), inplace=True)

        # One hot encoding
        self.onehot_encoder.fit(df.select_dtypes('object'))
        cat_df_encoded = pd.DataFrame(self.onehot_encoder.transform(df.select_dtypes('object')).toarray(), 
                                      columns=self.onehot_encoder.get_feature_names_out(), 
                                      index=df.index)
        df = pd.concat([df.select_dtypes(include='number'), cat_df_encoded], axis=1)

        return df

