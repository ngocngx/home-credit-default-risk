import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from optbinning import BinningProcess
from functions.functions import *

class InfinityToNanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X.replace([np.inf, -np.inf], np.nan)
def build_preprocessor(numerical_cols, categorical_cols):
    numerical_transformer = Pipeline(
        InfinityToNanTransformer(),
        SimpleImputer(strategy='mean'),
        RobustScaler()
    )
    categorical_transformer = Pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor 





def main():
    train = pd.read_csv('processed-data/application_train.csv')
    test = pd.read_csv('processed-data/application_test.csv')
    target = pd.read_csv('processed-data/target.csv')

    train.set_index('SK_ID_CURR', inplace=True)
    train.sort_index(inplace=True)
    test.set_index('SK_ID_CURR', inplace=True)
    test.sort_index(inplace=True)
    target.set_index('SK_ID_CURR', inplace=True)
    target = target['TARGET']
    # Create the pipeline
    numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = train.select_dtypes(include=['object']).columns

    preprocessor = build_preprocessor(num_cols, cat_cols)

    pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', LogisticRegressions())
])


# Execute the pipeline
train = pipeline.fit_transform(train, y)

# Save train and test
print('Saving...')
train.to_csv('processed-data/application_train.csv')
test.to_csv('processed-data/application_test.csv')
print('Done!')

