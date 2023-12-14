import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
class InfinityToNanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, bureau, y = None):
        return self

    def transform(self, bureau):
        return bureau.replace([np.inf, -np.inf], np.nan)
def build_preprocessor(numeric_features, categorical_featutres):
    numeric_transformer = make_pipeline(
         SimpleImputer(strategy='mean'),
         StandardScaler(),
])
    categoric_transformer = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore')
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categoric_transformer, categorical_features)
        ])
    return preprocessor

# Load data
def main():
    bureau = pd.read_csv('raw-data/dseb63_bureau.csv')
    bureau_balance = pd.read_csv('raw-data/dseb63_bureau_balance.csv')

    # Create the pipeline
    numeric_features = bureau_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = bureau_train.select_dtypes(include=['object']).columns

    preprocessor = build_preprocessor(numeric_features, categorical_featutres)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])


    pipeline.fit(bureau_train, bureau.test)

    joblib.dump(pipeline, 'pipeline.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
