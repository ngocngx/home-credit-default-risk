import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from functions.functions import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from optbinning import OptimalBinning
class InfinityToNanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, installments, y = None):
        return self

    def transform(self, installments):
        return installments.replace([np.inf, -np.inf], np.nan), installments.replace(['XNA', 'Unknown', 'not specified'], np.nan)
def build_preprocessor(numeric_features, categorical_featutres):
    numeric_transformer = make_pipeline(
         SimpleImputer(strategy='mean'),
         StandardScaler(),
)
    categoric_transformer = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore')
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categoric_transformer, categoric_features)
        ])
    return preprocessor
def main():
    # Load data
    installments = pd.read_csv('raw-data/dseb63_installments_payments.csv')
    installments.sort_values(['SK_ID_PREV', 'DAYS_INSTALMENT'], inplace=True)
    print('Initial shape: {}'.format(installments.shape))


    # Define the pipeline

    numeric_features = installments_train.select_dtypes(include=['int64', 'float64']).columns
    categoric_features = installments_train.select_dtypes(include=['object']).columns
    preprocessor = build_preprocessor(numeric_features, categoric_features)
    installments_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    installments_pipeline.fit(installments_train, installments_test)
    joblib.dump(installments_pipeline, 'installments_pipeline.pkl')
    joblib.dump(preprocessor, 'preprocessor_ins.pkl')

