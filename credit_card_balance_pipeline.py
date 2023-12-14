import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
class InfinityToNanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, cc, y = None):
        return self

    def transform(self, cc):
        return cc.replace([np.inf, -np.inf], np.nan)
def build_preprocessor(numeric_features, categoric_featutres):
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
    cc = pd.read_csv('raw-data/dseb63_credit_card_balance.csv')
    cc.set_index('SK_ID_CURR', inplace=True)
    print('Initial shape: {}'.format(cc.shape))


    # Create the pipeline
    numeric_features = cc_train.select_dtypes(include=['int64', 'float64']).columns
    categoric_features = cc_train.select_dtypes(include=['object']).columns
    preprocessor = build_preprocessor(numeric_features, categoric_features)
    # Define pipeline
    credit_card = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    credit_card.fit(cc_train, cc_test)

    joblib.dump(credit_card, 'credit_card.pkl')
    joblib.dump(preprocessor, 'preprocessor_cc.pkl')

        


