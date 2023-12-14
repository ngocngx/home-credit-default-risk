import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from optbinning import BinningProcess
from functions import *
class InfinityToNanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, bureau, y = None):
        return self

    def transform(self, bureau):
        return bureau.replace([np.inf, -np.inf], np.nan)
def build_preprocessor(numeric_features, categorical_featutres):
    numeric_transformer = Pipeline(
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
# Define binning function
def perform_binning(train, test, y):
    variable_names = train.columns.tolist()
    binning_process = BinningProcess(variable_names)
    binning_process.fit(train, y)

    train_binned = binning_process.transform(train)
    train_binned.columns = [train_binned.columns[i] + '_BINNED' for i in range(len(train_binned.columns))]
    train_binned.index = train.index
    test_binned = binning_process.transform(test)
    test_binned.columns = [test_binned.columns[i] + '_BINNED' for i in range(len(test_binned.columns))]
    test_binned.index = test.index

    # Merge original with binned
    train_binned = pd.concat([train, train_binned], axis=1)
    test_binned = pd.concat([test, test_binned], axis=1)

    return train_binned, test_binned

# Define feature selection function
def perform_feature_selection(train, y):
    print('Selecting features...')
    selected_features = select_features_iv(train, y, threshold=0.02)
    print(f'Number of selected features: {len(selected_features)}')
    return train[selected_features]
# Load data
bureau = pd.read_csv('raw-data/dseb63_bureau.csv')
bureau_balance = pd.read_csv('raw-data/dseb63_bureau_balance.csv')
# Aggregations for bureau_balance
bureau_balance = pd.get_dummies(bureau_balance, columns=['STATUS'], dummy_na=True)

bb_aggregations = bureau_balance.groupby('SK_ID_BUREAU').agg({
    'MONTHS_BALANCE': ['min', 'max', 'size', 'mean'],
    'STATUS_0': ['mean'],
    'STATUS_1': ['mean'],
    'STATUS_2': ['mean'],
    'STATUS_3': ['mean'],
    'STATUS_4': ['mean'],
    'STATUS_5': ['mean'],
    'STATUS_C': ['mean', 'count'],
    'STATUS_X': ['mean', 'count'],
    'STATUS_nan': ['mean', 'count'],
})

# Rename columns
bb_aggregations.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_aggregations.columns.tolist()])

# Create features for bureau
bureau = create_feature(bureau)

# Merge bureau_balance with bureau
bureau = bureau.merge(bb_aggregations, how='left', on='SK_ID_BUREAU')
bureau.drop('SK_ID_BUREAU', axis=1, inplace=True)
bureau.set_index('SK_ID_CURR', inplace= True)
# Aggregate
bureau_agg = bureau.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'var'])
bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
bureau_agg['BUR_COUNT'] = bureau.groupby('SK_ID_CURR').size()
print('After aggregation: {}'.format(bureau_agg.shape))
# Target
target = pd.read_csv('processed-data/target.csv')
target.set_index('SK_ID_CURR', inplace=True)
y_train = target[target.index.isin(bureau_agg.index)]['TARGET']

bureau_train = bureau_agg[bureau_agg.index.isin(target.index)]
bureau_test = bureau_agg[~bureau_agg.index.isin(target.index)]

# Drop columns with 1 unique value
cols_to_drop = [col for col in bureau_train.columns if bureau_train[col].nunique() <= 1]
bureau_train.drop(cols_to_drop, axis=1, inplace=True)
bureau_test.drop(cols_to_drop, axis=1, inplace=True)
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
