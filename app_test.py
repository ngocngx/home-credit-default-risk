import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from optbinning import BinningProcess
from functions import select_features_iv

# Load data
train = pd.read_csv('raw-data/dseb63_application_train.csv')
train.drop('Unnamed: 0', axis=1, inplace=True)
train.set_index('SK_ID_CURR', inplace=True)

test = pd.read_csv('raw-data/dseb63_application_test.csv')
test.drop('Unnamed: 0', axis=1, inplace=True)
test.set_index('SK_ID_CURR', inplace=True)

# Merge train and test
train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], axis=0)

# Replace positive if DAYS feature with nan
days_cols = [col for col in df.columns if 'DAYS' in col]
for col in days_cols:
    positive_mask = df[col] >= 0
    df.loc[positive_mask, col] = np.nan

df = df.replace(['XNA', 'Unknown', 'not specified'], np.nan)
print(f'df shape: {df.shape}')

# Define feature creation function
def create_features(df):
    # Your existing create_features function code

# Define binning function
def perform_binning(train, test, y):
    cat_cols = train.select_dtypes(include='object').columns.tolist()
    num_cols = train.select_dtypes(exclude='object').columns.tolist()

    variable_names = train.columns.tolist()
    binning_process = BinningProcess(variable_names, categorical_variables=cat_cols, max_n_prebins=30)

    binning_process.fit(train, y)

    train_binned = binning_process.transform(train, metric_missing=0.05)
    train_binned.columns = [f'{col}_BINNED' for col in train_binned.columns]
    train_binned.index = train.index
    test_binned = binning_process.transform(test, metric_missing=0.05)
    test_binned.columns = [f'{col}_BINNED' for col in test_binned.columns]
    test_binned.index = test.index

    train = train.select_dtypes('number')
    train = pd.concat([train, train_binned], axis=1)
    test = test.select_dtypes('number')
    test = pd.concat([test, test_binned], axis=1)
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')

    return train, test

# Define feature selection function
def perform_feature_selection(train, test, y):
    print('Selecting features...')
    selected_features = select_features_iv(train, y, threshold=0.02)
    train = train[selected_features]
    test = test[selected_features]
    print(f'Number of selected features: {len(selected_features)}')
    return train, test

# Create the pipeline
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('create_features', create_features),
    ('binning', perform_binning),
    ('preprocess', preprocessor),
    ('feature_selection', perform_feature_selection)
])

# Split train and test
train = df[df['is_train'] == 1]
test = df[df['is_train'] == 0]
train = train.drop('is_train', axis=1)
test = test.drop('is_train', axis=1)

# Target
y = train['TARGET']
train = train.drop('TARGET', axis=1)
test = test.drop('TARGET', axis=1)

# Replace inf values
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Execute the pipeline
train, test = pipeline.fit_transform(train, test, y)

# Save train and test
print('Saving...')
train.to_csv('processed-data/application_train.csv')
test.to_csv('processed-data/application_test.csv')
print('Done!')
