import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from functions.functions import sanitize_columns, select_features_lightgbm

class DataProcessor:
    def __init__(self, train, test, target, *dfs):
        self.train = train
        self.test = test
        self.target = target
        self.dfs = dfs
        
        self.features = None
        self.imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
        self.scaler = StandardScaler().set_output(transform='pandas')

    def process(self, train, test, target):
        # Set index and sort dataframes
        train.set_index('SK_ID_CURR', inplace=True)
        train.sort_index(inplace=True)
        test.set_index('SK_ID_CURR', inplace=True)
        test.sort_index(inplace=True)
        target.set_index('SK_ID_CURR', inplace=True)
        target = target['TARGET']

        # Add is_train column and merge train and test data
        train['is_train'] = 1
        test['is_train'] = 0
        data = pd.concat([train, test], axis=0)

        # Merge additional dataframes
        for df in self.dfs:
            data = data.merge(df, how='left', on='SK_ID_CURR')
            
        # Remove duplicated columns, replace infinite values with NaN, and drop TARGET column
        data = data.loc[:, ~data.columns.duplicated()]
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.drop(['TARGET'], axis=1, inplace=True, errors='ignore')
        data.set_index('SK_ID_CURR', inplace=True)
        print(f'Merged data shape: {data.shape}')

        # Split train and test data
        train = data[data['is_train'] == 1].drop(['is_train'], axis=1)
        test = data[data['is_train'] == 0].drop(['is_train'], axis=1)

        # Sanitize columns and convert data types to float64
        train = sanitize_columns(train)
        test = sanitize_columns(test)
        train = train.astype('float64')
        test = test.astype('float64')

        return train, test, target

    def fit(self, train, target):
        # Process train data and select features
        train = self.process(train, target)
        self.features = select_features_lightgbm(train, target, threshold=0.2)
        print(f'Number of selected features: {len(self.features)}')
        print('Top 10 features:', self.features.sort_values(ascending=False)[:20].index.tolist())

        # Fit imputer and scaler on train data
        train = train[self.features.index]
        self.imputer.fit(train)
        self.scaler.fit(train)

    def transform(self, data):
        # Transform data using imputer and scaler
        data = data[self.features.index]
        data = self.imputer.transform(data)
        data = self.scaler.transform(data)
        return data
    
    def fit_transform(self, train, target):
        # Select features, fit imputer and scaler, and transform train data
        self.features = select_features_lightgbm(train, target, threshold=0.2)
        train = train[self.features.index]
        train = self.imputer.fit_transform(train)
        train = self.scaler.fit_transform(train)
        return train

if __name__ == '__main__':
    # Read data from CSV files
    app_train = pd.read_csv('processed-data/application_train.csv')
    app_test = pd.read_csv('processed-data/application_test.csv')
    target = pd.read_csv('processed-data/target.csv')

    previous_application = pd.read_csv('processed-data/processed_previous_application.csv')
    credit_card_balance = pd.read_csv('processed-data/processed_credit_card_balance.csv')
    installments_payments = pd.read_csv('processed-data/processed_installments.csv')
    bureau = pd.read_csv('processed-data/processed_bureau.csv')
    pos_cash_balance = pd.read_csv('processed-data/processed_pos_cash.csv')

    # Initialize DataProcessor object
    processor = DataProcessor(app_train, app_test, target, previous_application,
                              credit_card_balance, installments_payments, bureau, pos_cash_balance)
    
    # Process train and test data, fit and transform train data, and transform test data
    train, test, target = processor.process(app_train, app_test, target)
    train = processor.fit_transform(train, target)
    test = processor.transform(test)

    # Initialize and train logistic regression model
    model = LogisticRegression(class_weight='balanced', C=0.001, solver='newton-cholesky', max_iter=200)
    
    # Cross validate the model and calculate ROC AUC scores
    print('Cross validating...')
    scores = cross_val_score(model, train, target, cv=5, scoring='roc_auc')
    mean_score = scores.mean()
    gini_score = round(2*mean_score - 1, 5)
    print(f'ROC AUC scores: {scores}')
    print(f'ROC AUC mean: {mean_score}, GINI: {gini_score}')
    
    # Fit the model on train data, predict probabilities for test data, and create submission file
    model.fit(train, target)
    y_pred = model.predict_proba(test)[:, 1]
    submission = pd.DataFrame(index=test.index, data={'TARGET': y_pred})
    submission.sort_index(inplace=True)

    submission.to_csv(f'submissions/submission{gini_score}.csv')