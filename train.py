from datetime import date
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Load data
train = pd.read_csv('processed-data/application_train.csv')
test = pd.read_csv('processed-data/application_test.csv')
target = pd.read_csv('processed-data/target.csv')

train.set_index('SK_ID_CURR', inplace=True)
train.sort_index(inplace=True)
test.set_index('SK_ID_CURR', inplace=True)
test.sort_index(inplace=True)
target.set_index('SK_ID_CURR', inplace=True)
target = target['TARGET']

print(f'Train shape: {train.shape}, Test shape: {test.shape}, Target shape: {target.shape}')

# Merge train and test
train['is_train'] = 1
test['is_train'] = 0
data = pd.concat([train, test], axis=0)

# # Merge with previous application
# previous_application = pd.read_csv('processed-data/processed_previous.csv')
# data = data.merge(previous_application, how='left', on='SK_ID_CURR')

# Merge with credit card balance
credit_card_balance = pd.read_csv('processed-data/processed_credit_card_balance.csv')
data = data.merge(credit_card_balance, how='left', on='SK_ID_CURR')

# # Merge with installments payments
# installments_payments = pd.read_csv('processed-data/processed_installments.csv')
# data = data.merge(installments_payments, how='left', on='SK_ID_CURR')

# Merge with bureau
bureau = pd.read_csv('processed-data/processed_bureau.csv')
data = data.merge(bureau, how='left', on='SK_ID_CURR')

# Set index
data.set_index('SK_ID_CURR', inplace=True)

# Split train and test
train = data[data['is_train'] == 1].drop(['is_train'], axis=1)
test = data[data['is_train'] == 0].drop(['is_train'], axis=1)
print(f'Train shape: {train.shape}, Test shape: {test.shape}')

# Fill missing values
imputer = SimpleImputer(strategy='median')
train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns, index=train.index)
test = pd.DataFrame(imputer.transform(test), columns=test.columns, index=test.index)

# Scale numerical features
robust_scaler = RobustScaler(quantile_range=(1, 99))
train = pd.DataFrame(robust_scaler.fit_transform(train), columns=train.columns, index=train.index)
test = pd.DataFrame(robust_scaler.transform(test), columns=test.columns, index=test.index)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
train = pd.DataFrame(minmax_scaler.fit_transform(train), columns=train.columns, index=train.index)
test = pd.DataFrame(minmax_scaler.transform(test), columns=test.columns, index=test.index)

# Train
log_reg = LogisticRegression(class_weight='balanced', solver='newton-cholesky',
                             max_iter=100)

# Cross validate
scores = cross_val_score(log_reg, train, target, cv=5, scoring='roc_auc')
print(f'ROC AUC scores: {scores}')
print(f'ROC AUC mean: {scores.mean()}, GINI: {2*scores.mean() - 1}')

# # Fit
# log_reg.fit(train, target)

# # Predict
# y_pred = log_reg.predict_proba(test)[:, 1]
# submission = pd.DataFrame(index=test.index, data={'TARGET': y_pred})
# submission.sort_index(inplace=True)
# submission

# # Save submission with date
# today = date.today()
# submission.to_csv(f'submissions/submission-{today}.csv')
