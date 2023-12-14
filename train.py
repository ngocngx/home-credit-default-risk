import pandas as pd
import numpy as np
from functions.functions import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from optbinning import BinningProcess

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

# Merge with previous application
previous_application = pd.read_csv('processed-data/processed_previous_application.csv')
print(f'Previous application shape: {previous_application.shape}')
data = data.merge(previous_application, how='left', on='SK_ID_CURR')

# Merge with credit card balance
credit_card_balance = pd.read_csv('processed-data/processed_credit_card_balance.csv')
print(f'Credit card balance shape: {credit_card_balance.shape}')
data = data.merge(credit_card_balance, how='left', on='SK_ID_CURR')

# Merge with installments payments
installments_payments = pd.read_csv('processed-data/processed_installments.csv')
print(f'Installments payments shape: {installments_payments.shape}')
data = data.merge(installments_payments, how='left', on='SK_ID_CURR')

# Merge with bureau
bureau = pd.read_csv('processed-data/processed_bureau.csv')
print(f'Bureau shape: {bureau.shape}')
data = data.merge(bureau, how='left', on='SK_ID_CURR')

# Merge with pos cash balance
pos_cash_balance = pd.read_csv('processed-data/processed_pos_cash.csv')
print(f'POS cash balance shape: {pos_cash_balance.shape}')
data = data.merge(pos_cash_balance, how='left', on='SK_ID_CURR')

# # Merge new features
# new_data = pd.read_csv('processed-data/all_data.csv')
# data = data.merge(new_data, how='left', on='SK_ID_CURR')

# Print shape after merge
print(f'Merged data shape: {data.shape}')

# Drop duplicate columns
data = data.loc[:, ~data.columns.duplicated()]

# Replace inf with nan
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop target
data.drop(['TARGET'], axis=1, inplace=True, errors='ignore')

# Set index
data.set_index('SK_ID_CURR', inplace=True)

# Split train and test
train = data[data['is_train'] == 1].drop(['is_train'], axis=1)
test = data[data['is_train'] == 0].drop(['is_train'], axis=1)
print(f'Train shape: {train.shape}, Test shape: {test.shape}')

# Sanitize column
train = sanitize_columns(train)
test = sanitize_columns(test)

# Astype into float
train = train.astype('float64')
test = test.astype('float64')

# Feature selection
selected_features = select_features_lightgbm(train, target, threshold=0.2)
print(f'Number of selected features: {len(selected_features)}')
print('Top 10 features:', selected_features.sort_values(ascending=False)[:20].index.tolist())
train = train[selected_features.index]
test = test[selected_features.index]

# Fill missing values
print('Filling missing values...')
imputer = SimpleImputer(strategy='mean')
train_imputed = imputer.fit_transform(train)
test_imputed = imputer.transform(test)

# # Select using SelectKBest
# print('Selecting features...')
# selector = SelectKBest(f_classif, k=400)
# selector.fit(train_imputed, target)
# train_imputed = selector.transform(train_imputed)
# test_imputed = selector.transform(test_imputed)

# Standardize
print('Standardizing...')
standard_scaler = StandardScaler()
train_scaled = standard_scaler.fit_transform(train_imputed)
test_scaled = standard_scaler.transform(test_imputed)

# Convert to dataframe
train = pd.DataFrame(index=train.index, data=train_scaled, columns=selected_features.index)
test = pd.DataFrame(index=test.index, data=test_scaled, columns=selected_features.index)

# Concat and save
# train_merged = pd.concat([train, target], axis=1)
# data_merged = pd.concat([train, test], axis=0)
# data_merged.to_csv('processed-data/processed_data.csv')

# Train
model = LogisticRegression(class_weight='balanced', C=0.001, solver='newton-cholesky', max_iter=200)

# Cross validate
print('Cross validating...')
scores = cross_val_score(model, train, target, cv=5, scoring='roc_auc')
mean_score = scores.mean()
gini_score = round(2*mean_score - 1, 5)
print(f'ROC AUC scores: {scores}')
print(f'ROC AUC mean: {mean_score}, GINI: {gini_score}')

# Fit
model.fit(train, target)
print('Top 20 features:', train.columns[np.argsort(model.coef_[0])[-20:]].tolist())

# Predict
y_pred = model.predict_proba(test)[:, 1]
submission = pd.DataFrame(index=test.index, data={'TARGET': y_pred})
submission.sort_index(inplace=True)

# Save submission
submission.to_csv(f'submissions/submission-{gini_score}.csv')