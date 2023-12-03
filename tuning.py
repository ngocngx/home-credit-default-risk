import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

# Load data
data = pd.read_csv('processed-data/processed_data.csv')
data.set_index('SK_ID_CURR', inplace=True)
print(f'Data shape: {data.shape}')

# Split data
train = data[data['TARGET'].notnull()]
test = data[data['TARGET'].isnull()]
target = train['TARGET']
train.drop('TARGET', axis=1, inplace=True)
test.drop('TARGET', axis=1, inplace=True)

# Tuning

# Split train data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [1e-3, 1, 100],
    'solver' : ['saga', 'lbfgs', 'newton-cholesky'],
    'class_weight': ['balanced', None],
#     'max_iter': [100, 500, 1000, 2000, 5000],
}

# Create the logistic regression model
model = LogisticRegression()
# Create the GridSearchCV object
grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

# Evaluate the model on the validation set
val_predictions = grid_search.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_predictions)
print('Validation AUC:', val_auc)

# You can also access the best model from the grid search
best_model = grid_search.best_estimator_