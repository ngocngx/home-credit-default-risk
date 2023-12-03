import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# optuna
import optuna

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
def objective(trial):
    # Set parameters
    params = {
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'C': trial.suggest_loguniform('C', 1e-5, 1e+5),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'max_iter': trial.suggest_int('max_iter', 100, 5000),
        'random_state': trial.suggest_int('random_state', 1, 9999),
        'n_jobs': -1
    }

    # Create model
    model = LogisticRegression(**params)

    # Cross validation
    score = cross_val_score(model, train, target, scoring='roc_auc', cv=5, n_jobs=-1).mean()

    return score

# Create study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print result
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print('Best score:', study.best_value)
print('Best parameters:', study.best_params)