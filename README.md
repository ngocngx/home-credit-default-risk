# Data Preparation & Visualization Project for Final - Group 6

This repository contains the code and resources for a data preparation project focused on predicting credit default risk for Home Credit customers. The goal is to build a robust model that can assess the likelihood of a customer defaulting on a loan based on various features.

## Repo Structure
    ├── README.md               <- Overview and guidance.
    ├── requirements.txt        <- List of Python packages needed for the project.
    ├── eda                     <- Notebooks for initial data exploration.
    ├── processed-data          <- Final data sets for model training.
    ├── raw-data                <- Original, unmodified data.
    ├── feature-processing      <- Scripts to turn raw data into features for modeling.
    │   ├── app_train.py
    │   ├── bureau.py
    │   ├── credit_card_balance.py
    │   ├── installment_payment.py
    │   ├── pos_cash.py
    │   └── previous.py
    └── train.py                <- Scripts for merging data and training.

*Note: Due to storage limit, raw-data and processed-data are ignored for commit. Here is the directory to [data folder](https://drive.google.com/drive/folders/1LnO8eeQOL7NZtanArT3QUq97TccZOFeM?usp=drive_link).*

## Assigned Work
- **Nguyen Vu Anh Ngoc (Group Leader):**
    - Modeling
    - EDA: main application table, credit_card_balance, installment_payment
    - Feature Engineering: all tables
    - Feature Selection

- **Ngo Khanh Linh**:
    - EDA: POS_CASH_balance, previous_application, bureau, bureau_balance
    - Feature Engineering: POS_CASH_balance, previous_application

- **Ngo Duc Quy**:
    - EDA: bureau, bureau_balance
    - Feature Engineering: bureau, bureau_balance