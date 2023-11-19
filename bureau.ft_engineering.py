#!/usr/bin/env python
# coding: utf-8

# In[208]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest
from ydata_profiling import ProfileReport


# In[209]:


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    IQR = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * IQR
    low_limit = quartile1 - 1.5 * IQR
    
    return low_limit, up_limit


# In[210]:


def check_outlier(dataframe, col_name):
    
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        
        return True
    
    else:
        
        return False


# In[211]:


def grab_col_names(dataframe, cat_th=10, car_th=20):
    
    """
    Returns the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Numerical-looking categorical variables are also included in the categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe for which variable names are to be obtained.
        cat_th: int, optional
                Class threshold value for numerical but categorical variables.
        car_th: int, optional
                Class threshold value for categorical but cardinal variables.

    Returns
    ------
        cat_cols: list
                List of categorical variables.
        num_cols: list
                List of numerical variables.
        cat_but_car: list
                List of categorical but cardinal variables.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is within cat_cols.
        The sum of the 3 lists returned is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    
    return cat_cols, num_cols, cat_but_car


# In[212]:


def remove_outlier(dataframe, col_name):
    
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    
    return df_without_outliers


# In[234]:


def missing_values_table(dataframe, na_name=False):
    
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio']).to_markdown()
    
    print(missing_df, end="\n")
    
    if na_name:
        
        return na_columns


# In[260]:


def label_encoder(dataframe, binary_col):
    
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    
    return dataframe


# In[235]:


bureau = pd.read_csv('/Users/quyngodseb63/Downloads/dseb63_bureau.csv', index_col = 'SK_ID_CURR')
bureau_balance = pd.read_csv('/Users/quyngodseb63/Downloads/dseb63_bureau_balance.csv', index_col = 'SK_ID_BUREAU')


# In[236]:


bureau


# In[237]:


bureau_balance.head()


# In[238]:


print(bureau.shape)
print(bureau_balance.shape)


# In[239]:


bureau.describe()


# In[240]:


bureau_balance.describe()


# In[241]:


bureau.dtypes


# In[242]:


bureau_balance.dtypes


# In[243]:


bureau.duplicated().sum()


# In[244]:


bureau_balance.duplicated().sum()


# In[245]:


bureau_balance.drop_duplicates()


# In[246]:


check_outlier(bureau, "AMT_CREDIT_SUM")


# In[247]:


cat_cols, num_cols, cat_but_car = grab_col_names(bureau)


# In[248]:


for col in num_cols:
    
    new_bureau = remove_outlier(bureau, col)


# In[249]:


bureau.shape[0] - new_bureau.shape[0]


# In[250]:


bureau.shape


# In[252]:


bureau.isnull().values.any()


# In[253]:


bureau.isnull().sum()


# In[254]:


bureau.notnull().sum()


# In[255]:


# Observations with at least one missing value;
bureau[bureau.isnull().any(axis=1)].head()


# In[256]:


missing_values_table(bureau)


# In[257]:


bureau.dropna()


# In[258]:


le = LabelEncoder()
le.fit_transform(bureau["CREDIT_ACTIVE"])[0:5]


# In[272]:


le.fit_transform(bureau["CREDIT_TYPE"])


# In[268]:


#le.inverse_transform([0, 2])


# In[270]:


binary_cols = [col for col in bureau.columns if bureau[col].dtype not in [int, float]]
binary_cols


# In[271]:


bureau[binary_cols].head()


# In[265]:


for col in binary_cols:
    label_encoder(bureau, col)
bureau


# In[ ]:




