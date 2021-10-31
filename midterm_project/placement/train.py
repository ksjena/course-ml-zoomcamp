#!/usr/bin/env python
# coding: utf-8

# ### Engineering Placements Prediction
# ### Predict Final Year Engineering College Placements
# 
# A University Announced Its On-Campus Placement Records For The Engineering Course. The Data Is From The Years 2013 And 2014.
# The Following Is The College Placements Data Compiled Over 2 years. **Use This Data To Predict And Analyse Whether A Student Gets Placed**, Based On His/Her Background.
# Perform Extensive EDAs And Bring Out Insights.
# Build classification model using various ML techniques
# ##### Kaggle URL: https://www.kaggle.com/tejashvi14/engineering-placements-prediction?select=collegePlace.csv
# ##### Dataset: kaggle datasets download -d tejashvi14/engineering-placements-prediction

# importing basic packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle

# Reading data
df = pd.read_csv('data.csv')

# ## Data preparation
# Make column names to lower and replace space with underscore if any
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Create a list of categorical variables
categorical_variables = list(df.dtypes[df.dtypes == 'object'].index)

# Make values in a categorical variables to lower and repalce space with underscore if any
for col in categorical_variables:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# ## Setting up the validation framework
#splitting train, val, test set
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.placedornot.values
y_val = df_val.placedornot.values
y_test = df_test.placedornot.values

del df_train['placedornot']
del df_val['placedornot']
del df_test['placedornot']

# ## One-hot encoding
# Use Scikit-Learn to encode categorical features
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

# Fitting the model
rf = RandomForestClassifier(n_estimators=200,
                            max_depth=10,
                            min_samples_leaf=3,
                            random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)

# saving random forest model in pickle format
with open('model1.bin', 'wb') as f_out:
    pickle.dump((dv, rf), f_out)