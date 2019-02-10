#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import utils
import seaborn as sns
'''
This project tackles the housing price prediction
Melbourne dataset
'''

# %%
print('\nAnalysis of data')

print('\n# Importing the data...')
data = pd.read_csv('./melb_data.csv')
print('The shape of the data is {}'.format(data.shape))
print(data.columns)
print(data.dtypes)
print(data.describe())
print('\n# Number of category for each feature \n {}'.format(data.nunique()))
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')


print('\n# Clean the data: drop out all columns that contains NAN')
data_clean = data.dropna(axis=0)

# print('\n# Encode the labels in the features')
# data_enc, dict_encoding = encoding_multiple_features(data_clean)

print('\n# Split the data set into traing, validation and testing sets')

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = data_clean[melbourne_features]
y = data_clean.Price

dataset = utils.split_data(X, y, test_size=.25)
X_train, X_val, y_train, y_val = dataset
# %%
print('\nBuild a ML model: Specify and fit the model')
# Define the model

modelType = 'RandomForest'
param = None
y_pred, mae = utils.model_regression(dataset, modelType, param)
print('\n# The mean average error (before tuning param)= {0}'.format(mae))

# %% compare MAE with differing values of max_leaf_nodes
print('\n# Tune the model by selecting the optimal max_leaf_nodes...')
candidate_max_leaf_nodes = [100, 250, 500, 700, 1000, 10000]
scores = {leaf_size: utils.model_regression(dataset, modelType, param=leaf_size)[1]
          for leaf_size in candidate_max_leaf_nodes}
print(scores)
max_leaf_nodes = min(scores, key=scores.get)
print('\n# The optimal parameter for max_leaf_nodes is {0}'.format(
    max_leaf_nodes))

print('\nRedo the prediction with the optimal parameter:')
y_pred, mae = utils.model_regression(dataset, modelType, param=max_leaf_nodes)
print(y_val.tolist()[0:6])
print(y_pred[0:6])
print('\n# The mean average error (after tuning param)= {0}'.format(mae))


#%%


#%%
