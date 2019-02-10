import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from collections import defaultdict


def encoding_multiple_features(data):
    print('Encoding features using sklearn...')
    from sklearn import preprocessing
    from collections import defaultdict

    # dictionnary the hold all the labelencoding
    dict_encoding = defaultdict(preprocessing.LabelEncoder)
    mask_object = data.dtypes == 'object'

    # Encoding the variable
    data.loc[:, mask_object] = data.loc[:, mask_object].apply(
        lambda x: dict_encoding[x.name].fit_transform(x))

    # Using the dictionary to label future data
    # data.apply(lambda x: dict_encoding[x.name].transform(x))
    return data, dict_encoding


def inverse_encoding_multiple_features(data_enc, dict_encoding):
    print('Inverse Encoding features using sklearn...')
    # Inverse the encoded
    data_recov = data_enc.apply(
        lambda x: dict_encoding[x.name].inverse_transform(x))
    return data_recov


def split_data(X, y, test_size=0.25):

    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=test_size, random_state=0)
    return train_X, val_X, train_y, val_y


def evaluate_model(y_actual, y_predicted):

    MAE = mean_absolute_error(y_actual, y_predicted)

    return MAE


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    pred_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, pred_val)
    return(mae)


def model_regression(dataset, modelType, param=None):

    # Define the model
    if modelType == 'DecisionTree':
        model = DecisionTreeRegressor(
            random_state=0, max_leaf_nodes=param)
    elif modelType == 'RandomForest':
        model = RandomForestRegressor(random_state=0, max_leaf_nodes=param)
    else:
        ValueError(
            'modelType should be "DecicionTree" or "RandomForest"')

    # %% Fit the model to the data (Training)
    X_train, X_val, y_train, y_val = dataset
    model.fit(X_train, y_train)

    # %% Prediction of the validation dataset
    y_pred = model.predict(X_val)

    # %% Evaluate the model
    eval_param = evaluate_model(y_val, y_pred)

    return y_pred, eval_param
