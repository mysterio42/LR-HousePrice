import glob
import math
import os
import random
import string

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """

    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def generate_model_name(size=5):
    """
    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def train_model(features, labels):
    """
    training of the Linear Regression model
    :param features: sqft_living features
    :param labels: house price labels
    :return: trained model weight
    """
    model = LinearRegression()
    model.fit(features, labels)

    mse = mean_squared_error(features, labels)
    print(f'MSE {math.sqrt(mse)}')
    print(f'R^2 value: {model.score(features, labels)}')
    print(f'b_0: {model.coef_[0][0]} \nb_1: {model.intercept_[0]}')

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        model_name = WEIGHTS_DIR + 'LinReg-' + generate_model_name(5) + '.pkl'
        with open(model_name, 'wb') as f:
            joblib.dump(value=model, filename=f, compress=3)
            print(f'Model saved at {model_name}')


def load_model(path):
    """

    :param path: weight path
    :return: load model based on the path
    """
    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def predict_model(model, x: int):
    """

    :param model: LinearRegression model
    :param x: sqft_living
    :return: predicted house price
    """
    return model.predict([[x]])[0][0]
