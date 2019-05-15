'''Applying Regression to solve Boston Housing Prediction Problem
https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef
'''

from sklearn.datasets import load_boston # Import DataSet from sklearn
from pandas import DataFrame as df
boston = load_boston()

# Transforming the set into DF, adding Target Column.
data = df(boston.data, columns = boston.feature_names)
data['MED'] = boston.target
print(data.head())

X = data.drop('MED', axis=1)
Y = data['MED']

from sklearn.cross_validation import train_test_split

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size = 0.33, random_state = 1)
print(Xtr.size())
