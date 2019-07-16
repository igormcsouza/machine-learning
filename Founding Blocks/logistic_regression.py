#*** Logistic Regression ***

# What I'm trying to do now is to figure out wheter the number is prime or not
# But something interesting, it doesn't work!!!

from pandas import read_csv
from sklearn.model_selection import train_test_split

import tensorflow as tf

## PREPROCESSING

# Load the database from file census.csv
data = read_csv('../databases/census.csv')
print(data.head())
print(data['income'].unique())

# We will need to make a transformation on the income column,
# instead of using >50K or <=50K, we will use 1 and 0.
def convert(label):
    if label == ' >50K':
        return 1
    else:
        return 0

data['income'] = data['income'].apply(convert)

print(data.head())
print(data['income'].unique())

X = data.drop('income', axis=1)
Y = data['income']
print(X.head())
print(Y.head())

# data.age.hist()

age = tf.feature_column.numeric_column('age')
cAge = [tf.feature_column.bucketized_column(age, boundaries=[
    20, 30, 40, 50, 60, 70, 80, 90
])]

print(X.columns)

# Spliting the Categorical Columns and creating a variable with all category in all columns
cColumnsNames = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'native-country'
]

cColumns = [tf.feature_column.categorical_column_with_vocabulary_list(
    key=key,
    vocabulary_list=X[key].unique()
) for key in cColumnsNames]

print(cColumns[0])

# Doing the same with the numerical columns

nColumnsNames = [
    'final-weight', 'education-num','capital-gain', 'capital-loos', 'hour-per-week'
]

nColumns = [tf.feature_column.numeric_column(key=key) for key in nColumnsNames]

print(nColumns[0])

# And then, lets wrap all together
columns = cAge + cColumns + nColumns

# We will now creat the train and test data
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3)
print('Split: {0} for testing and {1} for trainning'.format(Xte.shape[0], Xtr.shape[0]))

## LOGISTIC REGRESSION

trainFunc = tf.estimator.inputs.pandas_input_fn(
    x=Xtr, y=Ytr, batch_size=32, num_epochs=None, shuffle=True
)
classifier = tf.estimator.LinearClassifier(feature_columns=columns)
classifier.train(input_fn=trainFunc, steps=10000)