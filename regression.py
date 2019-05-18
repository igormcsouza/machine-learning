# Linear and Logistics regression aplication

#*** Linear Regression ***

#Applying Regression to solve Boston Housing Prediction Problem
#https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef


from sklearn.datasets import load_boston # Import DataSet from sklearn
from pandas import DataFrame as df
boston = load_boston()

# Transforming the set into DF, adding Target Column.
data = df(boston.data, columns = boston.feature_names)
data['MED'] = boston.target
#print(data.head())

X = data.drop('MED', axis=1)
Y = data['MED']
print("Database Loaded")

from sklearn.model_selection import train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size = 0.33, random_state = 1)

from sklearn.metrics import mean_squared_error as mse

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(Xtr, Ytr)
print("\nModel Trained Using Sklearn Linear Regression")
print('MSE: ', mse(Yte, model.predict(Xte)))

from sklearn.linear_model import SGDRegressor as gradient_descent
print("\nModel Trained Using Sklearn Gradient Descent")
model = gradient_descent()
model.fit(Xtr, Ytr)
print('MSE: ', mse(Yte, model.predict(Xte)))


#import matplotlib.pyplot as plt
#from matplotlib import rcParams

#Ypr = mse(Yte, model.predict(Xte))
#plt.scatter(Yte, Ypr)
#plt.xlabel("Prices")
#plt.ylabel("Predicted prices")
# plt.plot(Xte, Ypr, color='r') # Need to figure out how does it works!
#plt.show()

print("\n-------------------------------------------------------------------------\n")

#*** Logistic Regression ***

# What I'm trying to do now is to figure out wheter the number is prime or not

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2

from numpy import reshape
from others import isPrime, isEven

Xt = [i for i in range(1, 1000)]
Yt = [[0, 1] if isPrime(i) else [1, 0] for i in Xt]
x_train = reshape(Xt,(-1,1))
y_train = reshape(Yt, (-1, 2))
x_train, x_test, y_train, y_test = train_test_split(x_train, 
                                                    y_train, 
                                                    test_size = 0.33, 
                                                    random_state = 1)
#print("{0}, {1}, {2}, {3}".format(len(x_train), len(y_train), len(x_test), len(y_test)))

model = Sequential()
model.add(Dense(2,  # output dim is 2, one score per each class
                activation='softmax',
                kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                input_dim=1))  # input dimension = number of features your data has
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

def Prediction(n):
    p = model.predict(x=[[n]])
    return [0 if i < 0.5 else 1 for i in p[0]]

for n in range(1, 30):
    print("The number {0} is {1} prime".format(n, Prediction(n)))