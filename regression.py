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

def isPrime(n): 
    # Corner case 
    if n <= 1: 
        return False
    # Check from 2 to n-1 
    for i in range(2, n): 
        if (n % i == 0): 
            return False
    return True

from sklearn.linear_model import LogisticRegression
from numpy import reshape

X = [i for i in range(1, 1000)]
Y = [isPrime(i) for i in X]
Y = [1 if i else 0 for i in Y]
X = reshape(X,(-1,1))
Y = reshape(Y, (-1,))
model = LogisticRegression(solver='liblinear').fit(X, Y)
for n in X:
    p = reshape(n, (-1,1))
    print("The number {0} is {1}".format(n, model.predict(p)))

print('MSE: ', mse(Y, model.predict(X)))