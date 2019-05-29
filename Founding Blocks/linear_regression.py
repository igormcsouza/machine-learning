# Linear and Logistics regression aplication

#*** Linear Regression ***

#Applying Regression to solve Boston Housing Prediction Problem
#https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef


from sklearn.datasets import load_boston # Import DataSet from sklearn
from pandas import DataFrame, read_csv
data = read_csv('/home/souza/Documents/machine-learning/machine-learning/databases/boston-housing-prices.csv')
print(data.head())

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