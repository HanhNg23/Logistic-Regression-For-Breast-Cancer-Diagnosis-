# %%
from myFunctionGD import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

#Loading Dataset#
data = pd.read_csv('bdiag.csv')
#summary(data)
data.info()
#get statistical details(data)
print("Data describe --->")
data.describe()

# %%
#Check Missing Data#
print("Sum of cell missing data in each features --->")
data.isnull().sum()

# %%
#Data Analysis#
print("Data uniques in diagnosis column --->")
data["diagnosis"].unique()
print("Count the number of each uniques data in diagnosis column --->")
data["diagnosis"].value_counts()
# %%
#Exploring Data Analysis to see the features' relationships#
#chart for distribution of target variables 
fig = plt.figure(figsize=(10, 5)) 
fig.add_subplot(1, 2, 1) #the position of char chart
charchart = data["diagnosis"].value_counts(normalize=True).plot.pie()
fig.add_subplot(1, 2, 2) #the position of churn chart
churnchart = sns.countplot(x = data["diagnosis"]) 
plt.tight_layout()
plt.show()

#Correlation between features
corr_plot = sns.heatmap(data.corr())
plt.title("Correlation Plot")
plt.show()

# %%
#Pair Plot
sns.pairplot(data, hue = "diagnosis")
plt.title("Pair Plot")
plt.show()

# %%
#Data Cleaning#
#Get dummy variables of label data
diagnosisarr = pd.get_dummies(data["diagnosis"], drop_first = True)
#diagnosisarr.head(10)

#Delete unnecessary data
data.drop(["id", "diagnosis"], axis = 1, inplace = True)
#data.head(10)

#Concatenate Features and Label
data = pd.concat([data, diagnosisarr], axis = 1)
#data.head(10)
data = np.array(data)

# train-test-split 
data = np.hstack((np.ones((data.shape[0], 1)), data)) 
split_factor = 0.8
split = int(split_factor * data.shape[0]) 
  
X_train = data[:split, :-1] 
y_train = data[:split, -1].reshape((-1, 1)) 

X_test = data[split:, :-1] 
y_test = data[split:, -1].reshape((-1, 1)) 
  
print("Number of examples in training set = % d"%(X_train.shape[0])) 
print("Number of examples in testing set = % d"%(X_test.shape[0]))

#Let compute theta using minibatch gradient descent
theta, error_list = gradientDescent(X_train, y_train, 0.01, 64, 30) 
print("Bias = ", theta[0]) 
print("Coefficients = ", theta[1:]) 
print("Errorlist: ", error_list[-1])

#Train accuracy
p = predict(theta, X_train)
print("Train accuracy on X_train, y_train ", np.mean(np.double(p == y_train)) * 100)  
p = predict(theta, X_test)
print("Train accuracy on X_test, y_test ", np.mean(np.double(p == y_test)) * 100)

# t = sigmoid(hypothesis(X_train[7], theta))
# print("The probability of t: ", t , " the real answer: ", y_train[4])


# visualising gradient descent 
plt.figure(figsize=(15,5))
plt.plot(error_list) 
plt.xlabel("Number of iterations") 
plt.ylabel("Cost") 
plt.show() 