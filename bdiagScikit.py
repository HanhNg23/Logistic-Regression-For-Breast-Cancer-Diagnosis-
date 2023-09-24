# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

#Loading Dataset#
data = pd.read_csv('bdiag.csv')
diagnosisarr = pd.get_dummies(data["diagnosis"], drop_first = True)
data.drop(["id", "diagnosis"], axis = 1, inplace = True)
data = pd.concat([data, diagnosisarr], axis = 1)
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

#Compute theta using LogisticRegression form Sklearn
from sklearn.linear_model import LogisticRegression
loRegr = LogisticRegression()
loRegr.fit(X_train, y_train)
# score = loRegr.score(X_test, y_test)

#Coefficient of theta
print("Coefficients " ,loRegr.coef_)
print("\n")

#Train accuracy
from sklearn.metrics import accuracy_score
predictionsTest = loRegr.predict(X_test)
predictionsTrain = loRegr.predict(X_train)
print("Train accuracy on X_test, y_test ", accuracy_score(y_train, predictionsTrain) * 100)
print("Train accuracy on X_train, y_train ", accuracy_score(y_test, predictionsTest) * 100)  

