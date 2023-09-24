
import numpy as np
#functions
def hypothesis(X, theta): 
    return np.dot(X, theta) 

#sigmoid function
def sigmoid(z): 
    p = np.zeros((z.shape))
    p = 1. / (1 + np.exp(-z))
    return p
  
# function to compute gradient of error function w.r.t. theta 
def gradient(X, y, theta): 
    h = hypothesis(X, theta)
    h = sigmoid(h)
    grad = np.dot(X.transpose(), (h - y)) 
    return grad 
  
# function to compute the error for current values of theta 
def cost(X, y, theta): 
    h = hypothesis(X, theta) 
    h = sigmoid(h)
    J = np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(), np.log(1 - h)) 
    J = (-1 * J) / X.shape[0]
    return J[0] 
  
# function to create a list containing mini-batches 
def create_mini_batches(X, y, batch_size): 
    mini_batches = [] 
    data = np.hstack((X, y)) # X join with y
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches

def predict(theta, X):
    m = X.shape[0] #num of rows 
    p = np.zeros((m,1))
    p = sigmoid(hypothesis(X, theta))
    p = np.where(p >= 0.5, 1, p)
    p = np.where(p < 0.5, 0, p)
    return p


def gradientDescent(X, y, learning_rate, batch_size, max_iters): 
    theta = np.zeros((X.shape[1], 1)) 
    error_list = [] 
    for itr in range(max_iters): 
        mini_batches = create_mini_batches(X, y, batch_size) 
        for mini_batch in mini_batches: 
            X_mini, y_mini = mini_batch 
            theta = theta - (learning_rate / X_mini.shape[0]) * gradient(X_mini, y_mini, theta) 
            error_list.append(cost(X_mini, y_mini, theta)) 
    return theta, error_list 
