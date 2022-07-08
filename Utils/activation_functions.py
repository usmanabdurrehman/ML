import numpy as np

def relu(x):
    if(x<0): 
        return 0
    else:
        return x 

def relu6(x):
    if(x<0):
        return 0
    elif(x>6):
        return 6
    else:
        return x

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return x
