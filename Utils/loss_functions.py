import numpy as np
import math

def mse(actual, predicted):
   np.sqrt(np.sum(np.square(np.difference(actual, predicted))))

def mae(actual, predicted):
    np.sum(np.abs(np.difference(actual, predicted)))

def cross_entropy(actual, predicted):
    return -(actual*math.log(predicted) + (1-actual)*math.log(1-predicted))
