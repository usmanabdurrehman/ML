import numpy as np


def minkowski(P1, P2, p):
    return np.sum(np.abs(P1-P2)**p)**(1/p)


def euclidean(P1, P2):
    return minkowski(P1, P2, 2)


def manhattan(P1, P2):
    return minkowski(P1, P2, 1)
