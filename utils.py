import numpy as np

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val + 1e-8)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def one_hot(idx, size):
    arr = np.zeros(size)
    arr[idx] = 1
    return arr