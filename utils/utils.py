# random things that might be useful
import numpy as np

def oneHotEncode(value, size):
    arr = np.zeros(size)
    arr[value] = 1
    return arr