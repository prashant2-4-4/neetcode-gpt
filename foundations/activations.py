import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def func(self , x):
        return 1/(1+np.exp(-z[i]))
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        # for i in range(len(z)):
        #     z[i] = round(1/(1+np.exp(-z[i])) , 5)
        # z = np.apply(z , func)
        # z = np.round(z , 5)
        z = 1 / (1 + np.exp(-z))
        z = np.round(z, 5)
        return z

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        z=np.maximum(0 , z)

        return z
