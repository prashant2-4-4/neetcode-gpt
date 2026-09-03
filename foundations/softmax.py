import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        #numerical stabliity because e1000 is infinity so subtracting from max value gives stabliity
        z = z - max(z)
        exp_sum = np.sum(np.exp(z))
        z = np.exp(z)/exp_sum
        return np.round(z , 4)
