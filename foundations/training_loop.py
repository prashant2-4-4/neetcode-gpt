import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        w = np.zeros(X.shape[1])
        b = 0
        print(w , b)
        n = len(y)
        # y = y.reshape(-1 , 1)
        for epoch in range(epochs):
            y_hat = X @ w + b
            # print(y_hat)
            mse = (1/n) * np.sum((y_hat - y)**2)
            # print(mse , X.T.shape , y_hat.shape , y.shape)
            dl = 2/n * (np.dot(X.T  , (y_hat - y)))
            # print(dl)
            w = w - lr * dl
            # print(w)
            db = 2/n * np.sum(y_hat - y)

            b = b - lr * db
            # print(b)
            # break
        return (np.round(w , 5) , np.round(b , 5))
