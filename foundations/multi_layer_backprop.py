import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert to numpy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)

        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)

        y_true = np.array(y_true, dtype=float)

        # ====================
        # Forward Pass
        # ====================

        # z1 = xW1^T + b1
        z1 = W1 @ x + b1

        # ReLU
        a1 = np.maximum(0, z1)

        # z2 = a1W2^T + b2
        y_pred = W2 @ a1 + b2

        # MSE Loss
        loss = np.mean((y_pred - y_true) ** 2)

        # ====================
        # Backward Pass
        # ====================

        n = y_true.shape[0]

        # dL/dy_pred
        dz2 = (2 / n) * (y_pred - y_true)

        # dW2 = dz2 outer a1
        dW2 = np.outer(dz2, a1)

        # db2
        db2 = dz2

        # da1 = W2^T * dz2
        da1 = W2.T @ dz2

        # ReLU derivative
        relu_mask = (z1 > 0).astype(float)

        # dz1
        dz1 = da1 * relu_mask

        # dW1 = dz1 outer x
        dW1 = np.outer(dz1, x)

        # db1
        db1 = dz1

        # ====================
        # Return Rounded Output
        # ====================

        return {
            "loss": round(float(loss), 4),
            "dW1": (np.round(dW1, 4) + 0.0).tolist(),
            "db1": (np.round(db1, 4) + 0.0).tolist(),
            "dW2": (np.round(dW2, 4) + 0.0).tolist(),
            "db2": (np.round(db2, 4) + 0.0).tolist(),
        }