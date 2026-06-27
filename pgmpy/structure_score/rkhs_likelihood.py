import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

from pgmpy.structure_score._base import BaseStructureScore


class RKHSLikelihood(BaseStructureScore):
    _tags = {
        "name": "ll-rkhs",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": False,
    }
    def __init__(self, data, state_names=None, kernel="rbf", gamma=None, alpha=1.0, max_cache_size=10000):
        super().__init__(data=data, state_names=state_names, max_cache_size=max_cache_size)
        self._np_data = self.data.to_numpy()
        self._col_index = {col: i for i, col in enumerate(self.data.columns)}
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha

    def _kernel_matrix(self, X):
        """
        Computes the centered kernel matrix.
        """
        K = pairwise_kernels(X, metric=self.kernel, gamma=self.gamma)

        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n

        return H @ K @ H

    def _rkhs_regression(self, K_x, K_z):
        """
        Computes the RKHS regression

            F_hat = K_x (K_z + nλI)^(-1) K_z

        from Appendix A1.
        """
        n = K_x.shape[0]

        ridge_operator = np.linalg.solve(K_z + n * self.alpha * np.eye(n), K_z)
        fitted_kernel = K_x @ ridge_operator
        kernel_residuals = K_x - fitted_kernel

        return fitted_kernel, kernel_residuals

    def _residual_covariance(self, K_x, K_z):
        """
        Computes residual covariance matrix from Appendix A1.

        Sigma = RR^T / n
        """
        _, kernel_residuals = self._rkhs_regression(K_x, K_z)
        n = kernel_residuals.shape[0]

        residual_covariance = kernel_residuals @ kernel_residuals.T / n

        return residual_covariance

    def _log_likelihood(self, K_x, K_z):
        """
        Computes Equation (6).
        """

        residual_covariance = self._residual_covariance(K_x, K_z)

        _, logdet = np.linalg.slogdet(residual_covariance)
        n = residual_covariance.shape[0]

        ll = -(n * n / 2.0) * np.log(2 * np.pi) - (n / 2.0) * logdet - (n / 2.0)

        return ll

    def _local_score(self, variable, parents):
        y = self._np_data[:, self._col_index[variable]].reshape(-1, 1)
        K_x = self._kernel_matrix(y)
        if len(parents) == 0:
            K_z = np.zeros_like(K_x)
        else:
            X = self._np_data[:, [self._col_index[p] for p in parents]]
            K_z = self._kernel_matrix(X)
        return self._log_likelihood(K_x, K_z)
