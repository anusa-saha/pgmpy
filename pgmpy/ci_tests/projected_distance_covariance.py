import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression

from ._base import _BaseCITest


class ProjectedDistanceCovariance(_BaseCITest):
    r"""
    Projected Distance Covariance (P-DCov) [1] test for conditional independence.

    Regress :math:`X` and :math:`Y` on :math:`Z` (using linear regression) and obtain
    the residuals :math:`r_X` and :math:`r_Y`. Then compute the distance covariance
    between these residuals. The resulting test statistic is:

    .. math::
        T = n \cdot \widehat{V}^2(r_X, r_Y),

    where :math:`\widehat{V}^2` denotes the empirical distance covariance between
    :math:`r_X` and :math:`r_Y`, and :math:`n` is the sample size.

    Under the null hypothesis :math:`X \perp Y \mid Z`, the residuals are independent,
    and the test statistic is expected to be small. Larger values indicate dependence.

    The p-value is computed via permutation testing by randomly permuting the
    residuals of :math:`Y` and recomputing the test statistic.

    This implementation uses linear regression and permutation-based inference.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    num_perm : int, default=100
        Number of permutations for significance testing.

    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    statistic_ : float
        The P-DCov test statistic. Set after calling the test.
    p_value_ : float
        The empirical p-value computed via permutation testing.

    References
    ----------
    .. [1] Fan, J., Feng, Y., and Xia, L. "A Projection-based Conditional Dependence
           Measure with Applications to High-dimensional Undirected Graphical Models".

    """

    _tags = {
        "name": "projected_distance_covariance",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, num_perm: int = 100, random_state: int = None):
        self.data = data
        self.num_perm = num_perm
        self.random_state = random_state
        super().__init__()

    def run_test(self, X: str, Y: str, Z: list):

        data = self.data

        Xv = data[[X]].values.reshape(-1, 1)
        Yv = data[[Y]].values.reshape(-1, 1)

        if Z and len(Z) > 0:
            Zv = data[Z].values

            model_x = LinearRegression().fit(Zv, Xv)
            model_y = LinearRegression().fit(Zv, Yv)

            eps_x = Xv - model_x.predict(Zv)
            eps_y = Yv - model_y.predict(Zv)
        else:
            eps_x = Xv
            eps_y = Yv

        n = eps_x.shape[0]

        a = squareform(pdist(eps_x))
        b = squareform(pdist(eps_y))

        A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()

        V2 = (A * B).sum() / (n * n)
        S2 = a.mean() * b.mean()
        statistic = n * V2 / S2

        rng = np.random.default_rng(self.random_state)
        perm_stats = []

        for _ in range(self.num_perm):
            perm = rng.permutation(n)
            eps_y_perm = eps_y[perm]

            b_perm = squareform(pdist(eps_y_perm))
            Bp = b_perm - b_perm.mean(axis=0) - b_perm.mean(axis=1)[:, None] + b_perm.mean()

            V2_perm = (A * Bp).sum() / (n * n)
            S2_perm = a.mean() * b_perm.mean()
            perm_stats.append(n * V2_perm / S2_perm)

        perm_stats = np.array(perm_stats)
        p_value = np.mean(perm_stats >= statistic)

        self.statistic_ = statistic
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
