import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from pgmpy.ci_tests._base import _BaseCITest


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import ProjectedDistanceCovariance
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> Z = rng.normal(size=n)
    >>> X = Z + rng.normal(size=n)
    >>> Y = Z + rng.normal(size=n)
    >>> data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    >>> test = ProjectedDistanceCovariance(data=data, num_perm=100, random_state=0)
    >>> stat, pval = test.run_test("X", "Y", ["Z"])
    >>> round(stat,3)
    np.float64(1.487)
    >>> round(pval,2)
    np.float(0.18)


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

        Z_aug = list(Z) + ["intercept"]
        data_aug = data.assign(intercept=np.ones(data.shape[0]))

        Xv = data_aug[X].values
        Yv = data_aug[Y].values
        Zv = data_aug[Z_aug].values

        X_coef = np.linalg.lstsq(Zv, Xv, rcond=None)[0]
        Y_coef = np.linalg.lstsq(Zv, Yv, rcond=None)[0]

        residual_x = (Xv - Zv @ X_coef).reshape(-1, 1)
        residual_y = (Yv - Zv @ Y_coef).reshape(-1, 1)

        n = residual_x.shape[0]

        a = squareform(pdist(residual_x))
        b = squareform(pdist(residual_y))

        S1 = (a * b).sum() / (n * n)
        S2 = (a.sum() / (n * n)) * (b.sum() / (n * n))
        S3 = (a.sum(axis=1) * b.sum(axis=1)).sum() / (n**3)
        V2 = S1 + S2 - 2.0 * S3
        statistic = n * V2 / S2

        rng = np.random.default_rng(self.random_state)
        perm_stats = self._permutation_stats(residual_y, a, n, rng)

        p_value = np.mean(perm_stats >= statistic)

        self.statistic_ = statistic
        self.p_value_ = p_value

        return self.statistic_, self.p_value_

    def _permutation_stats(
        self,
        residual_y: np.ndarray,
        a: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Compute the P-DCov statistic for each permutation of the Y residuals.

        The X distance matrix ``a`` is precomputed once in :meth:`run_test` and
        passed in here unchanged, so only ``b`` needs recomputing per permutation.

        Parameters
        ----------
        residual_y : ndarray of shape (n, 1)
            Residuals for Y.
        a : ndarray of shape (n, n)
            Pairwise distance matrix for X residuals.
        n : int
            Sample size.
        rng : numpy.random.Generator
            The master RNG, advanced in-place across permutations.

        Returns
        -------
        perm_stats : ndarray of shape (num_perm,)
            Test statistic value for each permuted sample.
        """
        perm_stats = np.empty(self.num_perm)

        for i in range(self.num_perm):
            perm = rng.permutation(n)
            b_perm = squareform(pdist(residual_y[perm]))

            S1_p = (a * b_perm).sum() / (n * n)
            S2_p = (a.sum() / (n * n)) * (b_perm.sum() / (n * n))
            S3_p = (a.sum(axis=1) * b_perm.sum(axis=1)).sum() / (n**3)

            V2_p = S1_p + S2_p - 2.0 * S3_p
            perm_stats[i] = n * V2_p / S2_p

        return perm_stats
