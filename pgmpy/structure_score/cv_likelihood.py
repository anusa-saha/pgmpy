import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

from pgmpy.structure_score._base import BaseStructureScore


class CVLikelihood(BaseStructureScore):
    r"""
    Cross-validated likelihood score for continuous Bayesian networks.

    This score follows the cross-validated likelihood framework
    of Huang et al. (2018). For each local family, a Kernel Ridge
    Regression model is fitted on training folds and evaluated on
    held-out folds using the Gaussian log-likelihood of residuals.

    The local score is:

        S_CV(X_i, Pa_i) = (1/Q) * Σ_q ℓ(F̂_i^(q) | D_test^(q))

    where F̂_i^(q) is learned on the q-th training fold and
    ℓ(·) is the Gaussian log-likelihood evaluated on the
    corresponding test fold.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.structure_score import CVLikelihood
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": rng.normal(size=100),
    ...         "B": rng.normal(size=100),
    ...         "C": rng.normal(size=100),
    ...     }
    ... )
    >>> score = CVLikelihood(data)
    >>> print(score.local_score("B", ()))
    -13.310
    >>> print(score.local_score("B", ("A",)))
    -13.770
    >>> print(score.local_score("B", ("C",)))
    -14.420
    >>> print(score.local_score("B", ("A", "C")))
    -17.212

    >>> A = rng.normal(size=500)
    >>> B = np.sin(A) + 0.1 * rng.normal(size=500)

    >>> data = pd.DataFrame({"A": A, "B": B})
    >>> score = CVLikelihood(data)

    >>> print(score.local_score("B", ()))
    -50.834
    >>> print(score.local_score("B", ("A",)))
    46.235

    >>> A = rng.normal(size=500)
    >>> C = rng.normal(size=500)
    >>> B = np.sin(A) + 0.1 * rng.normal(size=500)

    >>> data = pd.DataFrame({"A": A, "B": B, "C": C})
    >>> score = CVLikelihood(data)
    >>> print(score.local_score("B", ()))
    >>> print(score.local_score("B", ("C",)))
    >>> print(score.local_score("B", ("A",)))
    >>> print(score.local_score("B", ("A", "C")))

    """
    _tags = {
        "name": "cv-ll",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(
        self,
        data,
        kernel="rbf",
        state_names=None,
        fold=10,
        alpha=1e-3,
        gamma=None,
        random_state=42,
        max_cache_size=10000,
    ):
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)
        self._np_data = self.data.to_numpy()
        self._col_index = {col: i for i, col in enumerate(self.data.columns)}
        self.kernel = kernel
        self.fold = fold      
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state
        if self.fold > len(data):
            raise ValueError(f"fold={fold} cannot exceed number of samples={len(data)}") 
        
    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        y = self._np_data[:, self._col_index[variable]]
        kf = KFold(n_splits=self.fold, shuffle=True, random_state=self.random_state)
        fold_ll = []
        for train_idx, test_idx in kf.split(y):
            y_train = y[train_idx]
            y_test = y[test_idx]
            if len(parents) == 0:
                pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
            else:
                parent_cols = [self._col_index[p] for p in parents]
                X = self._np_data[:, parent_cols]
                X_train = X[train_idx]
                X_test = X[test_idx]
                model = KernelRidge(
                    kernel=self.kernel,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

            resid = y_test - pred
            n_test = len(y_test)
            rss = float(resid @ resid)
            ll = (
                -0.5 * n_test * (np.log(2.0 * np.pi) + 1.0)
                - 0.5 * n_test * np.log(rss / n_test)
            )
            fold_ll.append(ll)

        return float(np.mean(fold_ll))