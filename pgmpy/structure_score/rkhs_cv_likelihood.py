import numpy as np

from sklearn.model_selection import KFold

from pgmpy.structure_score._base import RKHSLikelihood


class CrossValidatedRKHSLikelihood(RKHSLikelihood):
    _tags = {
        "name": "rkhs-cv-ll",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None, kernel="rbf", gamma=None, alpha=1.0, fold=10, random_state=42, max_cache_size=10000):
        super().__init__(data=data, state_names=state_names, kernel=kernel, gamma=gamma, alpha=alpha, max_cache_size=max_cache_size)
        self.fold = fold
        self.random_state = random_state
        if self.fold > len(self.data):
            raise ValueError(
                f"fold={fold} cannot exceed number of samples={len(self.data)}"
            )

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        y = self._np_data[:, self._col_index[variable]].reshape(-1, 1)

        if len(parents) == 0:
            X = None
        else:
            X = self._np_data[:, [self._col_index[p] for p in parents]]

        kf = KFold(n_splits=self.fold, shuffle=True, random_state=self.random_state)

        fold_scores = []

        for train_idx, _ in kf.split(y):
            y_train = y[train_idx]

            K_x = self._kernel_matrix(y_train)

            if X is None:
                K_z = np.zeros_like(K_x)
            else:
                X_train = X[train_idx]
                K_z = self._kernel_matrix(X_train)

            fold_scores.append(self._log_likelihood(K_x, K_z))

        return float(np.mean(fold_scores))