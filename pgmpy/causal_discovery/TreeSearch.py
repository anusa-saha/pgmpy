#!/usr/bin/env python

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery


class TreeSearch(_BaseCausalDiscovery):
    """
    Search class for learning tree related graph structure. The algorithms
    supported are Chow-Liu and Tree-augmented naive bayes (TAN).

    Chow-Liu constructs the maximum-weight spanning tree with mutual information
    score as edge weights.

    TAN is an extension of Naive Bayes classifier to allow a tree structure over
    the independent variables to account for interaction.

    Parameters
    ----------
    estimator_type : str, default="chow-liu"
        The algorithm to use for estimating the DAG. Options are:

        - ``"chow-liu"``: Chow-Liu maximum spanning tree algorithm.
        - ``"tan"``: Tree-augmented Naive Bayes (TAN).

    class_node : str, int, or any hashable python object, default=None
        Needed only if ``estimator_type='tan'``. In the estimated DAG, there
        will be edges from ``class_node`` to each of the feature variables.

    root_node : str, int, or any hashable python object, default=None
        The root node of the tree structure. If ``None``, the root node is
        auto-picked as the node with the highest sum of edge weights.

    edge_weights_fn : str or callable, default="mutual_info"
        Method to use for computing edge weights. Options are:

        - ``"mutual_info"``: Mutual Information Score.
        - ``"adjusted_mutual_info"``: Adjusted Mutual Information Score.
        - ``"normalized_mutual_info"``: Normalized Mutual Information Score.
        - A callable of the form ``fn(array, array) -> float``.

    n_jobs : int, default=-1
        Number of jobs to run in parallel. ``-1`` means use all processors.

    show_progress : bool, default=True
        If ``True``, shows a progress bar for the running algorithm.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph as a DAG.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph, i.e.
        ``causal_graph_``.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    References
    ----------
    .. [1] Chow, C. K.; Liu, C.N. (1968), "Approximating discrete probability
       distributions with dependence trees", IEEE Transactions on Information
       Theory, IT-14 (3): 462–467

    .. [2] Friedman N, Geiger D and Goldszmidt M (1997). Bayesian network
       classifiers. Machine Learning 29: 131–163

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import networkx as nx
    >>> import matplotlib.pyplot as plt
    >>> from pgmpy.causal_discovery import TreeSearch
    >>> values = pd.DataFrame(
    ...     np.random.randint(low=0, high=2, size=(1000, 5)),
    ...     columns=["A", "B", "C", "D", "E"],
    ... )

    Chow-Liu with a fixed root node:

    >>> est = TreeSearch(root_node="B")
    >>> est.fit(values)
    TreeSearch(root_node='B')
    >>> est.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> nx.draw_circular(
    ...     est.causal_graph_, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
    ... )
    >>> plt.show()

    Chow-Liu with auto root selection:

    >>> est = TreeSearch()
    >>> est.fit(values)
    TreeSearch()
    >>> est.causal_graph_ # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> nx.draw_circular(
    ...     est.causal_graph_, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
    ... )
    >>> plt.show()

    TAN with a fixed root node and class node:

    >>> est = TreeSearch(estimator_type="tan", class_node="A", root_node="B")
    >>> est.fit(values)
    TreeSearch(class_node='A', estimator_type='tan', root_node='B')
    >>> est.causal_graph_ # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> nx.draw_circular(
    ...     est.causal_graph_, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
    ... )
    >>> plt.show()

    TAN with automatic root selection:

    >>> est = TreeSearch(estimator_type="tan", class_node="A")
    >>> est.fit(values)
    TreeSearch(class_node='A', estimator_type='tan')
    >>> est.causal_graph_ # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> nx.draw_circular(
    ...     est.causal_graph_, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
    ... )
    >>> plt.show()
    """

    def __init__(
        self,
        estimator_type: str = "chow-liu",
        class_node=None,
        root_node=None,
        edge_weights_fn: str = "mutual_info",
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        self.estimator_type = estimator_type
        self.class_node = class_node
        self.root_node = root_node
        self.edge_weights_fn = edge_weights_fn
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame):
        """
        Estimate the ``DAG`` structure that fits best to the given data set
        without parametrization.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from. Each column
            represents one variable.

        Returns
        -------
        self : TreeSearch
            Returns the instance with the fitted attributes set.
        """
        # Step 1: Argument checks
        # Step 1.1: Only chow-liu and tan allowed as estimator type.
        if self.estimator_type not in {"chow-liu", "tan"}:
            raise ValueError(f"Invalid estimator_type. Expected either chow-liu or tan. Got: {self.estimator_type}")

        # Step 1.2: If estimator_type=tan, class_node must be specified and valid.
        if self.estimator_type == "tan" and self.class_node is None:
            raise ValueError("class_node argument must be specified for estimator_type='tan'")
        if self.estimator_type == "tan" and self.class_node not in X.columns:
            raise ValueError(f"Class node: {self.class_node} not found in data columns")

        # Step 1.3: Validate root_node if explicitly provided.
        if self.root_node is not None and self.root_node not in X.columns:
            raise ValueError(f"Root node: {self.root_node} not found in data columns.")

        # Step 1.4: Use a local variable so self.root_node is never mutated
        # between calls to fit(), preserving sklearn idempotence guarantees.
        root_node = self.root_node

        # Step 1.5: If root_node isn't specified, auto-pick the node with the
        # highest sum of edge weights.
        weights_computed = False
        if root_node is None:
            weights = TreeSearch._get_weights(X, self.edge_weights_fn, self.n_jobs, self.show_progress)
            weights_computed = True
            sum_weights = weights.sum(axis=0)
            maxw_idx = np.argsort(sum_weights)[::-1]
            root_node = X.columns[maxw_idx[0]]

        # Step 2: Compute all edge weights (skip recomputation if already done).
        if self.estimator_type == "chow-liu":
            if not weights_computed:
                weights = TreeSearch._get_weights(X, self.edge_weights_fn, self.n_jobs, self.show_progress)
        else:
            weights = TreeSearch._get_conditional_weights(
                X, self.class_node, self.edge_weights_fn, self.n_jobs, self.show_progress
            )

        # Step 3: Build the causal graph and store fitted attributes.
        if self.estimator_type == "chow-liu":
            self.causal_graph_ = TreeSearch._create_tree_and_dag(weights, X.columns, root_node)

        elif self.estimator_type == "tan":
            # Step 4.1: Ensure root_node and class_node are distinct.
            if root_node == self.class_node:
                raise ValueError(f"Root node: {root_node} and class node: {self.class_node} are identical")

            # Step 4.2: Construct Chow-Liu DAG on {data.columns - class_node}.
            class_node_idx = np.where(X.columns == self.class_node)[0][0]
            weights = np.delete(weights, class_node_idx, axis=0)
            weights = np.delete(weights, class_node_idx, axis=1)
            reduced_columns = np.delete(X.columns, class_node_idx)
            D = TreeSearch._create_tree_and_dag(weights, reduced_columns, root_node)

            # Step 4.3: Add edges from class_node to all other nodes.
            D.add_edges_from([(self.class_node, node) for node in reduced_columns])
            self.causal_graph_ = D

        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype="int")

        return self

    @staticmethod
    def _get_weights(data, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True):
        """
        Compute the pairwise edge weight matrix for the Chow-Liu algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe object where each column represents one variable.

        edge_weights_fn : str or callable, default="mutual_info"
            Method to use for computing edge weights. Options are:

            - ``"mutual_info"``: Mutual Information Score.
            - ``"adjusted_mutual_info"``: Adjusted Mutual Information Score.
            - ``"normalized_mutual_info"``: Normalized Mutual Information Score.
            - A callable of the form ``fn(array, array) -> float``.

        n_jobs : int, default=-1
            Number of jobs to run in parallel. ``-1`` means use all processors.

        show_progress : bool, default=True
            If ``True``, shows a progress bar for the running algorithm.

        Returns
        -------
        weights : np.ndarray, shape (n_columns, n_columns)
            Symmetric matrix where each element represents an edge weight.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.causal_discovery import TreeSearch
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> weights = TreeSearch._get_weights(values, show_progress=False)
        >>> weights.shape
        (5, 5)
        """
        # Step 0: Resolve the edge weight computation function.
        if edge_weights_fn == "mutual_info":
            edge_weights_fn = mutual_info_score
        elif edge_weights_fn == "adjusted_mutual_info":
            edge_weights_fn = adjusted_mutual_info_score
        elif edge_weights_fn == "normalized_mutual_info":
            edge_weights_fn = normalized_mutual_info_score
        elif not callable(edge_weights_fn):
            raise ValueError(
                f"edge_weights_fn should either be 'mutual_info', 'adjusted_mutual_info', "
                f"'normalized_mutual_info', or a callable of the form fn(array, array). "
                f"Got: {edge_weights_fn}"
            )

        # Step 1: Compute edge weights for a fully connected graph.
        n_vars = len(data.columns)
        pbar = combinations(data.columns, 2)
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(pbar, total=(n_vars * (n_vars - 1) / 2), desc="Building tree")

        vals = Parallel(n_jobs=n_jobs)(delayed(edge_weights_fn)(data.loc[:, u], data.loc[:, v]) for u, v in pbar)
        weights = np.zeros((n_vars, n_vars))
        indices = np.triu_indices(n_vars, k=1)
        weights[indices] = vals
        weights.T[indices] = vals

        return weights

    @staticmethod
    def _get_conditional_weights(data, class_node, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True):
        """
        Compute the conditional pairwise edge weight matrix for the TAN algorithm.

        Each weight is the conditional mutual information I(X; Y | class_node).

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe object where each column represents one variable.

        class_node : str
            The class node for TAN. Edge weights are computed as
            I(X, Y | class_node).

        edge_weights_fn : str or callable, default="mutual_info"
            Method to use for computing edge weights. Options are:

            - ``"mutual_info"``: Mutual Information Score.
            - ``"adjusted_mutual_info"``: Adjusted Mutual Information Score.
            - ``"normalized_mutual_info"``: Normalized Mutual Information Score.
            - A callable of the form ``fn(array, array) -> float``.

        n_jobs : int, default=-1
            Number of jobs to run in parallel. ``-1`` means use all processors.

        show_progress : bool, default=True
            If ``True``, shows a progress bar for the running algorithm.

        Returns
        -------
        weights : np.ndarray, shape (n_columns, n_columns)
            Symmetric matrix where each element represents a conditional edge
            weight.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.causal_discovery import TreeSearch
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> weights = TreeSearch._get_conditional_weights(values, class_node="A", show_progress=False)
        >>> weights.shape
        (5, 5)
        """
        # Step 0: Resolve the edge weight computation function.
        if edge_weights_fn == "mutual_info":
            edge_weights_fn = mutual_info_score
        elif edge_weights_fn == "adjusted_mutual_info":
            edge_weights_fn = adjusted_mutual_info_score
        elif edge_weights_fn == "normalized_mutual_info":
            edge_weights_fn = normalized_mutual_info_score
        elif not callable(edge_weights_fn):
            raise ValueError(
                f"edge_weights_fn should either be 'mutual_info', 'adjusted_mutual_info', "
                f"'normalized_mutual_info', or a callable of the form fn(array, array). "
                f"Got: {edge_weights_fn}"
            )

        # Step 1: Compute conditional edge weights for a fully connected graph.
        n_vars = len(data.columns)
        pbar = combinations(data.columns, 2)
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(pbar, total=(n_vars * (n_vars - 1) / 2), desc="Building tree")

        def _conditional_edge_weights_fn(u, v):
            """
            Computes I(u; v | class_node) as a weighted sum over class values.
            """
            cond_marginal = data.loc[:, class_node].value_counts() / data.shape[0]
            cond_edge_weight = 0.0
            for index, marg_prob in cond_marginal.items():
                df_cond_subset = data[data.loc[:, class_node] == index]
                cond_edge_weight += marg_prob * edge_weights_fn(df_cond_subset.loc[:, u], df_cond_subset.loc[:, v])
            return cond_edge_weight

        vals = Parallel(n_jobs=n_jobs)(delayed(_conditional_edge_weights_fn)(u, v) for u, v in pbar)
        weights = np.zeros((n_vars, n_vars))
        indices = np.triu_indices(n_vars, k=1)
        weights[indices] = vals
        weights.T[indices] = vals

        return weights

    @staticmethod
    def _create_tree_and_dag(weights, columns, root_node):
        """
        Build a DAG by computing the maximum spanning tree from a weight matrix
        and directing all edges away from ``root_node`` via BFS.

        Parameters
        ----------
        weights : np.ndarray, shape (n_columns, n_columns)
            Symmetric matrix where each element represents an edge weight.

        columns : list or array-like
            Names of the columns (and rows) of the weight matrix.

        root_node : str, int, or any hashable python object
            The root node of the tree structure.

        Returns
        -------
        model : pgmpy.base.DAG
            The estimated model structure.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.causal_discovery import TreeSearch
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> est = TreeSearch(root_node="B")
        >>> est.fit(values)
        TreeSearch(root_node='B')
        >>> est.causal_graph_  # doctest: +ELLIPSIS
        <pgmpy.base.DAG.DAG object at 0x...>
        """
        # Step 1: Compute the maximum spanning tree using the weights.
        T = nx.maximum_spanning_tree(
            nx.from_pandas_adjacency(
                pd.DataFrame(weights, index=columns, columns=columns),
                create_using=nx.Graph,
            )
        )

        # Step 2: Create DAG by directing all edges away from root_node and return.
        D = nx.bfs_tree(T, root_node)
        return DAG(D)
