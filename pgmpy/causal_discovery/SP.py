from collections.abc import Callable
from itertools import permutations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.ci_tests import get_ci_test


class SP(BaseCausalDiscovery):
    """
    The Sparsest Permutation (SP) algorithm exhaustively searches over all permutations of the variables. For each
    permutation, it constructs the corresponding minimal independence map (I-MAP) using conditional independence tests
    and returns the DAG with the fewest edges.

    The algorithm is statistically consistent under the Sparsest Markov Representation (SMR) assumption, which is weaker
    than the restricted faithfulness assumption required by many constraint-based methods.

    Parameters
    ----------
    ci_test : str or callable, default=None
        Conditional independence test to use for constructing the minimal I-MAP. If ``None``, an appropriate test is
        automatically selected based on the data type.

    significance_level : float, default=0.01
        Significance level used by the conditional independence test.

    max_iter : int or None, default=None
        Maximum number of permutations to evaluate. If None, all possible permutations are considered.

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph as a directed acyclic graph.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    optimal_permutations_ : list[tuple]
        All permutations that produce a DAG with the minimum number of edges.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/cancer")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the SP algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import SP
    >>> sp = SP(ci_test="chi_square")
    >>> sp.fit(df)
    SP(ci_test='chi_square')
    >>> sp.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> sp.n_features_in_
    5

    References
    ----------
    - :footcite:t:`raskutti2019learningdirectedacyclicgraphs`
    """

    def __init__(
        self,
        ci_test: str | Callable | None = None,
        significance_level: float = 0.01,
        max_iter: int | None = None,
        seed: int | None = None,
        show_progress: bool = True,
    ):
        self.ci_test = ci_test
        self.significance_level = significance_level
        self.max_iter = max_iter
        self.seed = seed
        self.show_progress = show_progress

    def _build_dag(self, permutation: tuple[str, ...]) -> DAG:
        """
        Construct the minimal I-MAP for a given variable ordering.

        For each variable in the permutation, every preceding variable is considered as a candidate parent. An edge is
        added from a predecessor to the current variable if they are conditionally dependent given all other
        predecessors of the current variable.

        Parameters
        ----------
        permutation : tuple of str
            A permutation of the variable names.

        Returns
        -------
        pgmpy.base.DAG
            The minimal I-MAP consistent with the given permutation.
        """
        dag = DAG()
        dag.add_nodes_from(permutation)

        for node_idx, node in enumerate(permutation):
            predecessors = permutation[:node_idx]
            if not predecessors:
                continue
            for predecessor in predecessors:
                conditioning_nodes = [pred for pred in predecessors if pred != predecessor]
                independent = self.ci_test_(
                    X=predecessor,
                    Y=node,
                    Z=conditioning_nodes,
                    significance_level=self.significance_level,
                )
                if not independent:
                    dag.add_edge(predecessor, node)
        return dag

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the SP algorithm.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.SP
            Returns the instance with the fitted attributes.
        """
        self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
        nodes = list(X.columns)

        min_edges = np.inf
        best_ordering = None
        optimal_permutations = []

        if self.show_progress and config.SHOW_PROGRESS:
            if self.max_iter is not None:
                total = self.max_iter
            else:
                total = 1
                for i in range(2, len(nodes) + 1):
                    total *= i
            pbar = tqdm(total=total)
            pbar.set_description("Searching over permutations: 0")

        for i, permutation in enumerate(permutations(nodes)):
            if self.max_iter is not None and i >= self.max_iter:
                break
            dag = self._build_dag(permutation)
            n_edges = dag.number_of_edges()
            if n_edges < min_edges:
                min_edges = n_edges
                best_ordering = permutation
                optimal_permutations = [permutation]
            elif n_edges == min_edges:
                optimal_permutations.append(permutation)

            if self.show_progress and config.SHOW_PROGRESS:
                pbar.update(1)
                pbar.set_description(f"Searching over permutations: {i + 1}")

        if self.show_progress and config.SHOW_PROGRESS:
            pbar.close()
        self.optimal_permutations_ = optimal_permutations
        self.causal_graph_ = self._build_dag(best_ordering)
        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self
