from collections.abc import Callable
from itertools import permutations

import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.ci_tests import get_ci_test


class SP(BaseCausalDiscovery):
    """
    The Sparsest Permutation (SP) algorithm exhaustively searches over all
    permutations of the variables. For each permutation, it constructs the
    corresponding minimal independence map (I-MAP) using conditional
    independence tests and returns the DAG with the fewest edges.

    The algorithm is statistically consistent under the Sparsest Markov
    Representation (SMR) assumption, which is weaker than the restricted
    faithfulness assumption required by many constraint-based methods.

    Parameters
    ----------
    ci_test : str or callable, default=None
        Conditional independence test to use for constructing the minimal
        I-MAP. If ``None``, an appropriate test is automatically selected
        based on the data type.

    significance_level : float, default=0.01
        Significance level used by the conditional independence test.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph as a directed acyclic graph.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    all_optimal_permutations_ : list[tuple]
        All permutations that produce a DAG with the minimum number of edges.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    References
    ----------
    Garvesh Raskutti and Caroline Uhler.
    "Learning Directed Acyclic Graph Models Based on Sparsest Permutations."
    Stat, 2018.
    https://dspace.mit.edu/entities/publication/cda35a77-5918-476f-8e53-41f19f456f22
    """

    def __init__(
        self,
        ci_test: str | Callable | None = None,
        significance_level: float = 0.01,
    ):
        self.ci_test = ci_test
        self.significance_level = significance_level

    def _build_dag(self, permutation: tuple[str, ...]) -> DAG:
        """
        Construct the minimal I-MAP for a given variable ordering.

        For each variable in the permutation, every preceding variable is
        considered as a candidate parent. An edge is added from a predecessor
        to the current variable if they are conditionally dependent given all
        other predecessors of the current variable.

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

        for child_idx, child in enumerate(permutation):
            predecessors = permutation[:child_idx]
            if not predecessors:
                continue
            for parent in predecessors:
                conditioning_set = [node for node in predecessors if node != parent]
                independent = self.ci_test_(
                    X=parent,
                    Y=child,
                    Z=conditioning_set,
                    significance_level=self.significance_level,
                )
                if not independent:
                    dag.add_edge(parent, child)
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
        variables = list(X.columns)

        min_edges = np.inf
        best_permutation = None
        optimal_permutations = []

        for permutation in permutations(variables):
            dag = self._build_dag(permutation)
            num_edges = dag.number_of_edges()
            if num_edges < min_edges:
                min_edges = num_edges
                best_permutation = permutation
                optimal_permutations = [permutation]
            elif num_edges == min_edges:
                optimal_permutations.append(permutation)

        self.all_optimal_permutations_ = optimal_permutations
        self.causal_graph_ = self._build_dag(best_permutation)
        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self
