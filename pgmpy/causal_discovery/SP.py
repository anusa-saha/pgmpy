from collections.abc import Callable
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.ci_tests import get_ci_test


class SP(BaseCausalDiscovery):
    """
    The Sparsest Permutation (SP) algorithm and its greedy counterpart, the Greedy Sparsest
    Permutation (GSP) algorithm.

    In its exhaustive form (``variant='exhaustive'``), SP exhaustively searches over all
    permutations of the variables. For each permutation, it constructs the corresponding
    minimal independence map (I-MAP) using conditional independence tests and returns either
    that DAG or its PDAG representation, depending on `return_type`. The algorithm is
    statistically consistent under the Sparsest Markov Representation (SMR) assumption, which
    is weaker than the restricted faithfulness assumption required by many constraint-based
    methods. Unfortunately, it requires searching over all p! permutations of the p variables,
    which is only feasible for a small number of variables.

    In its greedy form (``variant='greedy'``), SP instead implements Algorithm 4 (the "Greedy
    Sparsest Permutation" algorithm, GSP) of :footcite:t:`solus2021consistency`. Starting from a
    random permutation, it repeatedly looks for a sparser minimal I-MAP reachable via a weakly
    decreasing sequence of *covered arrow reversals* of length at most `depth`, using a
    depth-first search. When no such sparser DAG can be found, the search restarts from a new
    random permutation. This is repeated `n_restarts` times, and the sparsest DAG found across
    all restarts is returned. This greedy variant is consistent under the same "triangle
    assumption" of :footcite:t:`solus2021consistency`, which is implied by faithfulness and
    implies the SMR assumption, and it scales to problems with hundreds of variables since
    (outside of the worst case) it only needs to explore a small neighborhood of permutation
    space rather than all p! permutations.

    Parameters
    ----------
    ci_test : str or callable, default=None
        Conditional independence test to use for constructing the minimal I-MAP. This can be any of the CI test
        implemented in :mod:`pgmpy.ci_tests` or a custom function that follows the signature of the built-in CI tests.

        If None, the appropriate CI test will be chosen based on the data type.

    significance_level : float, default=0.01
        Significance level used by the conditional independence test.

    variant : str, default='exhaustive'
        Which version of the algorithm to run. Options are:

        - 'exhaustive': The classic Sparsest Permutation algorithm, which searches over all
          (or up to `max_iter`) permutations of the variables.
        - 'greedy': The Greedy Sparsest Permutation algorithm (Algorithm 4 of
          :footcite:t:`solus2021consistency`), which uses a bounded depth-first search over
          covered arrow reversals, repeated over `n_restarts` random starting permutations.

    max_iter : int or None, default=None
        Only used when ``variant='exhaustive'``. Maximum number of permutations to evaluate.
        If None, all possible permutations are considered.

    depth : int or None, default=4
        Only used when ``variant='greedy'``. The maximum length `d` of the weakly decreasing
        sequence of covered arrow reversals explored via depth-first search before giving up on
        finding a sparser DAG from the current starting permutation. :footcite:t:`solus2021consistency`
        find that, since the average Markov equivalence class contains about four DAGs, a depth
        of 4 is typically sufficient and recommend it as a default. Use None for unbounded depth.

    n_restarts : int or None, default=None
        Only used when ``variant='greedy'``. The number `r` of random restarts to perform. If
        None, defaults to the number of variables `p`, following the recommendation of
        :footcite:t:`solus2021consistency` that `r` be of the same order of magnitude as `p` in
        the low-dimensional setting.

    return_type : str, default='dag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    seed : int or None, default=None
        Seed for the random number generator used to shuffle the variables before searching over
        permutations (exhaustive variant) or to select random restart permutations (greedy variant).

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph as a directed acyclic graph (DAG) or partially directed acyclic graph (PDAG).

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    optimal_permutations_ : list[tuple]
        All permutations found that produce a DAG with the minimum number of edges. For
        ``variant='exhaustive'`` this is exhaustive over all evaluated permutations; for
        ``variant='greedy'`` this only reflects the local optima discovered across the
        `n_restarts` runs and is not guaranteed to be exhaustive.

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

    Use the exhaustive SP algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import SP
    >>> sp = SP(ci_test="chi_square")
    >>> sp.fit(df)
    SP(ci_test='chi_square')
    >>> sp.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> sp.n_features_in_
    5

    Use the Greedy SP algorithm instead, which scales to much larger variable sets:

    >>> gsp = SP(ci_test="chi_square", variant="greedy", depth=4, n_restarts=5, seed=42)
    >>> gsp.fit(df)
    SP(ci_test='chi_square', depth=4, n_restarts=5, seed=42, variant='greedy')

    References
    ----------
    - :footcite:t:`raskutti2019learningdirectedacyclicgraphs`
    - :footcite:t:`solus2021consistency`
    """

    def __init__(
        self,
        ci_test: str | Callable | None = None,
        significance_level: float = 0.01,
        variant: str = "exhaustive",
        max_iter: int | None = None,
        depth: int | None = 4,
        n_restarts: int | None = None,
        return_type: str = "dag",
        show_progress: bool = True,
        seed: int | None = None,
    ):
        self.ci_test = ci_test
        self.significance_level = significance_level
        self.variant = variant
        self.max_iter = max_iter
        self.depth = depth
        self.n_restarts = n_restarts
        self.return_type = return_type
        self.show_progress = show_progress
        self.seed = seed

    def _build_imap_edges(
        self,
        permutation: tuple[str, ...],
        max_edges: int | float = np.inf,
    ) -> list[tuple[str, str]] | None:
        """
        Construct the edges of the minimal I-MAP for a given variable ordering.

        For each variable in the permutation (skipping the first, which has no predecessors and therefore contributes no
        edges), every preceding variable is considered as a candidate parent. An edge is added from a predecessor to the
        current variable if the two variables are conditionally dependent given all other predecessors of the current
        variable.

        Parameters
        ----------
        permutation : tuple of str
            A permutation of the variable names.

        max_edges : int, default=np.inf
            Maximum number of edges allowed during construction. Returns None if the edge count exceeds this value,
            enabling early pruning of unpromising permutations.

        Returns
        -------
        list[tuple[str, str]] or None
            The edges of the minimal I-MAP, or None if the search was aborted early because `max_edges` was exceeded.
        """
        edges = []

        for node_idx in range(1, len(permutation)):
            node = permutation[node_idx]
            predecessors = permutation[:node_idx]
            for predecessor in predecessors:
                conditioning_nodes = {p for p in predecessors if p != predecessor}
                independent = self.ci_test_(
                    X=predecessor,
                    Y=node,
                    Z=conditioning_nodes,
                    significance_level=self.significance_level,
                )
                if not independent:
                    edges.append((predecessor, node))
                    if len(edges) > max_edges:
                        return None

        return edges

    def _covered_arrow_neighbors(
        self,
        nodes: list[str],
        edges: list[tuple[str, str]],
    ) -> list[tuple[str, ...]]:
        # Identify covered arrows: u -> v with Pa(u) = Pa(v) \ {u}.
        parents = {}
        for u, v in edges:
            parents.setdefault(u, set())
            parents.setdefault(v, set()).add(u)
        covered_edges = [
            (u, v)
            for u, v in edges
            if parents.get(u, set()) == parents.get(v, set()) - {u}
        ]

        neighbor_permutations = []
        for u, v in covered_edges:
            new_edge_list = [
                (v, u) if (a, b) == (u, v) else (a, b) for a, b in edges
            ]
            reversed_dag = nx.DiGraph()
            reversed_dag.add_nodes_from(nodes)
            reversed_dag.add_edges_from(new_edge_list)
            neighbor_permutations.append(tuple(nx.topological_sort(reversed_dag)))

        return neighbor_permutations

    def _find_sparser_neighbor(
        self,
        nodes: list[str],
        permutation: tuple[str, ...],
        edges: list[tuple[str, str]],
        depth: int | float,
    ) -> tuple[tuple[str, ...], list[tuple[str, str]]] | None:
        n_edges = len(edges)
        visited = {permutation}
        stack = [(permutation, edges, 0)]

        while stack:
            cur_permutation, cur_edges, cur_depth = stack.pop()
            if cur_depth >= depth:
                continue

            for new_permutation in self._covered_arrow_neighbors(nodes, cur_edges):
                if new_permutation in visited:
                    continue
                visited.add(new_permutation)

                # Prune branches that would exceed the edge count of the DFS root, so only
                # weakly decreasing sequences are explored.
                new_edges = self._build_imap_edges(new_permutation, max_edges=n_edges)
                if new_edges is None:
                    continue

                if len(new_edges) < n_edges:
                    return new_permutation, new_edges

                stack.append((new_permutation, new_edges, cur_depth + 1))

        return None

    def _fit(self, X: pd.DataFrame):
        
        self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
        nodes = list(self.feature_names_in_)
        rng = np.random.default_rng(self.seed)

        variant = self.variant.lower()

        if variant == "exhaustive":
            # Sparsest Permutation: exhaustive search over all permutations
            nodes = list(nodes)
            rng.shuffle(nodes)

            if self.max_iter is not None and self.max_iter < 1:
                raise ValueError(
                    f"max_iter must be at least 1 to evaluate at least one permutation, got {self.max_iter}."
                )

            min_edges = np.inf
            best_ordering = None
            best_edges = None
            optimal_permutations = []

            if self.show_progress and config.SHOW_PROGRESS:
                if self.max_iter is not None:
                    total = self.max_iter
                else:
                    total = np.prod(range(1, len(nodes) + 1))
                pbar = tqdm(total=total, desc="Searching over permutations")

            for i, permutation in enumerate(permutations(nodes)):
                if self.max_iter is not None and i >= self.max_iter:
                    break

                if self.show_progress and config.SHOW_PROGRESS:
                    pbar.update(1)

                edges = self._build_imap_edges(
                    permutation,
                    max_edges=np.inf if min_edges == np.inf else min_edges,
                )
                if edges is None:
                    continue

                n_edges = len(edges)

                # If new graph with minimum edges is found, it restarts the list of optimal permutations
                if n_edges < min_edges:
                    min_edges = n_edges
                    best_ordering = permutation
                    best_edges = edges
                    optimal_permutations = [permutation]
                # If the graph is tied with current minimum, it adds it to the list
                elif n_edges == min_edges:
                    optimal_permutations.append(permutation)

            if self.show_progress and config.SHOW_PROGRESS:
                pbar.close()

        elif variant == "greedy":
            # ----- Greedy Sparsest Permutation (Algorithm 4) -----
            nodes = list(nodes)
            depth = self.depth if self.depth is not None else np.inf
            n_restarts = self.n_restarts if self.n_restarts is not None else len(nodes)

            if n_restarts < 1:
                raise ValueError(
                    f"n_restarts must be at least 1 to perform at least one run, got {n_restarts}."
                )
            if depth is not np.inf and depth < 1:
                raise ValueError(f"depth must be at least 1, got {depth}.")

            local_optima = []  # (permutation, edges) local optimum per restart

            if self.show_progress and config.SHOW_PROGRESS:
                pbar = tqdm(total=n_restarts, desc="Greedy SP: random restarts")

            for _ in range(n_restarts):
                # Step 3: select a random starting permutation.
                start_nodes = list(nodes)
                rng.shuffle(start_nodes)
                permutation = tuple(start_nodes)
                edges = self._build_imap_edges(permutation)

                # Steps 4-8: keep moving to a sparser minimal I-MAP until none is reachable
                # within `depth` covered arrow reversals.
                sparser = self._find_sparser_neighbor(nodes, permutation, edges, depth)
                while sparser is not None:
                    permutation, edges = sparser
                    sparser = self._find_sparser_neighbor(nodes, permutation, edges, depth)

                # Steps 5-6: record this run's local optimum.
                local_optima.append((permutation, edges))

                if self.show_progress and config.SHOW_PROGRESS:
                    pbar.update(1)

            if self.show_progress and config.SHOW_PROGRESS:
                pbar.close()

            # Step 11: return the sparsest DAG found across all restarts.
            min_edges = min(len(edges) for _, edges in local_optima)
            optimal_permutations = [
                permutation
                for permutation, edges in local_optima
                if len(edges) == min_edges
            ]
            best_ordering, best_edges = next(
                (permutation, edges)
                for permutation, edges in local_optima
                if len(edges) == min_edges
            )

        else:
            raise ValueError(
                f"variant must be one of: exhaustive, greedy. Got: {self.variant}"
            )

        self.optimal_permutations_ = optimal_permutations

        current_model = DAG()
        current_model.add_nodes_from(best_ordering)
        current_model.add_edges_from(best_edges)

        if self.return_type.lower() == "dag":
            self.causal_graph_ = current_model
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = current_model.to_pdag()
        else:
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self