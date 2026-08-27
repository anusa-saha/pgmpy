from collections.abc import Callable
from itertools import islice, permutations
from math import factorial

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
    and returns either that DAG or its PDAG representation, depending on `return_type`.

    The algorithm is statistically consistent under the Sparsest Markov Representation (SMR) assumption, which is weaker
    than the restricted faithfulness assumption required by many constraint-based methods.

    Two search variants are supported:

    - 'exhaustive': The original sparsest permutation algorithm, every permutation of the variables (up to `max_iter`)
      is scored, and the sparsest I-MAP found across all of them is returned. This guarantees finding the globally
      sparsest I-MAP, but runs in the worst case in O(p!) time.
    - 'greedy': The greedy sparsest permutation algorithm, starting from a random ordering, the current
      I-MAP's covered edges are reversed via a bounded-depth depth-first search, moving to any resulting I-MAP that
      is strictly sparser, until no such move can be found within `search_depth` reversals. This is repeated from
      `n_restarts` random orderings, and the sparsest I-MAP found across all restarts is returned. This scales to
      hundreds of variables and is the version recommended for most applications.

    Parameters
    ----------
    ci_test : str or callable, default=None
        Conditional independence test to use for constructing the minimal I-MAP. This can be any of the CI test
        implemented in :mod:`pgmpy.ci_tests` or a custom function that follows the signature of the built-in CI tests.

        If None, the appropriate CI test will be chosen based on the data type.

    significance_level : float, default=0.01
        Significance level used by the conditional independence test.

    variant : str, default='exhaustive'
        The search strategy to use. Options are:

        - 'exhaustive': Full search over permutations, controlled by `max_iter`.
        - 'greedy': Bounded-depth local search with random restarts, controlled by `search_depth` and `n_restarts`.

    max_iter : int or None, default=None
        Maximum number of permutations to evaluate. Only used when `variant='exhaustive'`. If None, all possible
        permutations are considered.

    search_depth : int or None, default=4
        Maximum number of covered arrow reversals to explore at each step before giving up on finding a sparser I-MAP.
        Only used when `variant='greedy'`. The average Markov equivalence class contains around four graphs, so a depth
        of 4 is typically sufficient to escape it.

    n_restarts : int, default=10
        Number of random initial orderings to run the greedy search from. Only used when `variant='greedy'`. The
        sparsest I-MAP found across all restarts is returned.

    return_type : str, default='dag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    seed : int or None, default=None
        Seed for the random number generator used to shuffle/sample the variable orderings.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph as a directed acyclic graph (DAG) or partially directed acyclic graph (PDAG).

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    optimal_permutations_ : list[tuple]
        All permutations found that produce a DAG with the minimum number of edges.

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

    Or use the greedy variant (Algorithm 4), which scales to much larger numbers of variables:

    >>> sp_greedy = SP(ci_test="chi_square", variant="greedy", search_depth=4, n_restarts=10)
    >>> sp_greedy.fit(df)
    SP(ci_test='chi_square', variant='greedy')

    References
    ----------
    - :footcite:t:`raskutti2019learningdirectedacyclicgraphs`
    """

    def __init__(
        self,
        ci_test: str | Callable | None = None,
        significance_level: float = 0.01,
        variant: str = "exhaustive",
        max_iter: int | None = None,
        search_depth: int | None = 4,
        n_restarts: int = 1,
        return_type: str = "dag",
        show_progress: bool = True,
        seed: int | None = None,
    ):
        self.ci_test = ci_test
        self.significance_level = significance_level
        self.variant = variant
        self.max_iter = max_iter
        self.search_depth = search_depth
        self.n_restarts = n_restarts
        self.return_type = return_type
        self.show_progress = show_progress
        self.seed = seed

    def _build_imap_edges(
        self,
        permutation: tuple[str, ...],
        n_edge_limit: int | float = np.inf,
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

        n_edge_limit : int, default=np.inf
            Maximum number of edges allowed during construction. Returns None if the edge count exceeds this value,
            enabling early pruning of unpromising permutations.

        Returns
        -------
        list[tuple[str, str]] or None
            The edges of the minimal I-MAP, or None if the search was aborted early because `n_edge_limit` was exceeded.
        """
        edges = []

        for node_idx in range(1, len(permutation)):
            node = permutation[node_idx]
            predecessors = permutation[:node_idx]
            for predecessor in predecessors:
                conditioning_nodes = (p for p in predecessors if p != predecessor)
                independent = self.ci_test_(
                    X=predecessor,
                    Y=node,
                    Z=conditioning_nodes,
                    significance_level=self.significance_level,
                )
                if not independent:
                    edges.append((predecessor, node))
                    if len(edges) > n_edge_limit:
                        return None

        return edges

    def _find_sparser_imap(
        self,
        permutation: tuple[str, ...],
        edges: list[tuple[str, str]],
        max_depth: int | None,
    ) -> tuple[tuple[str, ...], list[tuple[str, str]]] | None:
        """
        Depth-first search for a sparser minimal I-MAP reachable from the current one via a weakly decreasing sequence
        of covered edge reversals.

        An edge (u, v) is covered if u and v have exactly the same parents, aside from u itself being a parent of v;
        this is checked via `DAG.get_parents` on the I-MAP built at each step of the search. Reversing a covered edge
        corresponds to swapping u and v in the ordering and rebuilding the I-MAP from that new ordering; this always
        produces a valid I-MAP, sometimes with the same number of edges and occasionally with strictly fewer. The search
        explores such reversals depth-first, up to `max_depth` reversals from the starting I-MAP (unbounded if
        `max_depth` is None), and returns as soon as it finds an I-MAP with strictly fewer edges than the starting one.

        Parameters
        ----------
        permutation : tuple of str
            The ordering whose I-MAP the search starts from.

        edges : list[tuple[str, str]]
            The edges of the I-MAP corresponding to `permutation`.

        max_depth : int or None
            Maximum number of covered edge reversals to explore. If None, the search is unbounded.

        Returns
        -------
        tuple[tuple[str, ...], list[tuple[str, str]]] or None
            The (ordering, edges) of a sparser I-MAP if one was found within `max_depth` reversals, else None.
        """
        start_n_edges = len(edges)
        visited = {frozenset(edges)}
        stack = [(permutation, edges, 0)]

        while stack:
            perm, es, depth = stack.pop()

            if max_depth is not None and depth >= max_depth:
                continue

            imap = DAG()
            imap.add_nodes_from(perm)
            imap.add_edges_from(es)

            for u, v in es:
                # (u, v) is covered if u and v have identical parent sets, other than u being a parent of v.
                if set(imap.get_parents(u)) != set(imap.get_parents(v)) - {u}:
                    continue

                new_perm = list(perm)
                idx_u, idx_v = new_perm.index(u), new_perm.index(v)
                new_perm[idx_u], new_perm[idx_v] = new_perm[idx_v], new_perm[idx_u]
                new_perm = tuple(new_perm)

                new_edges = self._build_imap_edges(new_perm)
                key = frozenset(new_edges)
                if key in visited:
                    continue
                visited.add(key)

                if len(new_edges) < start_n_edges:
                    return new_perm, new_edges

                stack.append((new_perm, new_edges, depth + 1))

        return None

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
        # Step 0: Check inputs.
        if self.variant.lower() not in ("exhaustive", "greedy"):
            raise ValueError(f"variant must be one of: exhaustive, greedy. Got: {self.variant}")
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1 to evaluate at least one permutation, got {self.max_iter}.")
        if self.return_type.lower() not in ("dag", "pdag"):
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        # Step 1: Initialize variables and data structures.
        self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
        nodes = list(self.feature_names_in_)
        rng = np.random.default_rng(self.seed)

        min_edges = np.inf
        best_ordering = None
        best_edges = None
        optimal_permutations = []

        # Step 2: Run the search, either over all permutations (exhaustive) or via bounded-depth local search with
        # random restarts (greedy)
        if self.variant.lower() == "exhaustive":
            rng.shuffle(nodes)
            max_permutations = factorial(len(nodes))
            n_iterations = min(self.max_iter, max_permutations) if self.max_iter is not None else max_permutations
            permutation_iter = islice(permutations(nodes), n_iterations)

            for permutation in tqdm(
                permutation_iter,
                total=n_iterations,
                desc="Searching over permutations",
                disable=not (self.show_progress and config.SHOW_PROGRESS),
            ):
                edges = self._build_imap_edges(permutation, n_edge_limit=min_edges)
                if edges is None:
                    continue

                n_edges = len(edges)
                if n_edges < min_edges:
                    min_edges = n_edges
                    best_ordering = permutation
                    best_edges = edges
                    optimal_permutations = [permutation]
                elif n_edges == min_edges:
                    optimal_permutations.append(permutation)

        else:  # variant == "greedy"
            for _ in tqdm(
                range(self.n_restarts),
                desc="Running greedy sparsest permutation search",
                disable=not (self.show_progress and config.SHOW_PROGRESS),
            ):
                permutation = tuple(rng.permutation(nodes))
                edges = self._build_imap_edges(permutation)

                while True:
                    sparser = self._find_sparser_imap(permutation, edges, self.search_depth)
                    if sparser is None:
                        break
                    permutation, edges = sparser

                n_edges = len(edges)
                if n_edges < min_edges:
                    min_edges = n_edges
                    best_ordering = permutation
                    best_edges = edges
                    optimal_permutations = [permutation]
                elif n_edges == min_edges:
                    optimal_permutations.append(permutation)

        self.optimal_permutations_ = optimal_permutations

        # Step 3: Construct the DAG using the optimal permutation and assign attributes.
        current_model = DAG()
        current_model.add_nodes_from(best_ordering)
        current_model.add_edges_from(best_edges)

        if self.return_type.lower() == "dag":
            self.causal_graph_ = current_model
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = current_model.to_pdag()

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self
