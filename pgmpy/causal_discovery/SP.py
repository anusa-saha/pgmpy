from collections.abc import Callable
from itertools import islice, permutations
from math import factorial

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
    The Sparsest Permutation (SP) algorithm exhaustively searches over all permutations of the variables. For each
    permutation, it constructs the corresponding minimal independence map (I-MAP) using conditional independence tests
    and returns either that DAG or its PDAG representation, depending on `return_type`.

    The algorithm is statistically consistent under the Sparsest Markov Representation (SMR) assumption, which is weaker
    than the restricted faithfulness assumption required by many constraint-based methods.

    Setting ``variant="greedy"`` instead runs the Greedy Sparsest Permutation (GSP) algorithm, a bounded approximation
    of SP that scales to much larger variable sets by performing a depth-first search over covered-arrow reversals from
    a small number of random starting permutations instead of exhaustively enumerating all permutations. At each step,
    the search greedily reverses the covered arrow that removes the most redundant edges; if no reversal removes any
    edges, it takes a non-improving step (up to `search_depth` of them, with backtracking) to escape the current Markov
    equivalence class in search of a sparser one.

    Parameters
    ----------
    ci_test : str or callable, default=None
        Conditional independence test to use for constructing the minimal I-MAP. This can be any of the CI test
        implemented in :mod:`pgmpy.ci_tests` or a custom function that follows the signature of the built-in CI tests.

        If None, the appropriate CI test will be chosen based on the data type.

    significance_level : float, default=0.01
        Significance level used by the conditional independence test.

    variant : str, default='exhaustive'
        Which search strategy to use. Options are:

        - 'exhaustive': The original Sparsest Permutation (SP) algorithm, searching over all (or up to `max_iter`)
          permutations.
        - 'greedy': The Greedy Sparsest Permutation (GSP) algorithm, searching over covered-arrow reversals from
          `n_runs` random starting permutations, up to `search_depth` non-improving steps each.

    max_iter : int or None, default=None
        Maximum number of permutations to evaluate. If None, all possible permutations are considered. Only used when
        ``variant="exhaustive"``; ignored when ``variant="greedy"``.

    return_type : str, default='dag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    search_depth : int, default=4
        Only used when ``variant="greedy"``. Maximum number of consecutive non-improving steps to take during the
        depth-first search over covered-arrow reversals before backtracking / restarting from a new permutation. The
        average Markov equivalence class has around 4 members, so the default of 4 is usually sufficient to escape a
        Markov equivalence class of minimal I-MAPs.

    n_runs : int, default=10
        Only used when ``variant="greedy"``. Number of independent GSP restarts from random starting permutations. It is
        recommended to scale this with the number of variables (e.g. `n_runs` on the order of the number of variables)
        for reliable recovery in the low-dimensional regime; for large, sparse, high-dimensional graphs, a small
        `search_depth` (e.g. 1) with many runs (e.g. 50) is more computationally efficient.

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    seed : int or None, default=None
        Seed for the random number generator used to shuffle the variables before searching over permutations.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph as a directed acyclic graph (DAG) or partially directed acyclic graph (PDAG).

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    optimal_permutations_ : list[tuple]
        For ``variant="exhaustive"``, all permutations that produce a DAG with the minimum number of edges. For
        ``variant="greedy"``, a topological ordering of the sparsest DAG found by each run (out of `n_runs`) that tied
        for the minimum number of edges; this is not a guarantee that all globally optimal DAGs were found.

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

    Use the greedy GSP variant instead, which scales to much larger variable sets:

    >>> gsp = SP(ci_test="chi_square", variant="greedy", search_depth=4, n_runs=10)
    >>> gsp.fit(df)
    SP(ci_test='chi_square', n_runs=10, search_depth=4, variant='greedy')
    >>> gsp.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>

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
        return_type: str = "dag",
        search_depth: int = 4,
        n_runs: int = 10,
        show_progress: bool = True,
        seed: int | None = None,
    ):
        self.ci_test = ci_test
        self.significance_level = significance_level
        self.variant = variant
        self.max_iter = max_iter
        self.return_type = return_type
        self.search_depth = search_depth
        self.n_runs = n_runs
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

    def _reverse_covered_edges(self, model: DAG) -> list[DAG]:
        """
        Return every DAG obtainable from `model` by reversing a single covered arrow, with any edges that become
        redundant as a result of the reversal already dropped.

        An arrow u -> v is covered if Pa(u) = Pa(v) - {u}. Reversing a covered arrow produces a DAG that is Markov
        equivalent to `model` (same skeleton, same number of edges) *before* accounting for redundant edges. Since
        u -> v is covered, Pa(u) = Pa(v) - {u}, so looping over Pa(u) visits every common parent of u and v. After
        reversing the arrow, u gains v as a new parent while v loses u as a parent (its other parents are unchanged).
        For each common parent p, we test whether p is still needed as a parent of u given u's new parent set
        (Pa(u) - {p} + {v}), and whether p is still needed as a parent of v given its new parent set (Pa(u) - {p});
        edges found to be redundant this way are dropped from the returned neighbor DAG. This is what makes it
        possible to find a sparser I-MAP by walking between Markov equivalent DAGs.

        `model` itself is never mutated; each neighbor is built fresh from `model`'s nodes and edges.

        Parameters
        ----------
        model : DAG
            The current DAG.

        Returns
        -------
        list[DAG]
            One DAG per covered arrow in `model`, with that arrow reversed and any now-redundant edges removed.
        """
        neighbors = []

        for u, v in model.edges():
            parents_u = model.get_parents(u)
            parents_v = model.get_parents(v)

            if parents_u != (parents_v - {u}):
                continue

            removed_edges = set()
            for parent in parents_u:
                rest = parents_u - {parent}

                independent_u = self.ci_test_(
                    X=parent,
                    Y=u,
                    Z=list(rest) + [v],
                    significance_level=self.significance_level,
                )
                if independent_u:
                    removed_edges.add((parent, u))

                independent_v = self.ci_test_(
                    X=parent,
                    Y=v,
                    Z=list(rest),
                    significance_level=self.significance_level,
                )
                if independent_v:
                    removed_edges.add((parent, v))

            new_edges = [edge for edge in model.edges() if edge != (u, v) and edge not in removed_edges]
            new_edges.append((v, u))

            new_model = DAG()
            new_model.add_nodes_from(model.nodes())
            new_model.add_edges_from(new_edges)
            neighbors.append(new_model)

        return neighbors

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
        # Step 0: Check inputs
        if self.variant not in ("exhaustive", "greedy"):
            raise ValueError(f"variant must be one of: exhaustive, greedy. Got: {self.variant}")
        if self.return_type.lower() not in ("dag", "pdag"):
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        if self.variant == "exhaustive":
            if self.max_iter is not None and self.max_iter < 1:
                raise ValueError(
                    f"max_iter must be at least 1 to evaluate at least one permutation, got {self.max_iter}."
                )
        elif self.variant == "greedy":
            if self.search_depth < 1:
                raise ValueError(f"search_depth must be at least 1, got {self.search_depth}.")
            if self.n_runs < 1:
                raise ValueError(f"n_runs must be at least 1, got {self.n_runs}.")

        # Step 1: Initialize variables and data structures.
        self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
        nodes = list(self.feature_names_in_)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(nodes)

        # Step 2: Run the search according to variant.
        min_edges = np.inf
        best_ordering = None
        best_edges = None
        optimal_permutations = []

        if self.variant == "exhaustive":
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
                # If new graph with minimum edges is found, it restarts the list of optimal permutations
                if n_edges < min_edges:
                    min_edges = n_edges
                    best_ordering = permutation
                    best_edges = edges
                    optimal_permutations = [permutation]
                # If the graph is tied with current minimum, it adds it to the list
                elif n_edges == min_edges:
                    optimal_permutations.append(permutation)

        elif self.variant == "greedy":
            with tqdm(
                total=self.n_runs,
                desc="Running Greedy Sparsest Permutation",
                disable=not (self.show_progress and config.SHOW_PROGRESS),
            ) as pbar:
                for run in range(self.n_runs):
                    # Choose a random starting permutation for this run.
                    starting_permutation = list(nodes)
                    rng.shuffle(starting_permutation)
                    starting_permutation = tuple(starting_permutation)

                    edges = self._build_imap_edges(starting_permutation)
                    model = DAG()
                    model.add_nodes_from(starting_permutation)
                    model.add_edges_from(edges)

                    best_model = model
                    neighbors = self._reverse_covered_edges(model)
                    all_visited = {frozenset(model.edges())}
                    trace = []

                    while True:
                        max_removed = (
                            max(len(model.edges()) - len(neighbor.edges()) for neighbor in neighbors)
                            if neighbors
                            else 0
                        )

                        can_step = neighbors and (
                            len(trace) != self.search_depth or max_removed > 0
                        )

                        if can_step:
                            if max_removed > 0:
                                # A sparser I-MAP is reachable: take the best reversal and restart the search.
                                trace = []
                                all_visited = set()
                                candidate_idxs = [
                                    idx
                                    for idx, neighbor in enumerate(neighbors)
                                    if len(model.edges()) - len(neighbor.edges()) == max_removed
                                ]
                                chosen_idx = candidate_idxs[rng.integers(len(candidate_idxs))]
                                model = neighbors.pop(chosen_idx)
                                if len(model.edges()) < len(best_model.edges()):
                                    best_model = model
                            else:
                                # No reversal helps directly: take a non-improving step to escape this
                                # equivalence class.
                                trace.append((model, neighbors))
                                chosen_idx = rng.integers(len(neighbors))
                                model = neighbors.pop(chosen_idx)

                            all_visited.add(frozenset(model.edges()))
                            neighbors = self._reverse_covered_edges(model)

                            # Drop moves that would lead back to an already-visited DAG.
                            neighbors = [
                                neighbor
                                for neighbor in neighbors
                                if frozenset(neighbor.edges()) not in all_visited
                            ]
                        else:
                            if not trace:
                                # Reached a local minimum within the search depth: stop this run.
                                break
                            model, neighbors = trace.pop()

                    n_edges = len(best_model.edges())

                    # If new graph with minimum edges is found, it restarts the list of optimal permutations
                    if n_edges < min_edges:
                        min_edges = n_edges
                        best_ordering = tuple(nx.topological_sort(best_model))
                        best_edges = list(best_model.edges())
                        optimal_permutations = [best_ordering]
                    # If the graph is tied with current minimum, it adds it to the list
                    elif n_edges == min_edges:
                        optimal_permutations.append(tuple(nx.topological_sort(best_model)))

                    pbar.set_postfix(run=run + 1, best_edges=min_edges, last_run_edges=n_edges)
                    pbar.update(1)

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