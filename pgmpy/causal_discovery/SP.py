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

    Setting ``variant="greedy"`` instead runs the Greedy Sparsest Permutation (GSP) algorithm, a bounded approximation
    of SP that scales to much larger variable sets by performing a depth-first search over covered-edge reversals from a
    small number of random starting permutations instead of exhaustively enumerating all permutations. At each step, the
    search greedily reverses the covered edge that removes the most redundant edges; if no reversal removes any edges,
    it takes a non-improving step (up to `depth` of them, with backtracking) to escape the current Markov equivalence
    class in search of a sparser one.

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
        - 'greedy': The Greedy Sparsest Permutation (GSP) algorithm, searching over covered-edge reversals from
          `n_restarts` random starting permutations, up to `depth` non-improving steps each.

    max_iter : int or None, default=None
        Maximum number of permutations to evaluate. If None, all possible permutations are considered. Only used when
        ``variant="exhaustive"``; ignored when ``variant="greedy"``.

    return_type : str, default='dag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    depth : int or None, default=4
        Only used when ``variant="greedy"``. Maximum number of consecutive non-improving steps to take during the
        depth-first search over covered-edge reversals before backtracking / restarting from a new permutation. The
        average Markov equivalence class has around 4 members, so the default of 4 is usually sufficient to escape a
        Markov equivalence class of minimal I-MAPs. Use None to search with unlimited depth.

    n_restarts : int, default=5
        Only used when ``variant="greedy"``. Number of independent GSP restarts from random starting permutations. It is
        recommended to scale this with the number of variables (e.g. `n_restarts` on the order of the number of
        variables) for reliable recovery in the low-dimensional regime; for large, sparse, high-dimensional graphs, a
        small `depth` (e.g. 1) with many restarts (e.g. 50) is more computationally efficient.

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
        Only set when `variant="exhaustive"`. All permutations that produce a DAG with the minimum number of edges.

    restart_n_edges_ : list[int]
        Only set when `variant="greedy"`. The number of edges of the final DAG found by each restart, in the order the
        restarts were run. Useful for sanity-checking how much variance there is across restarts.

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

    >>> gsp = SP(ci_test="chi_square", variant="greedy", depth=4, n_restarts=10)
    >>> gsp.fit(df)
    SP(ci_test='chi_square', depth=4, n_restarts=10, variant='greedy')
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
        depth: int | None = 4,
        n_restarts: int = 5,
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
            enabling early pruning of unpromising permutations. Only used by the exhaustive variant.

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

    def _covered_edge_reversals(self, dag: DAG) -> list[tuple[DAG, int]]:
        """
        Find every covered edge in `dag` and return, for each one, the DAG obtained by reversing it (with any
        now-redundant edges trimmed away) together with how many edges were trimmed. Only used by the greedy variant.

        An edge u -> v is "covered" if parents(u) == parents(v) - {u}. Reversing a covered edge always yields another
        DAG in the same Markov equivalence class and preserves acyclicity, which is what makes it a valid local move for
        the search. Since u -> v is covered, parents(u) is exactly the set of parents shared by u and v; after reversing
        the edge, parents(u) becomes that shared set plus v, and parents(v) becomes just the shared set, so each shared
        parent is re-tested for conditional independence under its new conditioning set and dropped if it's no longer
        needed.

        Parameters
        ----------
        dag : DAG
            The current DAG.

        Returns
        -------
        list[tuple[DAG, int]]
            One (new_dag, n_edges_removed) pair per covered edge in `dag`, where new_dag has that edge reversed and any
            now-redundant edges removed, and n_edges_removed is how many edges were removed during that trimming.
        """
        moves = []

        for u, v in dag.edges():
            shared_parents = set(dag.get_parents(u))
            parents_v = set(dag.get_parents(v)) - {u}
            if shared_parents != parents_v:
                continue  # u -> v is not covered

            new_dag = dag.copy()
            new_dag.remove_edge(u, v)
            new_dag.add_edge(v, u)

            n_removed = 0
            for parent in shared_parents:
                rest = shared_parents - {parent}

                # After the reversal, parents(u) = shared_parents U {v}. Test whether `parent` is
                # still needed given the rest of the shared parents plus the newly reversed edge.
                independent_u = self.ci_test_(
                    X=u,
                    Y=parent,
                    Z=list(rest | {v}),
                    significance_level=self.significance_level,
                )
                if independent_u:
                    new_dag.remove_edge(parent, u)
                    n_removed += 1

                # After the reversal, parents(v) = shared_parents. Test whether `parent` is still
                # needed given the rest of the shared parents.
                independent_v = self.ci_test_(
                    X=v,
                    Y=parent,
                    Z=list(rest),
                    significance_level=self.significance_level,
                )
                if independent_v:
                    new_dag.remove_edge(parent, v)
                    n_removed += 1

            moves.append((new_dag, n_removed))

        return moves

    def _greedy_search(self, start_dag: DAG, rng: np.random.Generator) -> DAG:
        """
        Depth-first search over covered-edge reversals starting from `start_dag`, looking for a strictly sparser DAG
        within `self.depth` non-improving hops. Only used by the greedy variant.

        At every state, all covered-edge reversals are evaluated. If any reversal makes the DAG strictly sparser, the
        search jumps to one such DAG (picked at random among the sparser options) and restarts its depth-first
        exploration from there. Otherwise, the search explores same-size neighbours depth-first (up to `self.depth`),
        backtracking when it runs out of unexplored, same-size neighbours. The search stops -- for this restart -- once
        no unvisited neighbour (sparser or same-size, within depth) remains.
        """
        current_dag = start_dag
        trace = []
        visited = set()

        while True:
            visited.add(frozenset(current_dag.edges()))

            candidate_moves = [
                (n_removed, new_dag)
                for new_dag, n_removed in self._covered_edge_reversals(current_dag)
                if frozenset(new_dag.edges()) not in visited
            ]

            sparser_dags = [d for n, d in candidate_moves if n > 0]

            if sparser_dags:
                # Found a sparser I-MAP: jump there and restart the local search.
                current_dag = sparser_dags[rng.integers(len(sparser_dags))]
                trace = []
                visited = set()
            elif candidate_moves and (self.depth is None or len(trace) < self.depth):
                # No sparser I-MAP available: explore a same-size neighbour depth-first.
                lateral_dags = [d for n, d in candidate_moves if n == 0]
                trace.append(current_dag)
                current_dag = lateral_dags[rng.integers(len(lateral_dags))]
            elif trace:
                # Dead end: backtrack.
                current_dag = trace.pop()
            else:
                # No moves left and nothing to backtrack to: this restart is done.
                break

        return current_dag

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the SP algorithm. Runs the exhaustive or greedy search depending on `self.variant`.

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
        variant = self.variant.lower()
        if variant not in ("exhaustive", "greedy"):
            raise ValueError(f"variant must be one of: exhaustive, greedy. Got: {self.variant}")
        if self.return_type.lower() not in ("dag", "pdag"):
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        if variant == "exhaustive":
            if self.max_iter is not None and self.max_iter < 1:
                raise ValueError(
                    f"max_iter must be at least 1 to evaluate at least one permutation, got {self.max_iter}."
                )
        else:
            if self.n_restarts < 1:
                raise ValueError(f"n_restarts must be at least 1, got {self.n_restarts}.")
            if self.depth is not None and self.depth < 0:
                raise ValueError(f"depth must be non-negative or None, got {self.depth}.")

        # Step 1: Initialize variables and data structures.
        self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
        nodes = list(self.feature_names_in_)
        rng = np.random.default_rng(self.seed)

        # Step 2: Run the search.
        if variant == "exhaustive":
            rng.shuffle(nodes)

            min_edges = np.inf
            best_ordering = None
            best_edges = None
            optimal_permutations = []

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

            self.optimal_permutations_ = optimal_permutations

            best_dag = DAG()
            best_dag.add_nodes_from(best_ordering)
            best_dag.add_edges_from(best_edges)

        else:
            best_dag = None
            best_n_edges = np.inf
            self.restart_n_edges_ = []

            for _ in tqdm(
                range(self.n_restarts),
                desc="Running GSP restarts",
                disable=not (self.show_progress and config.SHOW_PROGRESS),
            ):
                permutation = nodes.copy()
                rng.shuffle(permutation)
                permutation = tuple(permutation)

                start_dag = DAG()
                start_dag.add_nodes_from(permutation)
                start_dag.add_edges_from(self._build_imap_edges(permutation))
                final_dag = self._greedy_search(start_dag, rng)

                n_edges = len(final_dag.edges())
                self.restart_n_edges_.append(n_edges)

                if n_edges < best_n_edges:
                    best_n_edges = n_edges
                    best_dag = final_dag

        # Step 3: Assign attributes.
        if self.return_type.lower() == "dag":
            self.causal_graph_ = best_dag
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = best_dag.to_pdag()

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self
