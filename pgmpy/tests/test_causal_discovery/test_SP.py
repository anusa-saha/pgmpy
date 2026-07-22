import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery import SP


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [SP(max_iter=2)],
    expected_failed_checks=expected_failed_checks,
)
def test_sp_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def chain_data():
    rng = np.random.default_rng(42)
    n = 1000
    A = rng.normal(size=n)
    B = A + rng.normal(scale=0.1, size=n)
    C = B + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"A": A, "B": B, "C": C})

@pytest.fixture
def collider_data():
    rng = np.random.default_rng(42)
    n = 1000
    A = rng.normal(size=n)
    B = rng.normal(size=n)
    C = A + B + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"A": A, "B": B, "C": C})


class TestSP:
    def test_chain_recovery_and_attributes(self, chain_data):
        est = SP(ci_test="pearsonr", return_type="dag")
        est.fit(chain_data)
        assert isinstance(est.causal_graph_, DAG)

        graph = est.causal_graph_.to_undirected()
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        assert graph.number_of_edges() == 2

        assert est.n_features_in_ == 3
        assert list(est.feature_names_in_) == ["A", "B", "C"]
        
        adj = est.adjacency_matrix_
        assert sorted(list(adj.index)) == ["A", "B", "C"]
        assert sorted(list(adj.columns)) == ["A", "B", "C"]
        assert np.all(np.diag(adj) == 0)

    def test_returns_pdag(self, chain_data):
        est = SP(ci_test="pearsonr", return_type="pdag")
        est.fit(chain_data)
        assert isinstance(est.causal_graph_, PDAG)

    def test_collider_recovery(self, collider_data):
        est = SP(ci_test="pearsonr")
        est.fit(collider_data)
        assert set(est.causal_graph_.edges()) == {("A", "C"), ("B", "C")}

    def test_seed_and_max_iter(self, chain_data):
        est1 = SP(ci_test="pearsonr", max_iter=10, seed=42)
        est2 = SP(ci_test="pearsonr", max_iter=10, seed=42)
        est1.fit(chain_data)
        est2.fit(chain_data)
        assert set(est1.causal_graph_.edges()) == set(est2.causal_graph_.edges())
        assert est1.optimal_permutations_ == est2.optimal_permutations_
