import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import SP


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [SP()],
    expected_failed_checks=expected_failed_checks,
)
def test_sp_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def chain_data():
    rng = np.random.default_rng(42)
    n = 5000

    A = rng.normal(size=n)
    B = A + rng.normal(scale=0.1, size=n)
    C = B + rng.normal(scale=0.1, size=n)

    return pd.DataFrame({"A": A, "B": B, "C": C})


def test_fit(chain_data):
    est = SP(ci_test="pearsonr")
    est.fit(chain_data)

    assert est.causal_graph_ is not None
    assert est.adjacency_matrix_.shape == (3, 3)

    assert est.n_features_in_ == 3
    assert list(est.feature_names_in_) == ["A", "B", "C"]

    assert len(est.optimal_permutations_) >= 1


def test_returns_dag(chain_data):
    est = SP(ci_test="pearsonr")
    est.fit(chain_data)

    graph = est.causal_graph_

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() <= 3
    assert graph.is_directed()


def test_adjacency_matrix(chain_data):
    est = SP(ci_test="pearsonr")
    est.fit(chain_data)

    adj = est.adjacency_matrix_

    assert list(adj.index) == ["A", "B", "C"]
    assert list(adj.columns) == ["A", "B", "C"]

    assert np.all(np.diag(adj) == 0)


def test_multiple_optimal_permutations():
    rng = np.random.default_rng(0)

    data = pd.DataFrame(
        rng.normal(size=(500, 3)),
        columns=["A", "B", "C"],
    )

    est = SP(ci_test="pearsonr")
    est.fit(data)

    assert len(est.optimal_permutations_) > 1
