import os
import unittest

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.ci_tests import ProjectedDistanceCovariance
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.models import FunctionalBayesianNetwork, LinearGaussianBayesianNetwork


@unittest.skipUnless(
    _check_soft_dependencies("pyro-ppl", severity="none"),
    reason="requires pyro-ppl (FunctionalBayesianNetwork backend)",
)
@unittest.skipIf(os.getenv("GITHUB_ACTIONS") == "true", "Skipping residual tests on GitHub Actions.")
class TestProjectedDistanceCovariance(unittest.TestCase):
    def setUp(self):
        self.setUp_linear()
        self.setUp_nonlinear()

    def setUp_linear(self):
        model_indep_linear = LinearGaussianBayesianNetwork(
            [
                ("Z1", "X"),
                ("Z2", "X"),
                ("Z3", "X"),
                ("Z1", "Y"),
                ("Z2", "Y"),
                ("Z3", "Y"),
            ]
        )
        cpd_z1_linear = LinearGaussianCPD("Z1", [0], 1)
        cpd_z2_linear = LinearGaussianCPD("Z2", [0], 1)
        cpd_z3_linear = LinearGaussianCPD("Z3", [0], 1)
        cpd_x_linear = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
        cpd_y_indep_linear = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
        model_indep_linear.add_cpds(cpd_z1_linear, cpd_z2_linear, cpd_z3_linear, cpd_x_linear, cpd_y_indep_linear)
        self.df_indep_linear = model_indep_linear.simulate(n_samples=1000, seed=42)

        model_dep_linear = LinearGaussianBayesianNetwork(
            [
                ("Z1", "X"),
                ("Z2", "X"),
                ("Z3", "X"),
                ("Z1", "Y"),
                ("Z2", "Y"),
                ("Z3", "Y"),
                ("X", "Y"),
            ]
        )
        cpd_y_dep_linear = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
        model_dep_linear.add_cpds(cpd_z1_linear, cpd_z2_linear, cpd_z3_linear, cpd_x_linear, cpd_y_dep_linear)
        self.df_dep_linear = model_dep_linear.simulate(n_samples=1000, seed=42)

    def setUp_nonlinear(self):
        import pyro.distributions as dist

        config.set_backend("torch")

        model_indep_non_linear = FunctionalBayesianNetwork(
            [
                ("Z1", "X"),
                ("Z2", "X"),
                ("Z3", "X"),
                ("Z1", "Y"),
                ("Z2", "Y"),
                ("Z3", "Y"),
            ]
        )

        cpd_z1 = FunctionalCPD("Z1", lambda _: dist.Normal(0, 1))
        cpd_z2 = FunctionalCPD("Z2", lambda _: dist.Normal(0, 1))
        cpd_z3 = FunctionalCPD("Z3", lambda _: dist.Normal(0, 1))

        cpd_x = FunctionalCPD(
            "X",
            lambda p: dist.Normal(p["Z1"] ** 2 + p["Z2"] + p["Z3"], 1),
            parents=["Z1", "Z2", "Z3"],
        )

        cpd_y_indep = FunctionalCPD(
            "Y",
            lambda p: dist.Normal(p["Z1"] + p["Z2"] ** 2 + p["Z3"], 1),
            parents=["Z1", "Z2", "Z3"],
        )

        model_indep_non_linear.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_indep)
        self.df_indep_non_linear = model_indep_non_linear.simulate(n_samples=1000, seed=42)

        model_dep_non_linear = FunctionalBayesianNetwork(
            [
                ("Z1", "X"),
                ("Z2", "X"),
                ("Z3", "X"),
                ("Z1", "Y"),
                ("Z2", "Y"),
                ("Z3", "Y"),
                ("X", "Y"),
            ]
        )

        cpd_y_dep = FunctionalCPD(
            "Y",
            lambda p: dist.Normal(p["Z1"] + p["Z2"] ** 2 + p["Z3"] + np.sin(p["X"]), 1),
            parents=["Z1", "Z2", "Z3", "X"],
        )

        model_dep_non_linear.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_dep)
        self.df_dep_non_linear = model_dep_non_linear.simulate(n_samples=1000, seed=42)

    def test_projected_distance_covariance_linear(self):
        test = ProjectedDistanceCovariance(data=self.df_indep_linear, random_state=42)

        # Non-conditional test
        test("X", "Y", [])
        self.assertAlmostEqual(round(test.statistic_, 3), 44.67, delta=1)
        self.assertAlmostEqual(test.p_value_, 0.0)

        # Conditional test (independent)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertAlmostEqual(round(test.statistic_, 3), 1.489, delta=0.15)
        self.assertEqual(round(test.p_value_, 4), 0.3)

        # Conditional test (dependent)
        test = ProjectedDistanceCovariance(data=self.df_dep_linear, random_state=42)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertAlmostEqual(round(test.statistic_, 3), 40.412)
        self.assertAlmostEqual(test.p_value_, 0.0)

    def test_projected_distance_covariance_non_linear(self):
        test = ProjectedDistanceCovariance(data=self.df_indep_non_linear, random_state=42)

        # Non-conditional test
        test("X", "Y", [])
        self.assertAlmostEqual(round(test.statistic_, 3), 15.454)
        self.assertAlmostEqual(test.p_value_, 0.0)

        # Conditional test (independent)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertAlmostEqual(round(test.statistic_, 3), 0.59)
        self.assertEqual(round(test.p_value_, 4), 0.9)

        # Conditional test (dependent)
        test = ProjectedDistanceCovariance(data=self.df_dep_non_linear, random_state=42)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertAlmostEqual(round(test.statistic_, 3), 3.37)
        self.assertAlmostEqual(test.p_value_, 0.0)
