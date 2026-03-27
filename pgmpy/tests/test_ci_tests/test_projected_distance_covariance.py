import unittest

from pgmpy.ci_tests import ProjectedDistanceCovariance
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestProjectedDistanceCovariance(unittest.TestCase):
    def setUp(self):
        model_indep = LinearGaussianBayesianNetwork(
            [
                ("Z1", "X"),
                ("Z2", "X"),
                ("Z3", "X"),
                ("Z1", "Y"),
                ("Z2", "Y"),
                ("Z3", "Y"),
            ]
        )
        cpd_z1 = LinearGaussianCPD("Z1", [0], 1)
        cpd_z2 = LinearGaussianCPD("Z2", [0], 1)
        cpd_z3 = LinearGaussianCPD("Z3", [0], 1)
        cpd_x = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
        cpd_y_indep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
        model_indep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_indep)
        self.df_indep = model_indep.simulate(n_samples=1000, seed=42)

        model_dep = LinearGaussianBayesianNetwork(
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
        cpd_y_dep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
        model_dep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_dep)
        self.df_dep = model_dep.simulate(n_samples=1000, seed=42)

    def test_projected_distance_covariance_linear(self):
        test = ProjectedDistanceCovariance(data=self.df_indep, random_state=42)

        # Non-conditional test
        test("X", "Y", [])
        self.assertAlmostEqual(round(test.statistic_, 3), 44.67)
        self.assertAlmostEqual(test.p_value_, 0.0)

        # Conditional test (independent)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertAlmostEqual(round(test.statistic_, 3), 1.489)
        self.assertEqual(round(test.p_value_, 4), 0.3)

        # Conditional test (dependent)
        test = ProjectedDistanceCovariance(data=self.df_dep, random_state=42)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertAlmostEqual(round(test.statistic_, 3), 40.412)
        self.assertAlmostEqual(test.p_value_, 0.0)
