"""
模型验证模块单元测试
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from aerorom.identification import StateSpaceModel
from aerorom.validation import ModelValidator


class TestModelValidator(unittest.TestCase):
    """测试ModelValidator类"""

    def setUp(self):
        """测试前准备"""
        self.dt = 0.01
        self.n_samples = 500
        self.n_states = 4
        self.n_inputs = 2
        self.n_outputs = 2

        np.random.seed(42)
        A = np.random.randn(self.n_states, self.n_states)
        A = A / np.max(np.abs(np.linalg.eigvals(A))) * 0.9
        B = np.random.randn(self.n_states, self.n_inputs)
        C = np.random.randn(self.n_outputs, self.n_states)
        D = np.zeros((self.n_outputs, self.n_inputs))

        self.model = StateSpaceModel(A, B, C, D, self.dt)
        self.model.name = 'TestModel'

        self.U = np.random.randn(self.n_samples, self.n_inputs)
        self.Y, _ = self.model.simulate(self.U)
        self.Y += 0.01 * np.random.randn(*self.Y.shape)

        self.validator = ModelValidator(self.model)

    def test_set_model(self):
        """测试设置模型"""
        new_model = StateSpaceModel(self.model.A, self.model.B, self.model.C, self.model.D, self.dt)
        self.validator.set_model(new_model)
        self.assertEqual(self.validator.model, new_model)

    def test_one_step_prediction(self):
        """测试一步预测"""
        Y_pred, errors = self.validator.one_step_prediction(self.U, self.Y)

        self.assertEqual(Y_pred.shape, self.Y.shape)
        self.assertEqual(errors.shape, self.Y.shape)

        rmse = np.sqrt(np.mean(errors**2))
        self.assertLess(rmse, 0.1)

    def test_multi_step_prediction(self):
        """测试多步预测"""
        Y_pred, errors = self.validator.multi_step_prediction(
            self.U, self.Y, step_size=50
        )

        self.assertEqual(Y_pred.shape, self.Y.shape)
        self.assertEqual(errors.shape, self.Y.shape)

    def test_compute_metrics(self):
        """测试误差指标计算"""
        Y_pred, errors = self.validator.one_step_prediction(self.U, self.Y)
        metrics = self.validator.compute_metrics(self.Y, Y_pred, names=['out1', 'out2'])

        self.assertIn('overall', metrics)
        self.assertIn('out1', metrics)
        self.assertIn('out2', metrics)

        for metric in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'VAF']:
            self.assertIn(metric, metrics['overall'])

        self.assertGreater(metrics['overall']['R2'], 0.9)
        self.assertGreater(metrics['overall']['VAF'], 0.9)

    def test_residual_analysis(self):
        """测试残差分析"""
        _, errors = self.validator.one_step_prediction(self.U, self.Y)
        analysis = self.validator.residual_analysis(errors, dt=self.dt)

        for i in range(self.n_outputs):
            key = f'output_{i}'
            self.assertIn(key, analysis)
            self.assertIn('mean', analysis[key])
            self.assertIn('std', analysis[key])
            self.assertIn('autocorrelation', analysis[key])
            self.assertIn('ljung_box', analysis[key])

    def test_stability_analysis(self):
        """测试稳定性分析"""
        analysis = self.validator.stability_analysis()

        self.assertIn('stable', analysis)
        self.assertTrue(analysis['stable'])
        self.assertIn('eigenvalues', analysis)
        self.assertIn('max_magnitude', analysis)
        self.assertLess(analysis['max_magnitude'], 1.0)

    def test_frequency_domain_validation(self):
        """测试频域验证"""
        analysis = self.validator.frequency_domain_validation(self.U, self.Y)

        self.assertIn('omega', analysis)
        self.assertIn('H_model', analysis)
        self.assertIn('H_empirical', analysis)
        self.assertIn('coherence', analysis)
        self.assertIn('mean_relative_error', analysis)

        self.assertGreater(analysis['mean_relative_error'], 0)

    def test_pole_zero_analysis(self):
        """测试零极点分析"""
        analysis = self.validator.pole_zero_analysis()

        self.assertIn('poles', analysis)
        self.assertIn('zeros', analysis)
        self.assertIn('stable', analysis)
        self.assertTrue(analysis['stable'])

    def test_compare_models(self):
        """测试模型比较"""
        model2 = self.model.reduce_order(n_reduced=3)
        models = [self.model, model2]

        comparison = self.validator.compare_models(models, self.U, self.Y, metric='RMSE')

        self.assertIn('ranked', comparison)
        self.assertEqual(len(comparison['ranked']), 2)

    def test_cross_validation(self):
        """测试交叉验证"""
        cv_results = self.validator.cross_validation(
            self.U, self.Y, n_folds=3, method='ARX',
            input_order=2, output_order=2
        )

        self.assertIn('fold_metrics', cv_results)
        self.assertIn('mean_metrics', cv_results)
        self.assertIn('std_metrics', cv_results)
        self.assertEqual(len(cv_results['fold_metrics']), 3)


if __name__ == '__main__':
    unittest.main()
