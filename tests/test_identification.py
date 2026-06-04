"""
系统辨识模块单元测试
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from aerorom.identification import (
    StateSpaceModel,
    ERA,
    ARX,
    deconvolution
)


class TestStateSpaceModel(unittest.TestCase):
    """测试StateSpaceModel类"""

    def setUp(self):
        """测试前准备"""
        self.dt = 0.01
        self.A = np.array([
            [0.9, 0.1, 0.0],
            [-0.1, 0.9, 0.2],
            [0.0, -0.2, 0.8]
        ])
        self.B = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ])
        self.C = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        self.D = np.zeros((2, 2))

        self.model = StateSpaceModel(self.A, self.B, self.C, self.D, self.dt)
        self.model.name = 'TestModel'

    def test_model_properties(self):
        """测试模型属性"""
        self.assertEqual(self.model.n_states, 3)
        self.assertEqual(self.model.n_inputs, 2)
        self.assertEqual(self.model.n_outputs, 2)
        self.assertEqual(self.model.dt, self.dt)

    def test_simulate(self):
        """测试模型仿真"""
        n_samples = 100
        U = np.random.randn(n_samples, 2)
        x0 = np.array([0.1, -0.1, 0.05])

        Y, X = self.model.simulate(U, x0=x0)

        self.assertEqual(Y.shape, (n_samples, 2))
        self.assertEqual(X.shape, (n_samples, 3))
        np.testing.assert_allclose(X[0], x0)

    def test_frequency_response(self):
        """测试频率响应"""
        omega = np.logspace(-1, 2, 50)
        omega_out, H = self.model.frequency_response(omega)

        np.testing.assert_allclose(omega_out, omega)
        self.assertEqual(H.shape, (len(omega), 2, 2))
        self.assertTrue(np.all(np.isfinite(H)))

    def test_eigenvalues(self):
        """测试特征值计算"""
        eig = self.model.eigenvalues()
        self.assertEqual(len(eig), 3)
        expected_eig = np.linalg.eigvals(self.A)
        np.testing.assert_allclose(np.sort(eig), np.sort(expected_eig))

    def test_natural_frequencies_damping(self):
        """测试固有频率和阻尼比计算"""
        fn, zeta = self.model.natural_frequencies_damping()
        self.assertEqual(len(fn), 3)
        self.assertEqual(len(zeta), 3)

        eig = self.model.eigenvalues()
        magnitudes = np.abs(eig)
        self.assertTrue(np.all(magnitudes < 1.0) or np.any(zeta < 0))

    def test_reduce_order(self):
        """测试模型降阶"""
        reduced_model = self.model.reduce_order(n_reduced=2)
        self.assertEqual(reduced_model.n_states, 2)
        self.assertEqual(reduced_model.n_inputs, 2)
        self.assertEqual(reduced_model.n_outputs, 2)

        U = np.random.randn(50, 2)
        Y_orig, _ = self.model.simulate(U)
        Y_red, _ = reduced_model.simulate(U)

        self.assertEqual(Y_red.shape, Y_orig.shape)

    def test_to_continuous(self):
        """测试连续时间转换"""
        Ac, Bc, Cc, Dc = self.model.to_continuous(method='zoh')
        self.assertEqual(Ac.shape, (3, 3))
        self.assertEqual(Bc.shape, (3, 2))
        np.testing.assert_allclose(Cc, self.C)
        np.testing.assert_allclose(Dc, self.D)


class TestERA(unittest.TestCase):
    """测试ERA类"""

    def setUp(self):
        """测试前准备"""
        self.dt = 0.01
        self.era = ERA(dt=self.dt)

        self.n_samples = 500
        self.n_states = 4
        self.n_inputs = 2
        self.n_outputs = 2

        np.random.seed(42)
        A = np.random.randn(self.n_states, self.n_states)
        A = A / np.max(np.abs(np.linalg.eigvals(A))) * 0.95
        B = np.random.randn(self.n_states, self.n_inputs)
        C = np.random.randn(self.n_outputs, self.n_states)
        D = np.zeros((self.n_outputs, self.n_inputs))

        self.true_model = StateSpaceModel(A, B, C, D, self.dt)

        Y_impulse = np.zeros((self.n_samples, self.n_outputs, self.n_inputs))
        for j in range(self.n_inputs):
            U_impulse = np.zeros((self.n_samples, self.n_inputs))
            U_impulse[0, j] = 1.0 / self.dt
            Y_single, _ = self.true_model.simulate(U_impulse)
            Y_impulse[:, :, j] = Y_single

        U = 0.1 * np.random.randn(self.n_samples, self.n_inputs)
        Y, _ = self.true_model.simulate(U)
        self.Y = Y
        self.U = U
        self.Y_impulse = Y_impulse

    def test_identify(self):
        """测试ERA辨识（使用脉冲响应数据）"""
        model = self.era.identify(self.Y_impulse, n_states=self.n_states, block_rows=30)

        self.assertEqual(model.n_states, self.n_states)
        self.assertEqual(model.n_inputs, self.n_inputs)
        self.assertEqual(model.n_outputs, self.n_outputs)

        Y_pred, _ = model.simulate(self.U)
        rmse = np.sqrt(np.mean((Y_pred - self.Y)**2))
        self.assertLess(rmse, 1.0)

    def test_select_order(self):
        """测试阶数选择"""
        self.era.identify(self.Y_impulse, n_states=self.n_states, block_rows=30)
        suggested_order = self.era.select_order(max_states=20)
        self.assertGreater(suggested_order, 0)
        self.assertLessEqual(suggested_order, 20)

    def test_auto_order_selection(self):
        """测试自动阶数选择"""
        model = self.era.identify(self.Y_impulse, n_states=None, block_rows=30, tolerance=1e-4)
        self.assertGreater(model.n_states, 0)


class TestARX(unittest.TestCase):
    """测试ARX类"""

    def setUp(self):
        """测试前准备"""
        self.dt = 0.01
        self.arx = ARX(dt=self.dt)

        self.n_samples = 1000
        self.n_inputs = 2
        self.n_outputs = 2
        self.input_order = 3
        self.output_order = 3

        np.random.seed(42)
        U = np.random.randn(self.n_samples, self.n_inputs)

        Y = np.zeros((self.n_samples, self.n_outputs))
        for k in range(max(self.input_order, self.output_order), self.n_samples):
            for i in range(self.n_outputs):
                for j in range(self.output_order):
                    Y[k, i] -= 0.3 * Y[k - j - 1, i]
                for j in range(self.input_order):
                    Y[k, i] += 0.5 * U[k - j - 1, 0]
                    Y[k, i] += 0.2 * U[k - j - 1, 1]

        Y += 0.01 * np.random.randn(*Y.shape)
        self.U = U
        self.Y = Y

    def test_identify(self):
        """测试ARX辨识"""
        model = self.arx.identify(self.U, self.Y, self.input_order, self.output_order)

        expected_states = self.output_order * self.n_outputs + self.input_order * self.n_inputs
        self.assertEqual(model.n_states, expected_states)
        self.assertEqual(model.n_inputs, self.n_inputs)
        self.assertEqual(model.n_outputs, self.n_outputs)

        Y_pred, _ = model.simulate(self.U)
        rmse = np.sqrt(np.mean((Y_pred - self.Y)**2))
        self.assertLess(rmse, 2.0)

    def test_simulate(self):
        """测试ARX仿真"""
        model = self.arx.identify(self.U, self.Y, self.input_order, self.output_order)
        Y_pred = self.arx.simulate(self.U)
        self.assertEqual(Y_pred[0].shape, (self.n_samples, self.n_outputs))


class TestDeconvolution(unittest.TestCase):
    """测试解卷积函数"""

    def test_wiener_deconvolution(self):
        """测试Wiener解卷积"""
        dt = 0.01
        t = np.arange(0, 10, dt)
        h = np.exp(-t / 0.5) * np.sin(2 * np.pi * 3 * t)

        u = np.random.randn(len(t))
        y = np.convolve(u, h, mode='full')[:len(t)] * dt
        y += 0.01 * np.random.randn(len(y))

        h_est, residuals = deconvolution(y, u, dt, method='wiener', snr=100)

        correlation = np.corrcoef(h[:len(h_est)], h_est[:len(h)])[0, 1]
        self.assertGreater(correlation, 0.5)

    def test_least_squares_deconvolution(self):
        """测试最小二乘解卷积"""
        dt = 0.01
        t = np.arange(0, 5, dt)
        h = np.exp(-t / 0.3)

        u = np.random.randn(len(t))
        y = np.convolve(u, h, mode='full')[:len(t)] * dt
        y += 0.001 * np.random.randn(len(y))

        h_est, residuals = deconvolution(y, u, dt, method='least_squares', n=50)

        correlation = np.corrcoef(h[:50], h_est[:50])[0, 1]
        self.assertGreater(correlation, 0.7)


if __name__ == '__main__':
    unittest.main()
