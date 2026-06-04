"""
可视化模块单元测试
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')
from aerorom.visualization import Plotter


class TestPlotter(unittest.TestCase):
    """测试Plotter类"""

    def setUp(self):
        """测试前准备"""
        self.plotter = Plotter(style='default', figsize=(8, 6), dpi=80)
        self.test_dir = tempfile.mkdtemp()

        self.n_samples = 1000
        self.dt = 0.01
        self.time = np.arange(self.n_samples) * self.dt

        np.random.seed(42)
        self.U = np.random.randn(self.n_samples, 2)
        self.Y = np.zeros((self.n_samples, 2))
        for k in range(1, self.n_samples):
            self.Y[k, 0] = 0.9 * self.Y[k-1, 0] + 0.5 * self.U[k-1, 0] + 0.3 * self.U[k-1, 1]
            self.Y[k, 1] = 0.8 * self.Y[k-1, 1] + 0.2 * self.U[k-1, 0] + 0.7 * self.U[k-1, 1]
        self.Y += 0.01 * np.random.randn(*self.Y.shape)

        self.Y_pred = self.Y + 0.05 * np.random.randn(*self.Y.shape)
        self.errors = self.Y - self.Y_pred

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.plotter.close_all()

    def test_plot_time_series(self):
        """测试时域信号绘制"""
        save_path = os.path.join(self.test_dir, 'time_series.png')
        fig, ax = self.plotter.plot_time_series(
            self.time, self.Y, labels=['y1', 'y2'],
            title='Test Time Series', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_comparison(self):
        """测试对比图绘制"""
        save_path = os.path.join(self.test_dir, 'comparison.png')
        fig, axes = self.plotter.plot_comparison(
            self.time, self.Y, self.Y_pred, labels=['y1', 'y2'],
            title='Test Comparison', show_error=True, save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertEqual(len(axes), 4)

    def test_plot_bode(self):
        """测试Bode图绘制"""
        omega = np.logspace(-1, 2, 100)
        H = np.zeros((len(omega), 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                H[:, i, j] = 1.0 / (1 + 1j * omega / (10 * (i + 1) * (j + 1)))

        save_path = os.path.join(self.test_dir, 'bode.png')
        fig, axes = self.plotter.plot_bode(
            omega, H, outputs=[0, 1], inputs=[0, 1],
            title='Test Bode', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertEqual(axes.shape, (2, 4))

    def test_plot_singular_values(self):
        """测试奇异值分布图"""
        singular_values = np.exp(-np.arange(20)) + 1e-10

        save_path = os.path.join(self.test_dir, 'singular_values.png')
        fig, ax = self.plotter.plot_singular_values(
            singular_values, normalize=True, suggested_order=5,
            title='Test Singular Values', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertIsNotNone(fig)

    def test_plot_residuals(self):
        """测试残差分析图"""
        save_path = os.path.join(self.test_dir, 'residuals.png')
        fig, axes = self.plotter.plot_residuals(
            self.time, self.errors, labels=['y1', 'y2'],
            title='Test Residuals', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertEqual(axes.shape, (2, 3))

    def test_plot_pole_zero(self):
        """测试零极点图"""
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        poles = 0.8 * np.exp(1j * theta)
        zeros = 0.5 * np.exp(1j * (theta + np.pi / 10))

        save_path = os.path.join(self.test_dir, 'pole_zero.png')
        fig, ax = self.plotter.plot_pole_zero(
            poles, zeros=zeros, unit_circle=True,
            title='Test Pole-Zero', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertIsNotNone(fig)

    def test_plot_stability(self):
        """测试稳定性图"""
        fn = np.array([5.0, 12.0, 20.0])
        zeta = np.array([0.05, 0.03, -0.01])

        save_path = os.path.join(self.test_dir, 'stability.png')
        fig, ax = self.plotter.plot_stability(
            fn, zeta, title='Test Stability', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertIsNotNone(fig)

    def test_plot_frequency_comparison(self):
        """测试频率响应对比图"""
        omega = np.logspace(-1, 2, 100)
        H1 = np.zeros((len(omega), 2, 2), dtype=complex)
        H2 = np.zeros((len(omega), 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                H1[:, i, j] = 1.0 / (1 + 1j * omega / 10)
                H2[:, i, j] = 0.9 / (1 + 1j * omega / 12)

        save_path = os.path.join(self.test_dir, 'freq_comparison.png')
        fig, axes = self.plotter.plot_frequency_comparison(
            omega, H1, H2, label1='Model 1', label2='Model 2',
            title='Test Frequency Comparison', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertEqual(len(axes), 2)

    def test_plot_metrics_comparison(self):
        """测试模型性能对比柱状图"""
        model_names = ['Model 1', 'Model 2', 'Model 3']
        metrics_dict = {
            'RMSE': [0.01, 0.02, 0.015],
            'R2': [0.99, 0.95, 0.97]
        }

        save_path = os.path.join(self.test_dir, 'metrics_comparison.png')
        fig, ax = self.plotter.plot_metrics_comparison(
            model_names, metrics_dict, metric='RMSE',
            title='Test Metrics Comparison', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertIsNotNone(fig)

    def test_plot_coherence(self):
        """测试相干函数图"""
        freqs = np.linspace(0, 50, 100)
        coherence = np.zeros((100, 2, 2))
        for i in range(2):
            for j in range(2):
                coherence[:, i, j] = np.exp(-freqs / 20) * 0.9 + 0.05 * np.random.randn(100)
        coherence = np.clip(coherence, 0, 1)

        save_path = os.path.join(self.test_dir, 'coherence.png')
        fig, ax = self.plotter.plot_coherence(
            freqs, coherence, outputs=[0, 1], inputs=[0, 1],
            title='Test Coherence', save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        self.assertIsNotNone(fig)

    def test_no_save_path(self):
        """测试不保存文件，仅返回Figure对象"""
        fig, ax = self.plotter.plot_time_series(
            self.time, self.Y, labels=['y1', 'y2'],
            title='Test No Save'
        )

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)


if __name__ == '__main__':
    unittest.main()
