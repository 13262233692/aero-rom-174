"""
数据预处理模块单元测试
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
import pandas as pd
from aerorom.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """测试DataPreprocessor类"""

    def setUp(self):
        """测试前准备"""
        self.dt = 0.01
        self.n_samples = 1000
        self.time = np.arange(self.n_samples) * self.dt

        np.random.seed(42)
        U = np.zeros((self.n_samples, 2))
        U[:, 0] = np.sin(2 * np.pi * 2 * self.time) + 0.1 * np.random.randn(self.n_samples)
        U[:, 1] = 0.5 * np.cos(2 * np.pi * 3 * self.time) + 0.1 * np.random.randn(self.n_samples)

        Y = np.zeros((self.n_samples, 2))
        Y[:, 0] = 2 * U[:, 0] + 1.5 * U[:, 1] + 0.05 * np.random.randn(self.n_samples)
        Y[:, 1] = 1.0 * U[:, 0] + 3.0 * U[:, 1] + 0.05 * np.random.randn(self.n_samples)

        self.test_data = pd.DataFrame({
            'time': self.time,
            'u1': U[:, 0],
            'u2': U[:, 1],
            'y1': Y[:, 0],
            'y2': Y[:, 1]
        })

        self.preprocessor = DataPreprocessor(dt=self.dt)
        self.preprocessor.raw_data = self.test_data
        self.preprocessor.set_input_output(['u1', 'u2'], ['y1', 'y2'])

    def test_load_data(self):
        """测试数据加载"""
        temp_file = os.path.join(os.path.dirname(__file__), 'test_data.csv')
        self.test_data.to_csv(temp_file, index=False)

        preprocessor = DataPreprocessor()
        preprocessor.load_data(temp_file)
        self.assertEqual(len(preprocessor.raw_data), self.n_samples)
        self.assertAlmostEqual(preprocessor.dt, self.dt, places=6)

        os.remove(temp_file)

    def test_remove_outliers(self):
        """测试异常值移除"""
        data_with_outliers = self.test_data.copy()
        data_with_outliers.loc[100, 'y1'] = 100.0
        data_with_outliers.loc[200, 'y2'] = -100.0

        preprocessor = DataPreprocessor(dt=self.dt)
        preprocessor.raw_data = data_with_outliers
        preprocessor.set_input_output(['u1', 'u2'], ['y1', 'y2'])
        preprocessor.remove_outliers(method='zscore', threshold=3)

        self.assertLess(len(preprocessor.raw_data), self.n_samples)

    def test_interpolate_missing(self):
        """测试缺失值插值"""
        data_with_nan = self.test_data.copy()
        data_with_nan.loc[100:105, 'u1'] = np.nan
        data_with_nan.loc[200, 'y1'] = np.nan

        preprocessor = DataPreprocessor(dt=self.dt)
        preprocessor.raw_data = data_with_nan
        preprocessor.set_input_output(['u1', 'u2'], ['y1', 'y2'])
        preprocessor.interpolate_missing()

        self.assertFalse(preprocessor.raw_data.isnull().any().any())

    def test_denoise(self):
        """测试数据去噪"""
        original_std = self.preprocessor.raw_data['y1'].std()
        self.preprocessor.denoise(method='butterworth', order=2, cutoff=10.0)
        denoised_std = self.preprocessor.raw_data['y1'].std()
        self.assertLess(denoised_std, original_std)

    def test_resample(self):
        """测试重采样"""
        new_dt = 0.02
        self.preprocessor.resample(new_dt=new_dt)
        self.assertAlmostEqual(self.preprocessor.dt, new_dt, places=6)
        expected_len = int(self.n_samples * self.dt / new_dt)
        self.assertAlmostEqual(len(self.preprocessor.raw_data), expected_len, delta=2)

    def test_standardize(self):
        """测试标准化"""
        self.preprocessor.standardize(method='standard')
        U, Y = self.preprocessor.get_input_output()

        np.testing.assert_allclose(np.mean(U[:, 0]), 0.0, atol=1e-2)
        np.testing.assert_allclose(np.std(U[:, 0]), 1.0, atol=1e-2)
        np.testing.assert_allclose(np.mean(Y[:, 0]), 0.0, atol=1e-2)
        np.testing.assert_allclose(np.std(Y[:, 0]), 1.0, atol=1e-2)

    def test_inverse_transform(self):
        """测试反标准化"""
        original_U, original_Y = self.preprocessor.get_input_output()
        self.preprocessor.standardize(method='standard')

        U_scaled, Y_scaled = self.preprocessor.get_input_output()
        U_recovered = self.preprocessor.inverse_transform(U_scaled, data_type='input')
        Y_recovered = self.preprocessor.inverse_transform(Y_scaled, data_type='output')

        np.testing.assert_allclose(U_recovered, original_U, rtol=1e-5)
        np.testing.assert_allclose(Y_recovered, original_Y, rtol=1e-5)

    def test_split_data(self):
        """测试数据分割"""
        train_data, val_data = self.preprocessor.split_data(train_ratio=0.8)
        expected_train_len = int(self.n_samples * 0.8)

        self.assertEqual(len(train_data), expected_train_len)
        self.assertEqual(len(val_data), self.n_samples - expected_train_len)

    def test_get_input_output(self):
        """测试输入输出获取"""
        U, Y = self.preprocessor.get_input_output()
        self.assertEqual(U.shape, (self.n_samples, 2))
        self.assertEqual(Y.shape, (self.n_samples, 2))

    def test_build_hankel_matrices(self):
        """测试Hankel矩阵构建"""
        U, Y = self.preprocessor.get_input_output()
        block_rows = 20
        Hankel_U, Hankel_Y = self.preprocessor.build_hankel_matrices(U, Y, block_rows)

        expected_cols = self.n_samples - 2 * block_rows + 1
        self.assertEqual(Hankel_U.shape, (block_rows * 2, expected_cols))
        self.assertEqual(Hankel_Y.shape, (block_rows * 2, expected_cols))

    def test_build_arx_data(self):
        """测试ARX数据构建"""
        U, Y = self.preprocessor.get_input_output()
        input_order = 3
        output_order = 3
        Phi, Y_reg = self.preprocessor.build_arx_data(U, Y, input_order, output_order)

        max_order = max(input_order, output_order)
        expected_n_reg = self.n_samples - max_order
        n_params = output_order * 2 + input_order * 2

        self.assertEqual(Phi.shape, (expected_n_reg, n_params))
        self.assertEqual(Y_reg.shape, (expected_n_reg, 2))

    def test_get_time_vector(self):
        """测试时间向量获取"""
        time = self.preprocessor.get_time_vector()
        np.testing.assert_allclose(time, self.time)


if __name__ == '__main__':
    unittest.main()
