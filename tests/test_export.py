"""
模型导出模块单元测试
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
import tempfile
import shutil
from aerorom.identification import StateSpaceModel
from aerorom.export import ModelExporter


class TestModelExporter(unittest.TestCase):
    """测试ModelExporter类"""

    def setUp(self):
        """测试前准备"""
        self.dt = 0.01
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
        self.model.name = 'TestROM'

        self.exporter = ModelExporter(self.model)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_set_model(self):
        """测试设置模型"""
        new_model = StateSpaceModel(self.model.A, self.model.B, self.model.C, self.model.D, self.dt)
        self.exporter.set_model(new_model)
        self.assertEqual(self.exporter.model, new_model)

    def test_export_pickle(self):
        """测试导出Pickle格式"""
        filepath = os.path.join(self.test_dir, 'model.pkl')
        self.exporter.export_pickle(filepath)

        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)

        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        np.testing.assert_allclose(data['A'], self.model.A)
        np.testing.assert_allclose(data['B'], self.model.B)
        self.assertEqual(data['n_states'], self.n_states)

    def test_export_numpy(self):
        """测试导出NumPy格式"""
        filepath = os.path.join(self.test_dir, 'model')
        self.exporter.export_numpy(filepath)

        expected_path = filepath + '.npz'
        self.assertTrue(os.path.exists(expected_path))

        data = np.load(expected_path)
        np.testing.assert_allclose(data['A'], self.model.A)
        np.testing.assert_allclose(data['B'], self.model.B)

    def test_export_matlab(self):
        """测试导出MATLAB格式"""
        try:
            import scipy
        except ImportError:
            self.skipTest('scipy not installed')

        filepath = os.path.join(self.test_dir, 'model')
        self.exporter.export_matlab(filepath)

        expected_path = filepath + '.mat'
        self.assertTrue(os.path.exists(expected_path))

    def test_export_json(self):
        """测试导出JSON格式"""
        filepath = os.path.join(self.test_dir, 'model')
        self.exporter.export_json(filepath, include_matrices=True)

        expected_path = filepath + '.json'
        self.assertTrue(os.path.exists(expected_path))

        import json
        with open(expected_path, 'r') as f:
            data = json.load(f)

        self.assertEqual(data['name'], 'TestROM')
        self.assertEqual(data['n_states'], self.n_states)
        self.assertIn('A', data)

    def test_export_csv(self):
        """测试导出CSV格式"""
        directory = os.path.join(self.test_dir, 'csv_export')
        self.exporter.export_csv(directory)

        self.assertTrue(os.path.exists(directory))
        self.assertTrue(os.path.exists(os.path.join(directory, 'matrix_A.csv')))
        self.assertTrue(os.path.exists(os.path.join(directory, 'matrix_B.csv')))
        self.assertTrue(os.path.exists(os.path.join(directory, 'matrix_C.csv')))
        self.assertTrue(os.path.exists(os.path.join(directory, 'matrix_D.csv')))
        self.assertTrue(os.path.exists(os.path.join(directory, 'model_info.csv')))

    def test_export_simulink_sfunction(self):
        """测试导出Simulink S函数"""
        filepath = os.path.join(self.test_dir, 'rom_sfunction')
        self.exporter.export_simulink_sfunction(filepath)

        expected_path = filepath + '.m'
        self.assertTrue(os.path.exists(expected_path))

        with open(expected_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('function [sys,x0,str,ts]', content)
        self.assertIn('rom_sfunction', content)

    def test_export_fortran(self):
        """测试导出Fortran模块"""
        filepath = os.path.join(self.test_dir, 'rom_model')
        self.exporter.export_fortran(filepath)

        expected_path = filepath + '.f90'
        self.assertTrue(os.path.exists(expected_path))

        with open(expected_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('module', content)
        self.assertIn('subroutine rom_state_update', content)
        self.assertIn('subroutine rom_output', content)

    def test_export_python_module(self):
        """测试导出Python模块"""
        filepath = os.path.join(self.test_dir, 'rom_model')
        self.exporter.export_python_module(filepath)

        expected_path = filepath + '.py'
        self.assertTrue(os.path.exists(expected_path))

        with open(expected_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('def state_update(x, u):', content)
        self.assertIn('def output(x, u):', content)
        self.assertIn('def simulate(U, x0=None):', content)

        sys.path.insert(0, self.test_dir)
        import importlib
        spec = importlib.util.spec_from_file_location("rom_model", expected_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(module.n_states, self.n_states)
        np.testing.assert_allclose(module.A, self.model.A)

        U = np.random.randn(10, self.n_inputs)
        Y, X = module.simulate(U)
        self.assertEqual(Y.shape, (10, self.n_outputs))
        self.assertEqual(X.shape, (10, self.n_states))

    def test_export_nastran_dmi(self):
        """测试导出NASTRAN DMI格式"""
        filepath = os.path.join(self.test_dir, 'model')
        self.exporter.export_nastran_dmi(filepath)

        expected_path = filepath + '.bdf'
        self.assertTrue(os.path.exists(expected_path))

        with open(expected_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('DMI,A,1,REAL', content)
        self.assertIn('DMI,B,1,REAL', content)

    def test_export_all_formats(self):
        """测试批量导出多种格式"""
        model_dir = os.path.join(self.test_dir, 'all_formats')
        os.makedirs(model_dir)

        self.exporter.export_pickle(os.path.join(model_dir, 'model.pkl'))
        self.exporter.export_numpy(os.path.join(model_dir, 'model'))
        self.exporter.export_json(os.path.join(model_dir, 'model'))
        self.exporter.export_python_module(os.path.join(model_dir, 'rom_model'))
        self.exporter.export_csv(os.path.join(model_dir, 'csv'))

        self.assertEqual(len(os.listdir(model_dir)), 5)


if __name__ == '__main__':
    unittest.main()
