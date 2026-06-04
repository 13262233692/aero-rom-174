"""
颤振分析单元测试
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
import tempfile
import warnings
warnings.filterwarnings('ignore')

from aerorom import (
    FlutterAnalysis,
    create_standard_section_model,
    create_flutter_demo_rom,
    Plotter,
    StateSpaceModel
)

import matplotlib
matplotlib.use('Agg')


class TestFlutterAnalysis(unittest.TestCase):
    """颤振分析类测试"""

    def setUp(self):
        """测试初始化"""
        self.dt = 0.01
        self.structural_model = create_standard_section_model()
        self.rom_model = create_flutter_demo_rom(dt=self.dt, n_states=6)
        self.flutter = FlutterAnalysis(
            rom_model=self.rom_model,
            structural_model=self.structural_model
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.flutter.rom_model)
        self.assertIsNotNone(self.flutter.structural_model)
        self.assertIsInstance(self.flutter.rom_model, StateSpaceModel)
        self.assertIn('M', self.flutter.structural_model)
        self.assertIn('K', self.flutter.structural_model)

    def test_set_models(self):
        """测试设置模型"""
        flutter = FlutterAnalysis()
        flutter.set_rom_model(self.rom_model)
        flutter.set_structural_model(
            M=np.eye(2),
            K=np.diag([100, 1000]),
            C=np.diag([1, 2])
        )
        self.assertIsNotNone(flutter.rom_model)
        self.assertIsNotNone(flutter.structural_model)
        self.assertEqual(flutter.structural_model['M'].shape, (2, 2))

    def test_build_aeroelastic_system(self):
        """测试构建气动弹性系统"""
        A_sys, n_struct, n_aero = self.flutter._build_aeroelastic_system(
            velocity=50.0,
            rho=1.225,
            b_ref=1.0
        )

        expected_size = n_struct + n_aero
        self.assertEqual(A_sys.shape, (expected_size, expected_size))
        self.assertEqual(n_struct, 4)
        self.assertEqual(n_aero, 6)

    def test_compute_flutter_velocity(self):
        """测试计算颤振速度"""
        results = self.flutter.compute_flutter_velocity(
            v_min=20.0,
            v_max=100.0,
            n_points=15,
            rho=1.225,
            b_ref=1.0
        )

        self.assertIn('velocities', results)
        self.assertIn('dampings', results)
        self.assertIn('frequencies', results)
        self.assertIn('flutter_speed', results)

        self.assertEqual(len(results['velocities']), 15)
        n_vel, n_modes = results['dampings'].shape
        self.assertEqual(n_vel, 15)
        self.assertGreater(n_modes, 0)

        self.assertEqual(results['dampings'].shape, results['frequencies'].shape)

    def test_flutter_detection(self):
        """测试颤振检测逻辑"""
        velocities = np.array([10, 20, 30, 40, 50])
        dampings = np.array([
            [0.05, 0.02],
            [0.04, 0.015],
            [0.03, 0.005],
            [0.02, -0.01],
            [0.01, -0.02]
        ])
        frequencies = np.array([
            [5.0, 15.0],
            [4.9, 14.8],
            [4.8, 14.5],
            [4.7, 14.2],
            [4.6, 14.0]
        ])

        Vf, info = self.flutter._detect_flutter_speed(
            velocities, dampings, frequencies
        )

        self.assertIsNotNone(Vf)
        self.assertGreater(Vf, 30)
        self.assertLess(Vf, 40)
        self.assertEqual(info.get('flutter_mode'), 2)

    def test_compute_root_locus(self):
        """测试计算根轨迹"""
        results = self.flutter.compute_root_locus(
            v_min=20.0,
            v_max=100.0,
            n_points=10
        )

        self.assertIn('velocities', results)
        self.assertIn('roots', results)
        self.assertEqual(len(results['roots']), 10)
        self.assertGreater(len(results['roots'][0]), 0)

    def test_compute_damping_derivatives(self):
        """测试计算阻尼导数"""
        _ = self.flutter.compute_flutter_velocity(
            v_min=20.0,
            v_max=100.0,
            n_points=20
        )

        derivs = self.flutter.compute_damping_derivatives()

        self.assertEqual(derivs.shape[0], 20)
        self.assertGreater(derivs.shape[1], 0)

    def test_get_stability_margin(self):
        """测试获取稳定裕度"""
        self.flutter.results['flutter_speed'] = 100.0

        margin1 = self.flutter.get_stability_margin(50.0)
        self.assertTrue(margin1['stable'])
        self.assertAlmostEqual(margin1['margin_percent'], 100.0)

        margin2 = self.flutter.get_stability_margin(150.0)
        self.assertFalse(margin2['stable'])

    def test_print_summary(self):
        """测试打印摘要"""
        _ = self.flutter.compute_flutter_velocity(
            v_min=20.0, v_max=80.0, n_points=10
        )

        try:
            self.flutter.print_summary()
        except Exception as e:
            self.fail(f"print_summary() raised {e}")

    def test_no_flutter_case(self):
        """测试无颤振情况"""
        velocities = np.array([10, 20, 30])
        dampings = np.array([
            [0.05, 0.02],
            [0.04, 0.015],
            [0.03, 0.01]
        ])
        frequencies = np.array([
            [5.0, 15.0],
            [4.9, 14.8],
            [4.8, 14.6]
        ])

        Vf, info = self.flutter._detect_flutter_speed(
            velocities, dampings, frequencies
        )

        self.assertIsNone(Vf)
        self.assertEqual(len(info), 0)


class TestHelperFunctions(unittest.TestCase):
    """辅助函数测试"""

    def test_create_standard_section_model(self):
        """测试创建标准二元翼段模型"""
        model = create_standard_section_model(
            n_modes=2,
            f1=5.0,
            f2=15.0,
            zeta1=0.01,
            zeta2=0.01
        )

        self.assertIn('M', model)
        self.assertIn('K', model)
        self.assertIn('C', model)
        self.assertIn('natural_frequencies', model)
        self.assertIn('damping_ratios', model)

        self.assertEqual(model['M'].shape, (2, 2))
        self.assertEqual(model['K'].shape, (2, 2))
        self.assertEqual(model['C'].shape, (2, 2))
        np.testing.assert_allclose(model['natural_frequencies'], [5.0, 15.0])
        np.testing.assert_allclose(model['damping_ratios'], [0.01, 0.01])

    def test_create_flutter_demo_rom(self):
        """测试创建演示用气动力ROM"""
        dt = 0.01
        rom = create_flutter_demo_rom(
            dt=dt,
            n_states=6,
            flutter_mode='bending_torsion'
        )

        self.assertIsInstance(rom, StateSpaceModel)
        self.assertEqual(rom.n_states, 6)
        self.assertEqual(rom.n_inputs, 2)
        self.assertEqual(rom.n_outputs, 2)
        self.assertEqual(rom.dt, dt)

        eigvals = np.linalg.eigvals(rom.A)
        self.assertTrue(np.all(np.abs(eigvals) <= 0.95 + 1e-10))

    def test_create_flutter_demo_rom_control_surface(self):
        """测试创建操纵面颤振ROM"""
        rom = create_flutter_demo_rom(
            dt=0.01,
            n_states=8,
            flutter_mode='control_surface'
        )

        self.assertEqual(rom.n_states, 8)
        self.assertEqual(rom.n_inputs, 3)
        self.assertEqual(rom.n_outputs, 3)


class TestFlutterVisualization(unittest.TestCase):
    """颤振可视化测试"""

    def setUp(self):
        """测试初始化"""
        self.plotter = Plotter(style='default', figsize=(8, 6), dpi=72)
        self.dt = 0.01
        self.structural_model = create_standard_section_model()
        self.rom_model = create_flutter_demo_rom(dt=self.dt, n_states=6)
        self.flutter = FlutterAnalysis(
            rom_model=self.rom_model,
            structural_model=self.structural_model
        )

    def test_plot_vg(self):
        """测试绘制V-g图"""
        results = self.flutter.compute_flutter_velocity(
            v_min=20.0, v_max=80.0, n_points=10
        )

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig, axes = self.plotter.plot_vg(
                results['velocities'],
                results['dampings'],
                results['frequencies'],
                flutter_speed=results['flutter_speed'],
                save_path=save_path
            )

            self.assertIsNotNone(fig)
            self.assertEqual(len(axes), 2)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_plot_vg_no_frequency(self):
        """测试绘制不带频率的V-g图"""
        results = self.flutter.compute_flutter_velocity(
            v_min=20.0, v_max=80.0, n_points=10
        )

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig, axes = self.plotter.plot_vg(
                results['velocities'],
                results['dampings'],
                frequencies=None,
                flutter_speed=results['flutter_speed'],
                save_path=save_path
            )

            self.assertIsNotNone(fig)
            self.assertEqual(len(axes), 1)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_plot_root_locus(self):
        """测试绘制根轨迹图"""
        root_locus = self.flutter.compute_root_locus(
            v_min=20.0, v_max=80.0, n_points=10
        )

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig, ax = self.plotter.plot_root_locus(
                root_locus['roots'],
                velocities=root_locus['velocities'],
                flutter_speed=None,
                save_path=save_path
            )

            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_plot_damping_derivatives(self):
        """测试绘制阻尼导数曲线"""
        results = self.flutter.compute_flutter_velocity(
            v_min=20.0, v_max=80.0, n_points=15
        )
        derivs = self.flutter.compute_damping_derivatives()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig, ax = self.plotter.plot_damping_derivatives(
                results['velocities'],
                derivs,
                save_path=save_path
            )

            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_plot_flutter_summary(self):
        """测试绘制颤振分析汇总图"""
        results = self.flutter.compute_flutter_velocity(
            v_min=20.0, v_max=80.0, n_points=10
        )
        root_locus = self.flutter.compute_root_locus(
            v_min=20.0, v_max=80.0, n_points=8
        )
        results['root_locus'] = root_locus

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig, axes = self.plotter.plot_flutter_summary(
                results,
                save_path=save_path
            )

            self.assertIsNotNone(fig)
            self.assertEqual(len(axes), 3)
            self.assertTrue(os.path.exists(save_path))
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


class TestFlutterEdgeCases(unittest.TestCase):
    """边缘情况测试"""

    def test_missing_model_error(self):
        """测试缺少模型时的错误"""
        flutter = FlutterAnalysis()

        with self.assertRaises(ValueError):
            flutter._build_aeroelastic_system(50.0)

    def test_missing_rom_error(self):
        """测试缺少ROM时的错误"""
        flutter = FlutterAnalysis(
            structural_model=create_standard_section_model()
        )

        with self.assertRaises(ValueError):
            flutter._build_aeroelastic_system(50.0)

    def test_missing_structural_error(self):
        """测试缺少结构模型时的错误"""
        flutter = FlutterAnalysis(
            rom_model=create_flutter_demo_rom(dt=0.01)
        )

        with self.assertRaises(ValueError):
            flutter._build_aeroelastic_system(50.0)

    def test_damping_derivatives_without_results(self):
        """测试无结果时计算阻尼导数"""
        flutter = FlutterAnalysis(
            rom_model=create_flutter_demo_rom(dt=0.01),
            structural_model=create_standard_section_model()
        )

        derivs = flutter.compute_damping_derivatives(
            v_min=20.0, v_max=60.0, n_points=10
        )

        self.assertIsNotNone(derivs)
        self.assertEqual(derivs.shape[0], 10)

    def test_stability_margin_no_flutter(self):
        """测试无颤振时的稳定裕度"""
        flutter = FlutterAnalysis()
        flutter.results['flutter_speed'] = None

        margin = flutter.get_stability_margin(50.0)
        self.assertTrue(margin['stabile'])
        self.assertEqual(margin['margin'], np.inf)

    def test_antialiasing_filter_cutoff_error(self):
        """测试抗混叠滤波截止频率错误"""
        from aerorom import DataPreprocessor

        preprocessor = DataPreprocessor(dt=0.01)
        preprocessor.raw_data = type('obj', (object,), {'columns': ['input', 'output']})()
        preprocessor.input_labels = ['input']
        preprocessor.output_labels = ['output']

        with self.assertRaises(ValueError):
            preprocessor.antialiasing_filter(cutoff_freq=100.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
