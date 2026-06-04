"""
颤振分析模块

基于辨识的气动力降阶模型（ROM）进行气动弹性颤振分析，
包括V-g图计算和颤振临界速度预测。

主要功能：
    - 气动弹性系统状态方程构建
    - V-g图计算（速度-阻尼曲线）
    - V-f图计算（速度-频率曲线）
    - 颤振临界速度检测
    - 根轨迹分析
"""

import numpy as np
import warnings
from typing import Union, Optional, Tuple, List


class FlutterAnalysis:
    """
    颤振分析类

    基于气动力ROM和结构动力学模型进行颤振分析
    """

    def __init__(self, rom_model=None, structural_model=None):
        """
        初始化颤振分析器

        参数:
            rom_model: 辨识得到的气动力ROM模型 (StateSpaceModel)
            structural_model: 结构动力学模型字典，包含：
                - 'M': 质量矩阵
                - 'K': 刚度矩阵
                - 'C': 阻尼矩阵 (可选)
                - 'natural_frequencies': 固有频率 (Hz，可选)
                - 'damping_ratios': 阻尼比 (可选)
                - 'mode_shapes': 振型矩阵 (可选)
        """
        self.rom_model = rom_model
        self.structural_model = structural_model
        self.results = {}

    def set_rom_model(self, rom_model):
        """
        设置气动力ROM模型

        参数:
            rom_model: 气动力ROM模型 (StateSpaceModel)
        """
        self.rom_model = rom_model
        return self

    def set_structural_model(self, M=None, K=None, C=None,
                             natural_frequencies=None, damping_ratios=None,
                             mode_shapes=None):
        """
        设置结构动力学模型

        参数:
            M: 质量矩阵 (n_dof, n_dof)
            K: 刚度矩阵 (n_dof, n_dof)
            C: 阻尼矩阵 (n_dof, n_dof)，可选
            natural_frequencies: 固有频率数组 (Hz)，可选
            damping_ratios: 阻尼比数组，可选
            mode_shapes: 振型矩阵 (n_dof, n_modes)，可选
        """
        self.structural_model = {
            'M': np.array(M) if M is not None else None,
            'K': np.array(K) if K is not None else None,
            'C': np.array(C) if C is not None else None,
            'natural_frequencies': np.array(natural_frequencies) if natural_frequencies is not None else None,
            'damping_ratios': np.array(damping_ratios) if damping_ratios is not None else None,
            'mode_shapes': np.array(mode_shapes) if mode_shapes is not None else None
        }
        return self

    def _build_aeroelastic_system(self, velocity, rho=1.225, b_ref=1.0):
        """
        构建气动弹性系统状态空间矩阵

        状态向量: x = [x_struct; x_aero]
        其中 x_struct = [η; η_dot] 为结构模态坐标和速度
             x_aero = 气动力ROM状态

        参数:
            velocity: 气流速度 (m/s)
            rho: 空气密度 (kg/m³)
            b_ref: 参考弦长 (m)

        返回:
            A_sys: 系统状态矩阵
            n_struct_states: 结构状态数量
            n_aero_states: 气动力状态数量
        """
        if self.structural_model is None:
            raise ValueError("请先设置结构动力学模型")
        if self.rom_model is None:
            raise ValueError("请先设置气动力ROM模型")

        M = self.structural_model['M']
        K = self.structural_model['K']
        C = self.structural_model.get('C')
        n_dof = M.shape[0]

        if C is None:
            fn = self.structural_model.get('natural_frequencies')
            zeta = self.structural_model.get('damping_ratios')
            if fn is not None and zeta is not None:
                omega_n = 2 * np.pi * fn
                C = np.zeros_like(M)
                for i in range(min(len(omega_n), n_dof)):
                    C[i, i] = 2 * zeta[i] * omega_n[i] * M[i, i]
            else:
                C = np.zeros_like(M)

        q_dyn = 0.5 * rho * velocity**2
        A_aero = self.rom_model.A
        B_aero = self.rom_model.B
        C_aero = self.rom_model.C
        D_aero = self.rom_model.D

        n_aero_states = self.rom_model.n_states
        n_struct_states = 2 * n_dof

        n_total = n_struct_states + n_aero_states
        A_sys = np.zeros((n_total, n_total))

        M_inv = np.linalg.inv(M)

        A_sys[:n_dof, n_dof:2*n_dof] = np.eye(n_dof)

        A_sys[n_dof:2*n_dof, :n_dof] = -M_inv @ K
        A_sys[n_dof:2*n_dof, n_dof:2*n_dof] = -M_inv @ C

        if n_aero_states > 0:
            A_sys[n_dof:2*n_dof, 2*n_dof:] = q_dyn * M_inv @ C_aero
            A_sys[2*n_dof:, n_dof:2*n_dof] = velocity / b_ref * B_aero
            A_sys[2*n_dof:, 2*n_dof:] = A_aero

        return A_sys, n_struct_states, n_aero_states

    def compute_flutter_velocity(self, velocities=None, v_min=10.0, v_max=200.0,
                                 n_points=50, rho=1.225, b_ref=1.0,
                                 method='p-k'):
        """
        计算V-g图和V-f图，检测颤振临界速度

        参数:
            velocities: 速度点数组 (m/s)，如为None则自动生成
            v_min: 最小速度 (m/s)
            v_max: 最大速度 (m/s)
            n_points: 速度点数量
            rho: 空气密度 (kg/m³)
            b_ref: 参考弦长 (m)
            method: 分析方法 ('p-k' 或 'state_space')

        返回:
            results: 颤振分析结果字典
        """
        if velocities is None:
            velocities = np.linspace(v_min, v_max, n_points)
        else:
            velocities = np.atleast_1d(velocities)

        n_vel = len(velocities)
        freqs_list = []
        dampings_list = []
        eigenvalues_list = []

        print("\n" + "=" * 70)
        print("颤振速度计算")
        print("=" * 70)
        print(f"速度范围: {v_min:.1f} - {v_max:.1f} m/s")
        print(f"速度点数: {n_vel}")
        print(f"空气密度: {rho:.3f} kg/m^3")
        print(f"参考弦长: {b_ref:.3f} m")
        print(f"分析方法: {method}")
        print("-" * 70)

        for idx, V in enumerate(velocities):
            try:
                if method == 'state_space':
                    A_sys, n_struct, n_aero = self._build_aeroelastic_system(V, rho, b_ref)
                    eigvals = np.linalg.eigvals(A_sys)
                else:
                    A_sys, n_struct, n_aero = self._build_aeroelastic_system(V, rho, b_ref)
                    eigvals = np.linalg.eigvals(A_sys)

                dt = self.rom_model.dt if self.rom_model.dt else 0.01
                sigma = np.real(eigvals)
                omega = np.abs(np.imag(eigvals))

                valid_mask = omega > 1e-10
                if np.any(valid_mask):
                    freqs = omega[valid_mask] / (2 * np.pi)
                    zeta = -sigma[valid_mask] / np.sqrt(sigma[valid_mask]**2 + omega[valid_mask]**2)
                else:
                    freqs = np.array([])
                    zeta = np.array([])

                sort_idx = np.argsort(freqs)
                freqs = freqs[sort_idx]
                zeta = zeta[sort_idx]
                eigvals_sorted = eigvals[valid_mask][sort_idx]

                freqs_list.append(freqs)
                dampings_list.append(zeta)
                eigenvalues_list.append(eigvals_sorted)

                if idx % 10 == 0 or idx == n_vel - 1:
                    if len(zeta) > 0:
                        min_damp = np.min(zeta)
                        max_damp = np.max(zeta)
                        print(f"  V = {V:6.1f} m/s, 阻尼范围: [{min_damp:+.4f}, {max_damp:+.4f}]")

            except Exception as e:
                warnings.warn(f"速度 {V:.1f} m/s 计算失败: {e}")
                freqs_list.append(np.array([]))
                dampings_list.append(np.array([]))
                eigenvalues_list.append(np.array([]))

        n_modes_max = max(len(f) for f in freqs_list)

        freqs_array = np.full((n_vel, n_modes_max), np.nan)
        dampings_array = np.full((n_vel, n_modes_max), np.nan)

        for i in range(n_vel):
            n_modes = len(freqs_list[i])
            if n_modes > 0:
                freqs_array[i, :n_modes] = freqs_list[i]
                dampings_array[i, :n_modes] = dampings_list[i]

        flutter_speed, flutter_info = self._detect_flutter_speed(
            velocities, dampings_array, freqs_array
        )

        results = {
            'velocities': velocities,
            'frequencies': freqs_array,
            'dampings': dampings_array,
            'eigenvalues': eigenvalues_list,
            'flutter_speed': flutter_speed,
            'flutter_info': flutter_info,
            'rho': rho,
            'b_ref': b_ref
        }

        self.results = results

        print("-" * 70)
        if flutter_speed is not None:
            print(f"\n✓ 检测到颤振临界速度: {flutter_speed:.2f} m/s")
            print(f"  对应频率: {flutter_info.get('flutter_frequency', 'N/A'):.2f} Hz")
            print(f"  颤振模态: 第 {flutter_info.get('flutter_mode', 'N/A')} 阶")
        else:
            print(f"\n✓ 在分析速度范围内 ({v_min:.1f} - {v_max:.1f} m/s) 未检测到颤振")
            print("  结构在该速度范围内是稳定的")

        print("=" * 70 + "\n")

        return results

    def _detect_flutter_speed(self, velocities, dampings, frequencies):
        """
        检测颤振临界速度

        颤振判据：阻尼由正变负（穿越零点）

        参数:
            velocities: 速度数组
            dampings: 阻尼数组 (n_vel, n_modes)
            frequencies: 频率数组 (n_vel, n_modes)

        返回:
            flutter_speed: 颤振临界速度 (m/s)，如未检测到则为None
            flutter_info: 颤振详细信息字典
        """
        n_vel, n_modes = dampings.shape
        flutter_speed = None
        flutter_info = {}

        for mode in range(n_modes):
            damp_mode = dampings[:, mode]

            valid_idx = ~np.isnan(damp_mode)
            if np.sum(valid_idx) < 2:
                continue

            vel_valid = velocities[valid_idx]
            damp_valid = damp_mode[valid_idx]
            freq_valid = frequencies[valid_idx, mode]

            sign_changes = np.where(np.diff(np.signbit(damp_valid)))[0]

            for idx in sign_changes:
                if damp_valid[idx] > 0 and damp_valid[idx + 1] <= 0:
                    v1, v2 = vel_valid[idx], vel_valid[idx + 1]
                    d1, d2 = damp_valid[idx], damp_valid[idx + 1]

                    v_flutter = v1 - d1 * (v2 - v1) / (d2 - d1)

                    if flutter_speed is None or v_flutter < flutter_speed:
                        flutter_speed = float(v_flutter)

                        f1, f2 = freq_valid[idx], freq_valid[idx + 1]
                        f_flutter = f1 + (f2 - f1) * (v_flutter - v1) / (v2 - v1)

                        flutter_info = {
                            'flutter_speed': flutter_speed,
                            'flutter_frequency': float(f_flutter),
                            'flutter_mode': mode + 1,
                            'crossing_velocity_range': [float(v1), float(v2)],
                            'damping_crossing': [float(d1), float(d2)],
                            'mode_damping_history': damp_mode.copy()
                        }

        return flutter_speed, flutter_info

    def compute_root_locus(self, velocities=None, v_min=10.0, v_max=200.0,
                           n_points=50, rho=1.225, b_ref=1.0):
        """
        计算根轨迹（s平面极点随速度变化）

        参数:
            velocities: 速度点数组
            v_min: 最小速度
            v_max: 最大速度
            n_points: 速度点数量
            rho: 空气密度
            b_ref: 参考弦长

        返回:
            root_locus: 根轨迹结果字典
        """
        if velocities is None:
            velocities = np.linspace(v_min, v_max, n_points)

        roots = []

        for V in velocities:
            try:
                A_sys, _, _ = self._build_aeroelastic_system(V, rho, b_ref)
                eigvals = np.linalg.eigvals(A_sys)
                roots.append(eigvals)
            except Exception as e:
                warnings.warn(f"速度 {V:.1f} m/s 根轨迹计算失败: {e}")
                roots.append(np.array([]))

        root_locus = {
            'velocities': velocities,
            'roots': roots,
            'rho': rho,
            'b_ref': b_ref
        }

        self.results['root_locus'] = root_locus
        return root_locus

    def compute_damping_derivatives(self, velocities=None, v_min=10.0, v_max=200.0,
                                     n_points=50, rho=1.225, b_ref=1.0):
        """
        计算各阶模态的阻尼随速度变化的导数

        参数:
            velocities: 速度点数组
            v_min: 最小速度
            v_max: 最大速度
            n_points: 速度点数量
            rho: 空气密度
            b_ref: 参考弦长

        返回:
            damping_derivatives: 阻尼导数结果
        """
        if 'dampings' not in self.results or 'velocities' not in self.results:
            _ = self.compute_flutter_velocity(velocities, v_min, v_max,
                                               n_points, rho, b_ref)

        dampings = self.results['dampings']
        velocities = self.results['velocities']
        n_vel, n_modes = dampings.shape

        damping_derivatives = np.full((n_vel, n_modes), np.nan)

        for mode in range(n_modes):
            damp_mode = dampings[:, mode]
            valid_idx = ~np.isnan(damp_mode)

            if np.sum(valid_idx) >= 3:
                vel_valid = velocities[valid_idx]
                damp_valid = damp_mode[valid_idx]

                for i in range(len(vel_valid)):
                    if i == 0:
                        deriv = (damp_valid[i + 1] - damp_valid[i]) / (vel_valid[i + 1] - vel_valid[i])
                    elif i == len(vel_valid) - 1:
                        deriv = (damp_valid[i] - damp_valid[i - 1]) / (vel_valid[i] - vel_valid[i - 1])
                    else:
                        deriv = 0.5 * (
                            (damp_valid[i + 1] - damp_valid[i]) / (vel_valid[i + 1] - vel_valid[i]) +
                            (damp_valid[i] - damp_valid[i - 1]) / (vel_valid[i] - vel_valid[i - 1])
                        )

                    original_idx = np.where(valid_idx)[0][i]
                    damping_derivatives[original_idx, mode] = deriv

        self.results['damping_derivatives'] = damping_derivatives
        return damping_derivatives

    def get_stability_margin(self, velocity):
        """
        计算指定速度下的稳定裕度

        参数:
            velocity: 飞行速度 (m/s)

        返回:
            margin: 稳定裕度信息字典
        """
        if 'flutter_speed' not in self.results or self.results['flutter_speed'] is None:
            return {'stabile': True, 'margin': np.inf}

        flutter_speed = self.results['flutter_speed']

        if velocity < flutter_speed:
            margin = (flutter_speed - velocity) / velocity * 100
            stable = True
        else:
            margin = (velocity - flutter_speed) / flutter_speed * 100
            stable = False

        return {
            'stable': stable,
            'velocity': velocity,
            'flutter_speed': flutter_speed,
            'margin_percent': margin,
            'margin_velocity': flutter_speed - velocity
        }

    def print_summary(self):
        """
        打印颤振分析摘要
        """
        print("\n" + "=" * 70)
        print("颤振分析摘要")
        print("=" * 70)

        if 'flutter_speed' in self.results:
            Vf = self.results['flutter_speed']
            if Vf is not None:
                info = self.results['flutter_info']
                print(f"\n✗ 检测到颤振:")
                print(f"  颤振临界速度: {Vf:.2f} m/s ({Vf*3.6:.2f} km/h)")
                print(f"  颤振频率: {info.get('flutter_frequency', 'N/A'):.2f} Hz")
                print(f"  颤振模态: 第 {info.get('flutter_mode', 'N/A')} 阶")

                v_range = info.get('crossing_velocity_range', [])
                if len(v_range) == 2:
                    print(f"  穿越速度范围: {v_range[0]:.1f} - {v_range[1]:.1f} m/s")
            else:
                print(f"\n✓ 结构在分析速度范围内稳定")
                if 'velocities' in self.results:
                    vels = self.results['velocities']
                    print(f"  分析速度范围: {vels[0]:.1f} - {vels[-1]:.1f} m/s")

            if 'dampings' in self.results and 'velocities' in self.results:
                dampings = self.results['dampings']
                velocities = self.results['velocities']

                if not np.all(np.isnan(dampings)):
                    min_damp = np.nanmin(dampings)
                    min_vel = velocities[np.nanargmin(np.nanmin(dampings, axis=1))]
                    print(f"\n  最小阻尼: {min_damp:+.4f} (在 V = {min_vel:.1f} m/s)")

        print("\n" + "=" * 70 + "\n")


def create_standard_section_model(n_modes=2, f1=5.0, f2=15.0, zeta1=0.01, zeta2=0.01):
    """
    创建标准的二元翼段结构模型

    参数:
        n_modes: 模态数量 (2 表示沉浮+俯仰)
        f1: 第一阶模态频率 (Hz)，沉浮
        f2: 第二阶模态频率 (Hz)，俯仰
        zeta1: 第一阶模态阻尼比
        zeta2: 第二阶模态阻尼比

    返回:
        structural_model: 结构模型字典
    """
    omega1 = 2 * np.pi * f1
    omega2 = 2 * np.pi * f2

    M = np.diag([1.0, 1.0])
    K = np.diag([omega1**2, omega2**2])
    C = np.diag([2 * zeta1 * omega1, 2 * zeta2 * omega2])

    return {
        'M': M,
        'K': K,
        'C': C,
        'natural_frequencies': np.array([f1, f2]),
        'damping_ratios': np.array([zeta1, zeta2])
    }


def create_flutter_demo_rom(dt=0.01, n_states=6, flutter_mode='bending_torsion'):
    """
    创建演示用气动力ROM模型

    参数:
        dt: 采样时间
        n_states: 状态数
        flutter_mode: 颤振类型 ('bending_torsion', 'control_surface')

    返回:
        rom_model: 气动力ROM状态空间模型
    """
    from aerorom import StateSpaceModel

    np.random.seed(42)

    if flutter_mode == 'bending_torsion':
        A = np.zeros((n_states, n_states))
        omega_b = 2 * np.pi * 5
        omega_t = 2 * np.pi * 15
        coupling = 0.1

        for i in range(0, n_states, 2):
            if i == 0:
                A[i, i] = -0.05 * omega_b
                A[i, i + 1] = omega_b
                A[i + 1, i] = -omega_b - coupling * omega_t
                A[i + 1, i + 1] = -0.05 * omega_b
            elif i == 2:
                A[i, i] = -0.03 * omega_t
                A[i, i + 1] = omega_t
                A[i + 1, i] = -omega_t - coupling * omega_b
                A[i + 1, i + 1] = -0.03 * omega_t
            else:
                decay = 0.1 * (i + 1)
                freq = 2 * np.pi * (10 + i * 5)
                A[i, i] = -decay
                A[i, i + 1] = freq
                A[i + 1, i] = -freq
                A[i + 1, i + 1] = -decay

        max_eig = np.max(np.abs(np.linalg.eigvals(A)))
        A = A / max_eig * 0.95

        B = np.random.randn(n_states, 2)
        C = np.random.randn(2, n_states)
        D = np.zeros((2, 2))

    else:
        A = np.random.randn(n_states, n_states)
        max_eig = np.max(np.abs(np.linalg.eigvals(A)))
        A = A / max_eig * 0.9
        B = np.random.randn(n_states, 3)
        C = np.random.randn(3, n_states)
        D = np.zeros((3, 3))

    return StateSpaceModel(A, B, C, D, dt)
