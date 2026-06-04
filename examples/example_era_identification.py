"""
示例: ERA算法辨识气动力降阶模型

本示例演示如何使用ERA（特征系统实现算法）从CFD或试验数据中
辨识气动力状态空间模型。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from aerorom import (
    DataPreprocessor,
    ERA,
    ModelValidator,
    ModelExporter,
    Plotter
)


def generate_aerodynamic_data(dt=0.01, n_samples=2000):
    """
    生成模拟的气动力数据（二元翼段沉浮/俯仰运动）

    状态变量:
        - h: 沉浮位移
        - alpha: 俯仰角

    输入:
        - 控制面偏转
        - 阵风扰动

    输出:
        - 升力系数 Cl
        - 力矩系数 Cm
    """
    time = np.arange(n_samples) * dt

    omega_h = 2 * np.pi * 8.0
    omega_a = 2 * np.pi * 15.0
    zeta_h = 0.02
    zeta_a = 0.015

    U = np.zeros((n_samples, 2))
    U[:, 0] = 0.05 * np.sin(2 * np.pi * 3 * time) + 0.02 * np.sin(2 * np.pi * 7 * time)
    U[:, 1] = 0.03 * np.sin(2 * np.pi * 5 * time) + 0.01 * np.random.randn(n_samples)

    omega_h_d = omega_h * np.sqrt(1 - zeta_h**2)
    omega_a_d = omega_a * np.sqrt(1 - zeta_a**2)

    Y = np.zeros((n_samples, 2))
    Y[0] = [0.1, 0.05]

    for k in range(n_samples - 1):
        h_k = Y[k, 0]
        a_k = Y[k, 1]

        d_h = -2 * zeta_h * omega_h * h_k + 20 * U[k, 0] + 10 * U[k, 1]
        d_a = -2 * zeta_a * omega_a * a_k + 15 * U[k, 0] + 25 * U[k, 1]

        Y[k+1, 0] = (np.exp(-zeta_h * omega_h * dt) *
                     (h_k * np.cos(omega_h_d * dt) +
                      (zeta_h * omega_h * h_k + d_h) / omega_h_d * np.sin(omega_h_d * dt)))
        Y[k+1, 1] = (np.exp(-zeta_a * omega_a * dt) *
                     (a_k * np.cos(omega_a_d * dt) +
                      (zeta_a * omega_a * a_k + d_a) / omega_a_d * np.sin(omega_a_d * dt)))

    Y += 0.005 * np.random.randn(*Y.shape)

    return time, U, Y


def main():
    print("=" * 70)
    print("示例: ERA算法辨识气动力降阶模型")
    print("=" * 70)

    dt = 0.01
    print("\n[1] 生成气动力仿真数据...")
    time, U, Y = generate_aerodynamic_data(dt=dt, n_samples=2000)
    print(f"  时间范围: {time[0]:.2f}s - {time[-1]:.2f}s")
    print(f"  输入维度: {U.shape[1]}, 输出维度: {Y.shape[1]}, 样本数: {len(time)}")

    print("\n[2] 数据预处理...")
    preprocessor = DataPreprocessor(dt=dt)
    preprocessor.set_input_output(['control', 'gust'], ['Cl', 'Cm'])
    preprocessor.raw_data = preprocessor.raw_data = type('obj', (object,), {
        'time': time,
        'control': U[:, 0],
        'gust': U[:, 1],
        'Cl': Y[:, 0],
        'Cm': Y[:, 1]
    })

    print("\n[3] 使用ERA辨识状态空间模型...")
    era = ERA(dt=dt)
    model = era.identify(Y, U, n_states=8, block_rows=50)
    model.name = 'Aerodynamic_ROM_ERA'
    model.summary()

    suggested_order = era.select_order(max_states=30, plot=False)
    print(f"\n  建议模型阶数: {suggested_order}")

    print("\n[4] 模型降阶 (8 → 4 状态)...")
    reduced_model = model.reduce_order(n_reduced=4)
    reduced_model.name = 'Aerodynamic_ROM_ERA_reduced'
    reduced_model.summary()

    print("\n[5] 模型验证...")
    validator = ModelValidator(reduced_model)

    Y_pred, errors = validator.one_step_prediction(U, Y)
    metrics = validator.compute_metrics(Y, Y_pred, names=['Cl', 'Cm'])

    print("\n  误差指标:")
    for name in ['Cl', 'Cm']:
        print(f"\n  {name}:")
        print(f"    RMSE: {metrics[name]['RMSE']:.6f}")
        print(f"    MAE:  {metrics[name]['MAE']:.6f}")
        print(f"    R²:   {metrics[name]['R2']:.6f}")
        print(f"    VAF:  {metrics[name]['VAF']:.6f}")

    stab = validator.stability_analysis()
    print(f"\n  稳定性: {'稳定' if stab['stable'] else '不稳定'}")
    print(f"  固有频率 (Hz): {stab['natural_frequencies']}")
    print(f"  阻尼比: {stab['damping_ratios']}")

    print("\n[6] 生成可视化...")
    plotter = Plotter()
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'era_example')
    os.makedirs(output_dir, exist_ok=True)

    plotter.plot_time_series(
        time, U, labels=['控制面偏转', '阵风扰动'],
        title='输入信号', ylabel='幅值',
        save_path=os.path.join(output_dir, 'inputs.png')
    )

    plotter.plot_time_series(
        time, Y, labels=['升力系数 Cl', '力矩系数 Cm'],
        title='输出信号 (气动力系数)', ylabel='系数值',
        save_path=os.path.join(output_dir, 'outputs.png')
    )

    plotter.plot_comparison(
        time, Y, Y_pred, labels=['Cl', 'Cm'],
        title='ERA模型预测对比',
        save_path=os.path.join(output_dir, 'prediction.png')
    )

    omega, H = reduced_model.frequency_response()
    plotter.plot_bode(
        omega, H,
        title='气动力降阶模型 Bode图',
        save_path=os.path.join(output_dir, 'bode.png')
    )

    plotter.plot_singular_values(
        era.singular_values, suggested_order=4,
        title='Hankel矩阵奇异值分布',
        save_path=os.path.join(output_dir, 'singular_values.png')
    )

    poles = reduced_model.eigenvalues()
    fn, zeta = reduced_model.natural_frequencies_damping()
    plotter.plot_pole_zero(
        poles,
        title='降阶模型极点分布',
        save_path=os.path.join(output_dir, 'poles.png')
    )

    plotter.plot_stability(
        fn, zeta,
        title='气动力模态频率-阻尼分布',
        save_path=os.path.join(output_dir, 'stability.png')
    )

    plotter.plot_residuals(
        time, errors, labels=['Cl', 'Cm'],
        title='预测残差分析',
        save_path=os.path.join(output_dir, 'residuals.png')
    )

    print(f"\n[7] 导出模型...")
    exporter = ModelExporter(reduced_model)
    export_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'era_example', 'models')
    os.makedirs(export_dir, exist_ok=True)

    exporter.export_pickle(os.path.join(export_dir, 'aerodynamic_rom.pkl'))
    exporter.export_json(os.path.join(export_dir, 'aerodynamic_rom.json'))
    exporter.export_python_module(os.path.join(export_dir, 'aerodynamic_rom.py'))
    exporter.export_matlab(os.path.join(export_dir, 'aerodynamic_rom.mat'))
    exporter.export_nastran_dmi(os.path.join(export_dir, 'aerodynamic_rom.bdf'))

    print(f"\n  图表和模型已保存到: {os.path.abspath(output_dir)}")

    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
