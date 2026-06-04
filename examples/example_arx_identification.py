"""
示例: ARX模型辨识气动力降阶模型

本示例演示如何使用ARX（带外生输入的自回归）模型从时域数据中
辨识气动力降阶模型。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from aerorom import (
    DataPreprocessor,
    ARX,
    ModelValidator,
    ModelExporter,
    Plotter
)


def generate_test_data(dt=0.005, n_samples=3000):
    """
    生成试验类型的气动力数据

    模拟扫频激励下的气动力响应
    """
    time = np.arange(n_samples) * dt

    def chirp_signal(t, f0, f1):
        """线性扫频信号"""
        k = (f1 - f0) / (t[-1] - t[0])
        return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))

    U = np.zeros((n_samples, 2))
    U[:, 0] = 0.1 * chirp_signal(time, 0.5, 30)
    U[:, 1] = 0.05 * np.sin(2 * np.pi * 12 * time) + 0.02 * np.random.randn(n_samples)

    Y = np.zeros((n_samples, 3))

    omega = [2 * np.pi * f for f in [6, 14, 22]]
    zeta = [0.03, 0.02, 0.015]
    gains = [[15, 8], [10, 20], [5, 12]]

    for k in range(n_samples - 1):
        for i in range(3):
            omega_d = omega[i] * np.sqrt(1 - zeta[i]**2)
            force = gains[i][0] * U[k, 0] + gains[i][1] * U[k, 1]
            damping = -2 * zeta[i] * omega[i] * Y[k, i]

            Y[k+1, i] = (np.exp(-zeta[i] * omega[i] * dt) *
                         (Y[k, i] * np.cos(omega_d * dt) +
                          (zeta[i] * omega[i] * Y[k, i] + damping + force) / omega_d *
                          np.sin(omega_d * dt)))

    Y += 0.002 * np.random.randn(*Y.shape)

    data = pd.DataFrame({
        'time': time,
        'u1': U[:, 0],
        'u2': U[:, 1],
        'y1': Y[:, 0],
        'y2': Y[:, 1],
        'y3': Y[:, 2]
    })

    return data


def main():
    print("=" * 70)
    print("示例: ARX模型辨识气动力降阶模型")
    print("=" * 70)

    dt = 0.005
    print("\n[1] 生成试验数据...")
    data = generate_test_data(dt=dt, n_samples=3000)
    print(f"  样本数: {len(data)}, 采样频率: {1/dt:.1f} Hz")
    print(f"  列名: {list(data.columns)}")

    print("\n[2] 数据预处理...")
    preprocessor = DataPreprocessor(dt=dt)
    preprocessor.raw_data = data
    preprocessor.set_input_output(['u1', 'u2'], ['y1', 'y2', 'y3'])
    preprocessor.remove_outliers(method='zscore', threshold=4)
    preprocessor.denoise(method='savgol', window_length=51, polyorder=3)
    preprocessor.standardize(method='standard')

    U_train, Y_train = preprocessor.get_input_output()
    time = preprocessor.get_time_vector()

    print("\n[3] 分割训练/验证集 (80/20)...")
    train_data, val_data = preprocessor.split_data(train_ratio=0.8, shuffle=False)
    U_train, Y_train = preprocessor.get_input_output(train_data)
    U_val, Y_val = preprocessor.get_input_output(val_data)
    time_train = preprocessor.get_time_vector(train_data)
    time_val = preprocessor.get_time_vector(val_data)
    print(f"  训练集: {len(U_train)} 样本")
    print(f"  验证集: {len(U_val)} 样本")

    print("\n[4] ARX模型辨识...")
    arx = ARX(dt=dt)

    orders_to_test = [(2, 2), (3, 3), (4, 4), (5, 5)]
    val_errors = []

    for input_order, output_order in orders_to_test:
        print(f"\n  测试阶数: 输入={input_order}, 输出={output_order}")
        model = arx.identify(U_train, Y_train, input_order, output_order)

        validator = ModelValidator(model)
        Y_pred, _ = validator.one_step_prediction(U_val, Y_val)
        metrics = validator.compute_metrics(Y_val, Y_pred)
        val_rmse = metrics['overall']['RMSE']
        val_errors.append(val_rmse)
        print(f"    验证集 RMSE: {val_rmse:.6f}")

    best_idx = np.argmin(val_errors)
    best_input_order, best_output_order = orders_to_test[best_idx]
    print(f"\n  最优阶数: 输入={best_input_order}, 输出={best_output_order}")

    model = arx.identify(U_train, Y_train, best_input_order, best_output_order)
    model.name = 'Aerodynamic_ROM_ARX'
    model.summary()

    print("\n[5] 模型验证...")
    validator = ModelValidator(model)

    print("\n  [5.1] 训练集验证...")
    Y_pred_train, errors_train = validator.one_step_prediction(U_train, Y_train)
    metrics_train = validator.compute_metrics(Y_train, Y_pred_train, names=['y1', 'y2', 'y3'])

    print("\n  [5.2] 验证集验证...")
    Y_pred_val, errors_val = validator.one_step_prediction(U_val, Y_val)
    metrics_val = validator.compute_metrics(Y_val, Y_pred_val, names=['y1', 'y2', 'y3'])

    print("\n  训练集 vs 验证集 对比:")
    print(f"  {'指标':<10} {'训练集':>12} {'验证集':>12}")
    print("-" * 40)
    for metric in ['RMSE', 'MAE', 'R2', 'VAF']:
        train_val = metrics_train['overall'][metric]
        val_val = metrics_val['overall'][metric]
        print(f"  {metric:<10} {train_val:>12.6f} {val_val:>12.6f}")

    stab = validator.stability_analysis()
    print(f"\n  模型稳定性: {'稳定' if stab['stable'] else '不稳定'}")
    print(f"  固有频率 (Hz): {stab['natural_frequencies']}")
    print(f"  阻尼比: {stab['damping_ratios']}")

    print("\n[6] 交叉验证 (5折)...")
    cv_results = validator.cross_validation(
        U_train, Y_train, n_folds=5, method='ARX',
        input_order=best_input_order, output_order=best_output_order
    )
    print(f"\n  交叉验证结果:")
    print(f"    平均 RMSE: {cv_results['mean_metrics']['RMSE']:.6f} ± {cv_results['std_metrics']['RMSE']:.6f}")
    print(f"    平均 R²:   {cv_results['mean_metrics']['R2']:.6f} ± {cv_results['std_metrics']['R2']:.6f}")

    print("\n[7] 生成可视化...")
    plotter = Plotter()
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'arx_example')
    os.makedirs(output_dir, exist_ok=True)

    plotter.plot_time_series(
        time_train, U_train, labels=['输入 1 (扫频)', '输入 2 (正弦)'],
        title='训练集输入信号',
        save_path=os.path.join(output_dir, 'train_inputs.png')
    )

    plotter.plot_comparison(
        time_val, Y_val, Y_pred_val, labels=['y1', 'y2', 'y3'],
        title='验证集预测对比',
        save_path=os.path.join(output_dir, 'validation_prediction.png')
    )

    omega, H = model.frequency_response()
    plotter.plot_bode(
        omega, H, outputs=[0, 1], inputs=[0, 1],
        title='ARX模型 Bode图',
        save_path=os.path.join(output_dir, 'bode.png')
    )

    plotter.plot_residuals(
        time_val, errors_val, labels=['y1', 'y2', 'y3'],
        title='验证集残差分析',
        save_path=os.path.join(output_dir, 'residuals.png')
    )

    poles = model.eigenvalues()
    fn, zeta = model.natural_frequencies_damping()
    plotter.plot_pole_zero(
        poles,
        title='ARX模型极点分布',
        save_path=os.path.join(output_dir, 'poles.png')
    )

    order_names = [f'nb={io},na={oo}' for io, oo in orders_to_test]
    plotter.plot_metrics_comparison(
        order_names, {'RMSE': val_errors}, metric='RMSE',
        title='不同阶数验证集误差',
        save_path=os.path.join(output_dir, 'order_selection.png')
    )

    print(f"\n  图表已保存到: {os.path.abspath(output_dir)}")

    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
