"""
AeroROM 主程序入口

提供命令行界面，支持完整的气动力降阶模型辨识流程。
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from aerorom import (
    DataPreprocessor,
    ERA,
    ARX,
    ModelValidator,
    ModelExporter,
    Plotter
)


def generate_demo_data(dt=0.01, n_samples=1000):
    """
    生成演示用的气动力仿真数据

    参数:
        dt: 采样时间
        n_samples: 样本数

    返回:
        (U, Y, time): 输入、输出和时间向量
    """
    time = np.arange(n_samples) * dt

    omega1 = 2 * np.pi * 5.0
    omega2 = 2 * np.pi * 12.0
    zeta1 = 0.05
    zeta2 = 0.03

    U = np.zeros((n_samples, 2))
    U[:, 0] = 0.1 * np.sin(2 * np.pi * 2 * time) + 0.05 * np.random.randn(n_samples)
    U[:, 1] = 0.08 * np.sin(2 * np.pi * 3 * time) + 0.04 * np.random.randn(n_samples)

    Y = np.zeros((n_samples, 2))
    for i in range(1, n_samples):
        Y[i, 0] = (np.exp(-zeta1 * omega1 * dt) * (Y[i-1, 0] * np.cos(omega1 * np.sqrt(1-zeta1**2) * dt)
                   + (Y[i-1, 0] * zeta1 * omega1 + 10 * U[i-1, 0] + 5 * U[i-1, 1]) / (omega1 * np.sqrt(1-zeta1**2))
                   * np.sin(omega1 * np.sqrt(1-zeta1**2) * dt)))
        Y[i, 1] = (np.exp(-zeta2 * omega2 * dt) * (Y[i-1, 1] * np.cos(omega2 * np.sqrt(1-zeta2**2) * dt)
                   + (Y[i-1, 1] * zeta2 * omega2 + 8 * U[i-1, 0] + 12 * U[i-1, 1]) / (omega2 * np.sqrt(1-zeta2**2))
                   * np.sin(omega2 * np.sqrt(1-zeta2**2) * dt)))

    Y += 0.01 * np.random.randn(*Y.shape)

    return U, Y, time


def run_full_pipeline(args):
    """
    运行完整的模型辨识流程

    参数:
        args: 命令行参数
    """
    print("=" * 70)
    print("AeroROM - 气动力降阶模型辨识工具")
    print("=" * 70)

    if args.demo:
        print("\n[1/6] 生成演示数据...")
        U, Y, time = generate_demo_data(dt=args.dt, n_samples=args.n_samples)
        print(f"  输入维度: {U.shape[1]}, 输出维度: {Y.shape[1]}, 样本数: {len(time)}")
    else:
        print(f"\n[1/6] 加载数据: {args.input_file}")
        preprocessor = DataPreprocessor(dt=args.dt)
        preprocessor.load_data(args.input_file)
        preprocessor.set_input_output(args.input_cols, args.output_cols)
        preprocessor.summary()

        if args.denoise:
            print("\n[2/6] 数据去噪...")
            preprocessor.denoise(method=args.denoise_method, cutoff=args.cutoff_freq)

        if args.standardize:
            print("\n[3/6] 数据标准化...")
            preprocessor.standardize(method=args.standardize_method)

        U, Y = preprocessor.get_input_output()
        time = preprocessor.get_time_vector()
        print(f"  输入维度: {U.shape[1]}, 输出维度: {Y.shape[1]}, 样本数: {len(time)}")

    if args.noise_level > 0:
        print(f"\n  添加 {args.noise_level}% 噪声...")
        Y += args.noise_level / 100.0 * np.std(Y, axis=0) * np.random.randn(*Y.shape)

    print(f"\n[4/6] 辨识模型 ({args.method})...")
    if args.method == 'ERA':
        identifier = ERA(dt=args.dt)
        model = identifier.identify(Y, U, n_states=args.n_states, block_rows=args.block_rows)
    elif args.method == 'ARX':
        identifier = ARX(dt=args.dt)
        model = identifier.identify(U, Y, input_order=args.input_order, output_order=args.output_order)
    else:
        raise ValueError(f"不支持的辨识方法: {args.method}")

    model.summary()

    if args.reduce_order and args.n_reduced < model.n_states:
        print(f"\n[5/6] 模型降阶: {model.n_states} → {args.n_reduced}...")
        model = model.reduce_order(args.n_reduced)
        model.summary()

    print("\n[6/6] 模型验证...")
    validator = ModelValidator(model)

    Y_pred, errors = validator.one_step_prediction(U, Y)
    metrics = validator.compute_metrics(Y, Y_pred)

    print("\n误差指标:")
    print(f"  RMSE: {metrics['overall']['RMSE']:.6f}")
    print(f"  MAE:  {metrics['overall']['MAE']:.6f}")
    print(f"  R²:   {metrics['overall']['R2']:.6f}")
    print(f"  VAF:  {metrics['overall']['VAF']:.6f}")

    stab = validator.stability_analysis()
    print(f"\n稳定性分析:")
    print(f"  模型稳定: {'是' if stab['stable'] else '否'}")
    print(f"  最大极点幅值: {stab['max_magnitude']:.6f}")

    if args.export:
        print(f"\n导出模型到: {args.export_dir}")
        exporter = ModelExporter(model)
        os.makedirs(args.export_dir, exist_ok=True)

        exporter.export_pickle(os.path.join(args.export_dir, 'model.pkl'))
        exporter.export_numpy(os.path.join(args.export_dir, 'model.npz'))
        exporter.export_json(os.path.join(args.export_dir, 'model.json'))
        exporter.export_python_module(os.path.join(args.export_dir, 'rom_model.py'))

        if args.export_matlab:
            exporter.export_matlab(os.path.join(args.export_dir, 'model.mat'))
        if args.export_simulink:
            exporter.export_simulink_sfunction(os.path.join(args.export_dir, 'rom_sfunction.m'))
        if args.export_fortran:
            exporter.export_fortran(os.path.join(args.export_dir, 'rom_model.f90'))
        if args.export_nastran:
            exporter.export_nastran_dmi(os.path.join(args.export_dir, 'model.bdf'))
        if args.export_csv:
            exporter.export_csv(os.path.join(args.export_dir, 'csv_export'))

    if args.plot:
        print("\n生成可视化图表...")
        plotter = Plotter()
        os.makedirs(args.output_dir, exist_ok=True)

        plotter.plot_comparison(
            time, Y, Y_pred,
            title='模型预测对比',
            save_path=os.path.join(args.output_dir, 'prediction_comparison.png')
        )

        omega, H = model.frequency_response()
        plotter.plot_bode(
            omega, H,
            title='Bode图',
            save_path=os.path.join(args.output_dir, 'bode_plot.png')
        )

        poles = model.eigenvalues()
        fn, zeta = model.natural_frequencies_damping()
        plotter.plot_pole_zero(
            poles,
            title='零极点图',
            save_path=os.path.join(args.output_dir, 'pole_zero.png')
        )

        plotter.plot_stability(
            fn, zeta,
            title='频率-阻尼分布',
            save_path=os.path.join(args.output_dir, 'stability.png')
        )

        plotter.plot_residuals(
            time, errors,
            title='残差分析',
            save_path=os.path.join(args.output_dir, 'residuals.png')
        )

        if args.method == 'ERA' and hasattr(identifier, 'singular_values'):
            plotter.plot_singular_values(
                identifier.singular_values,
                suggested_order=args.n_states,
                title='奇异值分布',
                save_path=os.path.join(args.output_dir, 'singular_values.png')
            )

        print(f"  图表已保存到: {args.output_dir}")

    print("\n" + "=" * 70)
    print("辨识流程完成！")
    print("=" * 70)

    return model


def main():
    parser = argparse.ArgumentParser(
        description='AeroROM - 气动力降阶模型辨识工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用演示数据运行ERA辨识
  python -m aerorom.main --demo --method ERA --n_states 4 --plot

  # 使用真实数据
  python -m aerorom.main --input_file data.csv --input_cols u1 u2 --output_cols y1 y2 --method ARX --input_order 3 --output_order 3

  # 完整流程，包含导出
  python -m aerorom.main --demo --method ERA --n_states 6 --reduce_order --n_reduced 4 --export --export_dir output --plot
        """
    )

    parser.add_argument('--demo', action='store_true', help='使用演示数据')
    parser.add_argument('--input_file', type=str, help='输入数据文件路径')
    parser.add_argument('--input_cols', type=str, nargs='+', default=['u1', 'u2'], help='输入列名')
    parser.add_argument('--output_cols', type=str, nargs='+', default=['y1', 'y2'], help='输出列名')

    parser.add_argument('--method', type=str, default='ERA', choices=['ERA', 'ARX'], help='辨识方法')
    parser.add_argument('--dt', type=float, default=0.01, help='采样时间 (s)')
    parser.add_argument('--n_samples', type=int, default=1000, help='演示数据样本数')

    parser.add_argument('--n_states', type=int, default=4, help='ERA模型阶数')
    parser.add_argument('--block_rows', type=int, default=None, help='Hankel矩阵块行数')
    parser.add_argument('--input_order', type=int, default=3, help='ARX输入阶数')
    parser.add_argument('--output_order', type=int, default=3, help='ARX输出阶数')

    parser.add_argument('--reduce_order', action='store_true', help='执行模型降阶')
    parser.add_argument('--n_reduced', type=int, default=4, help='降阶后的状态数')

    parser.add_argument('--denoise', action='store_true', help='执行数据去噪')
    parser.add_argument('--denoise_method', type=str, default='butterworth', help='去噪方法')
    parser.add_argument('--cutoff_freq', type=float, default=20.0, help='截止频率 (Hz)')
    parser.add_argument('--standardize', action='store_true', help='执行数据标准化')
    parser.add_argument('--standardize_method', type=str, default='standard', help='标准化方法')
    parser.add_argument('--noise_level', type=float, default=0.0, help='添加噪声水平 (%)')

    parser.add_argument('--export', action='store_true', help='导出模型')
    parser.add_argument('--export_dir', type=str, default='output/models', help='模型导出目录')
    parser.add_argument('--export_matlab', action='store_true', help='导出MATLAB格式')
    parser.add_argument('--export_simulink', action='store_true', help='导出Simulink S函数')
    parser.add_argument('--export_fortran', action='store_true', help='导出Fortran模块')
    parser.add_argument('--export_nastran', action='store_true', help='导出NASTRAN DMI格式')
    parser.add_argument('--export_csv', action='store_true', help='导出CSV格式')

    parser.add_argument('--plot', action='store_true', help='生成可视化图表')
    parser.add_argument('--output_dir', type=str, default='output/figures', help='图表输出目录')

    args = parser.parse_args()

    if not args.demo and args.input_file is None:
        parser.error("必须指定 --demo 或 --input_file")

    try:
        run_full_pipeline(args)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
