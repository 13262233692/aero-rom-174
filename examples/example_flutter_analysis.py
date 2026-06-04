"""
颤振分析示例 - 基于辨识的ROM进行V-g图计算和颤振临界速度预测

本示例展示：
1. 创建结构动力学模型（二元翼段）
2. 创建或辨识气动力ROM模型
3. 构建气动弹性系统状态方程
4. 计算V-g图和V-f图
5. 检测颤振临界速度
6. 绘制颤振分析结果图表
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from aerorom import (
    FlutterAnalysis,
    Plotter,
    create_standard_section_model,
    create_flutter_demo_rom,
    ModelExporter
)


def example_1_basic_flutter_analysis():
    """
    示例1: 基础颤振分析 - 二元翼段
    """
    print("\n" + "=" * 70)
    print("示例1: 基础颤振分析 - 二元翼段")
    print("=" * 70)

    dt = 0.01

    print("\n1. 创建结构动力学模型（二元翼段：沉浮+俯仰）")
    structural_model = create_standard_section_model(
        n_modes=2,
        f1=5.0,      # 沉浮模态频率 5 Hz
        f2=15.0,     # 俯仰模态频率 15 Hz
        zeta1=0.01,  # 沉浮模态阻尼比 1%
        zeta2=0.01   # 俯仰模态阻尼比 1%
    )

    print(f"   结构模态信息:")
    print(f"     模态1: {structural_model['natural_frequencies'][0]:.1f} Hz, "
          f"阻尼比 {structural_model['damping_ratios'][0]:.3f}")
    print(f"     模态2: {structural_model['natural_frequencies'][1]:.1f} Hz, "
          f"阻尼比 {structural_model['damping_ratios'][1]:.3f}")

    print("\n2. 创建气动力ROM模型")
    rom_model = create_flutter_demo_rom(
        dt=dt,
        n_states=6,
        flutter_mode='bending_torsion'
    )
    rom_model.summary()

    print("\n3. 初始化颤振分析器")
    flutter = FlutterAnalysis(rom_model=rom_model, structural_model=structural_model)

    print("\n4. 计算V-g图和颤振速度")
    results = flutter.compute_flutter_velocity(
        v_min=10.0,
        v_max=150.0,
        n_points=40,
        rho=1.225,
        b_ref=1.0
    )

    print("\n5. 计算根轨迹")
    root_locus = flutter.compute_root_locus(
        v_min=10.0,
        v_max=150.0,
        n_points=20
    )
    results['root_locus'] = root_locus

    print("\n6. 计算阻尼导数")
    damping_derivs = flutter.compute_damping_derivatives()

    print("\n7. 打印分析摘要")
    flutter.print_summary()

    Vf = results['flutter_speed']
    if Vf is not None:
        print(f"\n8. 计算各速度下的稳定裕度")
        test_velocities = [30, 50, 80, Vf * 0.8]
        for V in test_velocities:
            margin = flutter.get_stability_margin(V)
            status = "稳定" if margin['stable'] else "不稳定"
            print(f"   V = {V:.1f} m/s: {status}, "
                  f"稳定裕度 = {margin['margin_percent']:.1f}% "
                  f"({margin['margin_velocity']:.1f} m/s)")

    print("\n9. 绘制颤振分析图表")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    plotter = Plotter(figsize=(10, 6), dpi=100)

    plotter.plot_vg(
        results['velocities'],
        results['dampings'],
        results['frequencies'],
        flutter_speed=results['flutter_speed'],
        title='V-g / V-f 图 - 二元翼段颤振分析',
        save_path=os.path.join(output_dir, 'flutter_vg_vf.png')
    )
    print("   ✓ 已保存: output/flutter_vg_vf.png")

    plotter.plot_root_locus(
        root_locus['roots'],
        velocities=root_locus['velocities'],
        flutter_speed=results['flutter_speed'],
        title='根轨迹图 - 二元翼段颤振分析',
        xlim=[-5, 2],
        ylim=[-150, 150],
        save_path=os.path.join(output_dir, 'flutter_root_locus.png')
    )
    print("   ✓ 已保存: output/flutter_root_locus.png")

    plotter.plot_damping_derivatives(
        results['velocities'],
        damping_derivs,
        title='阻尼导数曲线',
        save_path=os.path.join(output_dir, 'flutter_damping_derivs.png')
    )
    print("   ✓ 已保存: output/flutter_damping_derivs.png")

    results_with_rl = dict(results)
    results_with_rl['root_locus'] = root_locus
    plotter.plot_flutter_summary(
        results_with_rl,
        title='颤振分析汇总 - 二元翼段',
        save_path=os.path.join(output_dir, 'flutter_summary.png')
    )
    print("   ✓ 已保存: output/flutter_summary.png")

    print("\n10. 导出颤振分析结果")
    exporter = ModelExporter(rom_model)
    exporter.export_json(
        os.path.join(output_dir, 'flutter_rom_model.json'),
        indent=2
    )
    print("   ✓ 已保存: output/flutter_rom_model.json")

    flutter_results_file = os.path.join(output_dir, 'flutter_results.npz')
    np.savez(
        flutter_results_file,
        velocities=results['velocities'],
        dampings=results['dampings'],
        frequencies=results['frequencies'],
        flutter_speed=np.array([results['flutter_speed']] if results['flutter_speed'] else []),
        flutter_frequency=np.array([results['flutter_info'].get('flutter_frequency', np.nan)]
                                   if results['flutter_info'] else []),
        flutter_mode=np.array([results['flutter_info'].get('flutter_mode', np.nan)]
                              if results['flutter_info'] else []),
        rho=results['rho'],
        b_ref=results['b_ref']
    )
    print("   ✓ 已保存: output/flutter_results.npz")

    return results


def example_2_parametric_study():
    """
    示例2: 参数化分析 - 质量比对颤振速度的影响
    """
    print("\n" + "=" * 70)
    print("示例2: 参数化分析 - 质量比对颤振速度的影响")
    print("=" * 70)

    dt = 0.01
    rom_model = create_flutter_demo_rom(dt=dt, n_states=6)

    mass_ratios = [0.5, 1.0, 2.0, 5.0]
    flutter_speeds = []

    for idx, mr in enumerate(mass_ratios):
        print(f"\n分析质量比 {mr} ({idx + 1}/{len(mass_ratios)})...")

        f1 = 5.0 / np.sqrt(mr)
        structural_model = create_standard_section_model(
            n_modes=2,
            f1=f1,
            f2=15.0 / np.sqrt(mr),
            zeta1=0.01,
            zeta2=0.01
        )

        structural_model['M'] = structural_model['M'] * mr
        structural_model['K'] = structural_model['K'] * mr

        flutter = FlutterAnalysis(rom_model=rom_model, structural_model=structural_model)

        results = flutter.compute_flutter_velocity(
            v_min=20.0,
            v_max=200.0,
            n_points=30
        )

        flutter_speeds.append(results['flutter_speed'])

        if results['flutter_speed']:
            print(f"  质量比 {mr}: 颤振速度 = {results['flutter_speed']:.2f} m/s")
        else:
            print(f"  质量比 {mr}: 分析范围内未检测到颤振")

    print("\n参数化分析结果汇总:")
    print("-" * 50)
    print(f"{'质量比':<10} {'沉浮频率(Hz)':<15} {'颤振速度(m/s)':<15}")
    print("-" * 50)
    for mr, Vf in zip(mass_ratios, flutter_speeds):
        f1 = 5.0 / np.sqrt(mr)
        Vf_str = f"{Vf:.2f}" if Vf else "N/A"
        print(f"{mr:<10.1f} {f1:<15.2f} {Vf_str:<15}")

    return mass_ratios, flutter_speeds


def example_3_real_data_workflow():
    """
    示例3: 完整工作流 - 从ROM辨识到颤振分析
    """
    print("\n" + "=" * 70)
    print("示例3: 完整工作流 - 从ROM辨识到颤振分析")
    print("=" * 70)

    from aerorom import DataPreprocessor, ARX
    from scipy import signal as sig

    print("\n1. 生成仿真数据（模拟CFD/试验数据）")
    dt = 0.01
    n_samples = 2000
    t = np.arange(n_samples) * dt

    structural_model = create_standard_section_model(f1=5, f2=15)
    true_rom = create_flutter_demo_rom(dt=dt, n_states=8)

    u_chirp = sig.chirp(t, f0=1, f1=50, t1=t[-1]).reshape(-1, 1)
    u_chirp2 = sig.chirp(t, f0=0.5, f1=30, t1=t[-1], method='logarithmic').reshape(-1, 1)
    U = np.hstack([u_chirp, u_chirp2])

    Y_aero, _ = true_rom.simulate(U)
    Y_aero += 0.02 * np.random.randn(*Y_aero.shape)

    print(f"   数据长度: {n_samples} 样本")
    print(f"   采样频率: {1/dt:.0f} Hz")
    print(f"   输入维度: {U.shape[1]}")
    print(f"   输出维度: {Y_aero.shape[1]}")

    print("\n2. 数据预处理")
    preprocessor = DataPreprocessor(dt=dt)
    import pandas as pd
    df = pd.DataFrame({
        'time': t,
        'input_1': U[:, 0],
        'input_2': U[:, 1],
        'output_1': Y_aero[:, 0],
        'output_2': Y_aero[:, 1]
    })
    preprocessor.raw_data = df
    preprocessor.set_input_output(['input_1', 'input_2'], ['output_1', 'output_2'])

    preprocessor.check_sampling_requirements()
    preprocessor.denoise(method='butterworth', cutoff=40, order=4)

    U_proc, Y_proc = preprocessor.get_input_output()

    print("\n3. ARX模型辨识")
    arx = ARX(dt=dt)
    params = preprocessor.recommend_identification_params()

    input_order = 12
    output_order = 12

    rom_identified = arx.identify(
        U_proc, Y_proc,
        input_order=input_order,
        output_order=output_order
    )
    rom_identified.summary()

    print("\n4. 模型验证")
    from aerorom import ModelValidator
    validator = ModelValidator(rom_identified)
    Y_pred, errors = validator.one_step_prediction(U_proc, Y_proc)
    metrics = validator.compute_metrics(Y_proc, Y_pred)
    print(f"   辨识 RMSE: {metrics['overall']['RMSE']:.6f}")
    print(f"   辨识 R²: {metrics['overall']['R2']:.6f}")

    print("\n5. 模型降阶")
    rom_reduced = rom_identified.reduce_order(n_reduced=6)
    rom_reduced.summary()

    print("\n6. 颤振分析")
    flutter = FlutterAnalysis(rom_model=rom_reduced, structural_model=structural_model)
    results = flutter.compute_flutter_velocity(
        v_min=10.0,
        v_max=150.0,
        n_points=30
    )
    flutter.print_summary()

    print("\n7. 输出目录: output/")
    print("   - flutter_vg_vf.png: V-g/V-f图")
    print("   - flutter_root_locus.png: 根轨迹图")
    print("   - flutter_summary.png: 颤振分析汇总图")
    print("   - flutter_results.npz: 颤振分析结果数据")

    return rom_identified, results


def run_all_examples():
    """
    运行所有示例
    """
    print("\n" + "=" * 70)
    print("气动力降阶模型辨识工具 - 颤振分析示例")
    print("=" * 70)
    print("""
本示例展示如何基于辨识的气动力ROM进行颤振分析：
  1. 基础颤振分析 - 二元翼段V-g图计算
  2. 参数化分析 - 质量比对颤振速度的影响
  3. 完整工作流 - 从ROM辨识到颤振分析

关键函数：
  - FlutterAnalysis: 颤振分析主类
  - compute_flutter_velocity(): 计算V-g图和颤振速度
  - compute_root_locus(): 计算根轨迹
  - compute_damping_derivatives(): 计算阻尼导数
  - get_stability_margin(): 计算指定速度下的稳定裕度
  - create_standard_section_model(): 创建二元翼段结构模型
  - create_flutter_demo_rom(): 创建演示用气动力ROM

颤振判据：
  阻尼比由正变负（穿越零轴线）对应的速度即为颤振临界速度
""")

    print("\n" + "-" * 70)
    print("开始运行示例...")
    print("-" * 70)

    results1 = example_1_basic_flutter_analysis()

    try:
        mr, Vf = example_2_parametric_study()
    except Exception as e:
        print(f"示例2执行出错: {e}")

    try:
        rom, results3 = example_3_real_data_workflow()
    except Exception as e:
        print(f"示例3执行出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("颤振分析示例运行完成！")
    print("=" * 70)
    print("""
分析结果说明：
  ✓ V-g图：各阶模态阻尼随速度的变化曲线
  ✓ V-f图：各阶模态频率随速度的变化曲线
  ✓ 根轨迹：s平面极点随速度变化的轨迹
  ✓ 阻尼导数：阻尼随速度的变化率

输出文件：
  - output/flutter_vg_vf.png        V-g/V-f图
  - output/flutter_root_locus.png   根轨迹图
  - output/flutter_damping_derivs.png 阻尼导数曲线
  - output/flutter_summary.png      颤振分析汇总图
  - output/flutter_results.npz      颤振分析结果数据
  - output/flutter_rom_model.json   ROM模型JSON文件
""")


if __name__ == '__main__':
    run_all_examples()
