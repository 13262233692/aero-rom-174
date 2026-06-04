"""
扫频信号辨识示例 - 演示如何检测和解决高频段失真问题

本示例展示：
1. 生成扫频信号作为系统输入
2. 演示不正确的采样频率导致的高频失真问题
3. 使用频率诊断工具检测问题
4. 使用抗混叠滤波解决问题
5. 对比滤波前后的辨识效果
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

from aerorom import (
    DataPreprocessor,
    ERA,
    ARX,
    StateSpaceModel,
    ModelValidator,
    ModelExporter,
    Plotter
)


def generate_test_system(dt=0.01, n_states=6, n_inputs=1, n_outputs=1):
    """
    生成测试用的气动力系统模型
    """
    np.random.seed(42)

    A = np.random.randn(n_states, n_states)
    A = A / np.max(np.abs(np.linalg.eigvals(A))) * 0.9
    B = np.random.randn(n_states, n_inputs)
    C = np.random.randn(n_outputs, n_states)
    D = np.zeros((n_outputs, n_inputs))

    return StateSpaceModel(A, B, C, D, dt)


def generate_chirp_signal(n_samples, dt, f_start=1.0, f_end=50.0, method='linear'):
    """
    生成扫频输入信号

    参数:
        n_samples: 样本数
        dt: 采样时间
        f_start: 起始频率 (Hz)
        f_end: 终止频率 (Hz)
        method: 扫频方式 ('linear', 'logarithmic', 'quadratic')
    """
    t = np.arange(n_samples) * dt
    u = sig.chirp(t, f0=f_start, f1=f_end, t1=t[-1], method=method)
    return u, t


def simulate_bad_sampling():
    """
    演示：采样频率不足导致的高频段失真问题
    """
    print("=" * 70)
    print("示例1: 演示采样频率不足导致的高频段失真问题")
    print("=" * 70)

    dt_good = 0.005
    dt_bad = 0.02

    n_samples = 2000
    t_good = np.arange(n_samples) * dt_good

    true_model = generate_test_system(dt=dt_good, n_states=6, n_inputs=1, n_outputs=1)

    f_start = 1.0
    f_end = 40.0

    u_chirp, _ = generate_chirp_signal(n_samples, dt_good, f_start, f_end, method='linear')
    u_chirp = u_chirp.reshape(-1, 1)

    Y_good, _ = true_model.simulate(u_chirp)
    Y_good += 0.01 * np.random.randn(*Y_good.shape)

    print(f"\n系统设置:")
    print(f"  真实采样频率: {1/dt_good:.0f} Hz (奈奎斯特频率: {0.5/dt_good:.0f} Hz)")
    print(f"  重采样频率: {1/dt_bad:.0f} Hz (奈奎斯特频率: {0.5/dt_bad:.0f} Hz)")
    print(f"  扫频范围: {f_start:.0f} - {f_end:.0f} Hz")

    step = int(dt_bad / dt_good)
    t_bad = t_good[::step]
    u_bad = u_chirp[::step]
    Y_bad = Y_good[::step]

    print(f"\n频率分析...")
    from aerorom.identification import estimate_max_frequency, check_frequency_characteristics

    fs_bad = 1.0 / dt_bad
    max_freq = estimate_max_frequency(Y_bad, fs_bad)
    print(f"  估计信号最高频率: {max_freq:.1f} Hz")

    check_frequency_characteristics(Y_bad, u_bad, dt_bad, 'Bad Sampling Demo')

    print(f"\n使用重采样数据进行ARX辨识...")
    arx_bad = ARX(dt=dt_bad)
    model_bad = arx_bad.identify(u_bad, Y_bad, input_order=10, output_order=10)

    print(f"\n模型验证 - 分频段分析...")
    validator_bad = ModelValidator(model_bad)
    band_analysis = validator_bad.frequency_band_analysis(u_bad, Y_bad, n_bands=6)

    return model_bad, u_bad, Y_bad, t_bad, true_model


def demonstrate_correct_procedure():
    """
    演示：正确的扫频信号辨识流程
    """
    print("\n" + "=" * 70)
    print("示例2: 正确的扫频信号辨识流程")
    print("=" * 70)

    dt = 0.005
    n_samples = 2000
    t = np.arange(n_samples) * dt

    true_model = generate_test_system(dt=dt, n_states=6, n_inputs=1, n_outputs=1)

    f_start = 1.0
    f_end = 40.0

    u_chirp, _ = generate_chirp_signal(n_samples, dt, f_start, f_end, method='linear')
    u_chirp = u_chirp.reshape(-1, 1)

    Y, _ = true_model.simulate(u_chirp)
    Y += 0.01 * np.random.randn(*Y.shape)

    print(f"\n系统设置:")
    print(f"  采样频率: {1/dt:.0f} Hz (奈奎斯特频率: {0.5/dt:.0f} Hz)")
    print(f"  扫频范围: {f_start:.0f} - {f_end:.0f} Hz")
    print(f"  过采样比: {(0.5/dt)/f_end:.1f}x")

    print(f"\n步骤1: 数据预处理 - 频率检查...")
    preprocessor = DataPreprocessor(dt=dt)
    import pandas as pd
    df = pd.DataFrame({
        'time': t,
        'input1': u_chirp.flatten(),
        'output1': Y.flatten()
    })
    preprocessor.raw_data = df
    preprocessor.set_input_output(['input1'], ['output1'])

    preprocessor.check_sampling_requirements()
    preprocessor.recommend_identification_params()

    print(f"\n步骤2: 数据清洗 - 去噪...")
    preprocessor.denoise(method='butterworth', cutoff=50, order=4)

    print(f"\n步骤3: ARX辨识...")
    U_proc, Y_proc = preprocessor.get_input_output()
    arx = ARX(dt=dt)
    model = arx.identify(U_proc, Y_proc, input_order=10, output_order=10)

    print(f"\n步骤4: 模型验证...")
    validator = ModelValidator(model)
    Y_pred, errors = validator.one_step_prediction(U_proc, Y_proc)

    metrics = validator.compute_metrics(Y_proc, Y_pred)
    print(f"  RMSE: {metrics['overall']['RMSE']:.6f}")
    print(f"  R²: {metrics['overall']['R2']:.6f}")

    print(f"\n步骤5: 分频段拟合质量分析...")
    band_analysis = validator.frequency_band_analysis(U_proc, Y_proc, n_bands=6)

    return model, U_proc, Y_proc, t, true_model


def demonstrate_antialiasing_filter():
    """
    演示：使用抗混叠滤波器解决高频失真问题
    """
    print("\n" + "=" * 70)
    print("示例3: 使用抗混叠滤波器解决高频失真问题")
    print("=" * 70)

    dt_orig = 0.005
    dt_downsampled = 0.02
    n_samples = 2000
    t_orig = np.arange(n_samples) * dt_orig

    true_model = generate_test_system(dt=dt_orig, n_states=6, n_inputs=1, n_outputs=1)

    f_start = 1.0
    f_end = 40.0

    u_chirp, _ = generate_chirp_signal(n_samples, dt_orig, f_start, f_end)
    u_chirp = u_chirp.reshape(-1, 1)

    Y_orig, _ = true_model.simulate(u_chirp)
    Y_orig += 0.01 * np.random.randn(*Y_orig.shape)

    print(f"\n原始数据:")
    print(f"  采样频率: {1/dt_orig:.0f} Hz")
    print(f"  扫频范围: {f_start:.0f} - {f_end:.0f} Hz")

    step = int(dt_downsampled / dt_orig)
    t_down = t_orig[::step]
    u_down = u_chirp[::step]
    Y_down = Y_orig[::step]

    print(f"\n重采样后（未滤波）:")
    print(f"  采样频率: {1/dt_downsampled:.0f} Hz")
    print(f"  奈奎斯特频率: {0.5/dt_downsampled:.0f} Hz")

    print(f"\n频率诊断 - 未滤波数据:")
    from aerorom.identification import estimate_max_frequency
    max_freq_before = estimate_max_frequency(Y_down, 1.0/dt_downsampled)
    print(f"  估计信号最高频率: {max_freq_before:.1f} Hz")

    arx_before = ARX(dt=dt_downsampled)
    model_before = arx_before.identify(u_down, Y_down, input_order=8, output_order=8,
                                         check_frequency=False)

    validator_before = ModelValidator(model_before)
    Y_pred_before, _ = validator_before.one_step_prediction(u_down, Y_down)
    rmse_before = np.sqrt(np.mean((Y_pred_before - Y_down)**2))
    print(f"  辨识 RMSE: {rmse_before:.6f}")

    print(f"\n应用抗混叠滤波器...")
    cutoff_freq = 0.4 * 0.5 / dt_downsampled
    preprocessor = DataPreprocessor(dt=dt_orig)
    import pandas as pd
    df_orig = pd.DataFrame({
        'time': t_orig,
        'input1': u_chirp.flatten(),
        'output1': Y_orig.flatten()
    })
    preprocessor.raw_data = df_orig
    preprocessor.set_input_output(['input1'], ['output1'])

    preprocessor.antialiasing_filter(cutoff_freq=cutoff_freq, filter_order=6)
    U_filtered, Y_filtered = preprocessor.get_input_output()

    u_filtered_down = U_filtered[::step]
    Y_filtered_down = Y_filtered[::step]

    print(f"\n频率诊断 - 滤波后数据:")
    max_freq_after = estimate_max_frequency(Y_filtered_down, 1.0/dt_downsampled)
    print(f"  估计信号最高频率: {max_freq_after:.1f} Hz")

    arx_after = ARX(dt=dt_downsampled)
    model_after = arx_after.identify(u_filtered_down, Y_filtered_down,
                                       input_order=8, output_order=8,
                                       check_frequency=False)

    validator_after = ModelValidator(model_after)
    Y_pred_after, _ = validator_after.one_step_prediction(u_filtered_down, Y_filtered_down)
    rmse_after = np.sqrt(np.mean((Y_pred_after - Y_filtered_down)**2))
    print(f"  辨识 RMSE: {rmse_after:.6f}")

    print(f"\n改进效果:")
    print(f"  RMSE改善: {rmse_before:.6f} -> {rmse_after:.6f}")
    print(f"  相对改善: {(1 - rmse_after/rmse_before)*100:.1f}%")

    return {
        'before': {'model': model_before, 'Y_pred': Y_pred_before, 'RMSE': rmse_before},
        'after': {'model': model_after, 'Y_pred': Y_pred_after, 'RMSE': rmse_after},
        't': t_down,
        'Y_true': Y_down,
        'Y_filtered': Y_filtered_down
    }


def run_full_comparison():
    """
    运行完整的对比实验
    """
    print("\n" + "=" * 70)
    print("气动力降阶模型辨识 - 高频失真问题解决方案")
    print("=" * 70)
    print("""
问题描述:
  当输入信号为扫频信号时，如果采样频率与信号最高频率不满足
  辨识理论要求（奈奎斯特采样定理 + 充足的过采样），会导致
  低频段拟合良好，但高频段完全失真。

解决方案:
  1. 提高采样频率（推荐5~10倍过采样）
  2. 使用抗混叠滤波器降低信号带宽
  3. 增加模型阶数以捕捉高频动态
  4. 使用预处理工具进行频率分析和参数推荐
""")

    print("\n" + "-" * 70)
    print("开始运行示例...")
    print("-" * 70)

    results_filter = demonstrate_antialiasing_filter()

    model_good, U_good, Y_good, t_good, true_model = demonstrate_correct_procedure()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
关键要点:
  ✓ 系统辨识前必须进行频率特性分析
  ✓ 采样频率应至少为信号最高频率的5~10倍
  ✓ 如无法提高采样频率，应使用抗混叠滤波
  ✓ 使用frequency_band_analysis()检查各频段拟合质量
  ✓ 使用diagnose_high_frequency_issues()进行综合诊断

工具函数:
  - DataPreprocessor.check_sampling_requirements()  - 采样频率验证
  - DataPreprocessor.antialiasing_filter()         - 抗混叠滤波
  - DataPreprocessor.recommend_identification_params() - 参数推荐
  - ModelValidator.frequency_band_analysis()        - 分频段拟合分析
  - ModelValidator.diagnose_high_frequency_issues() - 高频问题诊断
""")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    run_full_comparison()
