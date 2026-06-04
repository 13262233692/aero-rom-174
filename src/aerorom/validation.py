"""
模型验证模块

提供模型验证和误差分析功能，包括：
- 时域验证（一步预测、多步预测）
- 频域验证（频率响应对比）
- 误差指标计算（RMSE, MAE, R², VAF等）
- 残差分析
- 稳定性分析
"""

import numpy as np
from scipy import stats
from scipy.signal import welch, csd


class ModelValidator:
    """
    模型验证类

    提供多种验证方法评估辨识模型的质量。
    """

    def __init__(self, model=None):
        """
        初始化模型验证器

        参数:
            model: 待验证的状态空间模型
        """
        self.model = model
        self.results = {}

    def set_model(self, model):
        """
        设置待验证的模型

        参数:
            model: 状态空间模型

        返回:
            self: 链式调用
        """
        self.model = model
        return self

    def one_step_prediction(self, U, Y, x0=None):
        """
        一步预测验证

        参数:
            U: 输入数据 (n_samples, n_inputs)
            Y: 真实输出数据 (n_samples, n_outputs)
            x0: 初始状态

        返回:
            Y_pred: 预测输出 (n_samples, n_outputs)
            errors: 预测误差 (n_samples, n_outputs)
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        U = np.atleast_2d(U)
        Y = np.atleast_2d(Y)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n_samples = U.shape[0]
        n_states = self.model.n_states
        n_outputs = self.model.n_outputs

        X = np.zeros((n_samples, n_states))
        Y_pred = np.zeros((n_samples, n_outputs))

        if x0 is not None:
            X[0] = x0

        for k in range(n_samples):
            Y_pred[k] = self.model.C @ X[k] + self.model.D @ U[k]
            if k < n_samples - 1:
                X[k + 1] = self.model.A @ X[k] + self.model.B @ U[k]

        errors = Y - Y_pred
        self.results['one_step'] = {
            'Y_true': Y,
            'Y_pred': Y_pred,
            'errors': errors
        }
        return Y_pred, errors

    def multi_step_prediction(self, U, Y, x0=None, step_size=None):
        """
        多步预测验证

        参数:
            U: 输入数据 (n_samples, n_inputs)
            Y: 真实输出数据 (n_samples, n_outputs)
            x0: 初始状态
            step_size: 预测步长，默认整个序列

        返回:
            Y_pred: 预测输出 (n_samples, n_outputs)
            errors: 预测误差 (n_samples, n_outputs)
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        U = np.atleast_2d(U)
        Y = np.atleast_2d(Y)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if step_size is None:
            step_size = len(U)

        n_samples = U.shape[0]
        Y_pred = np.zeros_like(Y)

        for i in range(0, n_samples, step_size):
            end = min(i + step_size, n_samples)
            U_seg = U[i:end]

            if i == 0 and x0 is not None:
                x0_seg = x0
            else:
                x0_seg = None

            Y_seg, _ = self.model.simulate(U_seg, x0=x0_seg)
            Y_pred[i:end] = Y_seg

        errors = Y - Y_pred
        self.results['multi_step'] = {
            'Y_true': Y,
            'Y_pred': Y_pred,
            'errors': errors,
            'step_size': step_size
        }
        return Y_pred, errors

    def compute_metrics(self, Y_true, Y_pred, names=None):
        """
        计算多种误差指标

        参数:
            Y_true: 真实输出 (n_samples, n_outputs)
            Y_pred: 预测输出 (n_samples, n_outputs)
            names: 输出名称列表

        返回:
            metrics: 包含各指标的字典
        """
        Y_true = np.atleast_2d(Y_true)
        Y_pred = np.atleast_2d(Y_pred)
        if Y_true.ndim == 1:
            Y_true = Y_true.reshape(-1, 1)
        if Y_pred.ndim == 1:
            Y_pred = Y_pred.reshape(-1, 1)

        n_outputs = Y_true.shape[1]
        if names is None:
            names = [f'output_{i}' for i in range(n_outputs)]

        metrics = {}

        for i, name in enumerate(names):
            yt = Y_true[:, i]
            yp = Y_pred[:, i]
            error = yt - yp

            mse = np.mean(error**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(error))
            mape = np.mean(np.abs(error / (np.abs(yt) + 1e-10))) * 100

            ss_res = np.sum(error**2)
            ss_tot = np.sum((yt - np.mean(yt))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)

            vaf = 1 - np.var(error) / (np.var(yt) + 1e-10)

            max_error = np.max(np.abs(error))

            metrics[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2,
                'VAF': vaf,
                'MaxError': max_error
            }

        metrics['overall'] = {
            'MSE': np.mean([m['MSE'] for m in metrics.values()]),
            'RMSE': np.mean([m['RMSE'] for m in metrics.values()]),
            'MAE': np.mean([m['MAE'] for m in metrics.values()]),
            'MAPE': np.mean([m['MAPE'] for m in metrics.values()]),
            'R2': np.mean([m['R2'] for m in metrics.values()]),
            'VAF': np.mean([m['VAF'] for m in metrics.values()])
        }

        self.results['metrics'] = metrics
        return metrics

    def residual_analysis(self, errors, dt=None):
        """
        残差分析

        参数:
            errors: 残差序列 (n_samples, n_outputs)
            dt: 采样时间

        返回:
            analysis: 残差分析结果
        """
        errors = np.atleast_2d(errors)
        if errors.ndim == 1:
            errors = errors.reshape(-1, 1)

        n_samples, n_outputs = errors.shape
        if dt is None:
            dt = self.model.dt if self.model else 1.0

        analysis = {}

        for i in range(n_outputs):
            err = errors[:, i]

            mean = np.mean(err)
            std = np.std(err)
            skewness = stats.skew(err)
            kurtosis = stats.kurtosis(err)

            _, p_value = stats.shapiro(err) if n_samples < 5000 else (0, 0.5)

            ad_stat, ad_crit, ad_sig = stats.anderson(err, dist='norm')

            auto_corr = np.correlate(err - mean, err - mean, mode='full')
            auto_corr = auto_corr[n_samples - 1:] / (n_samples * std**2)

            freqs, psd = welch(err, fs=1.0/dt, nperseg=min(256, n_samples//4))

            ljung_box = self._ljung_box_test(auto_corr, n_samples, lags=20)

            analysis[f'output_{i}'] = {
                'mean': mean,
                'std': std,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shapiro_p': p_value,
                'ad_statistic': ad_stat,
                'autocorrelation': auto_corr[:40],
                'psd_freqs': freqs,
                'psd': psd,
                'ljung_box': ljung_box
            }

        self.results['residual_analysis'] = analysis
        return analysis

    def _ljung_box_test(self, autocorr, n_samples, lags=20):
        """
        Ljung-Box白噪声检验

        参数:
            autocorr: 自相关系数
            n_samples: 样本数
            lags: 滞后阶数

        返回:
            Q: 检验统计量
            p_value: p值
        """
        acf = autocorr[:lags + 1]
        Q = n_samples * (n_samples + 2) * np.sum(acf[1:]**2 / (n_samples - np.arange(1, lags + 1)))
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(Q, lags)
        return {'Q': Q, 'p_value': p_value}

    def stability_analysis(self):
        """
        模型稳定性分析

        返回:
            analysis: 稳定性分析结果
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        eig = self.model.eigenvalues()
        magnitudes = np.abs(eig)
        stable = np.all(magnitudes < 1.0)

        fn, zeta = self.model.natural_frequencies_damping()

        analysis = {
            'stable': stable,
            'eigenvalues': eig,
            'magnitudes': magnitudes,
            'max_magnitude': np.max(magnitudes),
            'natural_frequencies': fn,
            'damping_ratios': zeta,
            'unstable_modes': np.sum(magnitudes >= 1.0)
        }

        self.results['stability'] = analysis
        return analysis

    def frequency_domain_validation(self, U, Y, dt=None, omega=None):
        """
        频域验证

        参数:
            U: 输入数据 (n_samples, n_inputs)
            Y: 输出数据 (n_samples, n_outputs)
            dt: 采样时间
            omega: 频率点向量

        返回:
            analysis: 频域验证结果
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        U = np.atleast_2d(U)
        Y = np.atleast_2d(Y)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if dt is None:
            dt = self.model.dt
        fs = 1.0 / dt

        n_inputs = U.shape[1]
        n_outputs = Y.shape[1]

        if omega is None:
            omega = np.logspace(-2, np.log10(np.pi / dt), 100)

        omega_model, H_model = self.model.frequency_response(omega)

        H_empirical = np.zeros((len(omega), n_outputs, n_inputs), dtype=complex)
        coherence = np.zeros((len(omega), n_outputs, n_inputs))

        nperseg = min(512, len(U) // 4)

        for i in range(n_outputs):
            for j in range(n_inputs):
                f, Pxy = csd(U[:, j], Y[:, i], fs=fs, nperseg=nperseg)
                _, Pxx = welch(U[:, j], fs=fs, nperseg=nperseg)
                _, Pyy = welch(Y[:, i], fs=fs, nperseg=nperseg)

                H_emp_ij = Pxy / (Pxx + 1e-20)
                coh_ij = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-20)

                for k, w in enumerate(omega):
                    f_idx = np.argmin(np.abs(f - w / (2 * np.pi)))
                    H_empirical[k, i, j] = H_emp_ij[f_idx]
                    coherence[k, i, j] = coh_ij[f_idx]

        freq_error = np.abs(H_model - H_empirical) / (np.abs(H_empirical) + 1e-10) * 100

        analysis = {
            'omega': omega,
            'H_model': H_model,
            'H_empirical': H_empirical,
            'coherence': coherence,
            'relative_error': freq_error,
            'mean_relative_error': np.mean(freq_error)
        }

        self.results['frequency_domain'] = analysis
        return analysis

    def pole_zero_analysis(self):
        """
        零极点分析

        返回:
            analysis: 零极点分析结果
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        poles = self.model.eigenvalues()

        zeros = []
        for i in range(self.model.n_outputs):
            for j in range(self.model.n_inputs):
                A = self.model.A
                B = self.model.B[:, j:j+1]
                C = self.model.C[i:i+1, :]
                D = self.model.D[i:i+1, j:j+1]

                M = np.block([[A, B], [C, D]])
                z = np.linalg.eigvals(M)
                zeros.append(z)

        analysis = {
            'poles': poles,
            'zeros': zeros,
            'pole_magnitudes': np.abs(poles),
            'stable': np.all(np.abs(poles) < 1.0)
        }

        self.results['pole_zero'] = analysis
        return analysis

    def compare_models(self, models, U, Y, metric='RMSE'):
        """
        比较多个模型的性能

        参数:
            models: 模型列表
            U: 输入数据
            Y: 输出数据
            metric: 比较指标

        返回:
            comparison: 比较结果
        """
        comparison = {}

        for model in models:
            original_model = self.model
            self.model = model

            Y_pred, _ = self.one_step_prediction(U, Y)
            metrics = self.compute_metrics(Y, Y_pred)

            comparison[model.name] = {
                'n_states': model.n_states,
                'metric_value': metrics['overall'][metric],
                'metrics': metrics
            }

            self.model = original_model

        sorted_models = sorted(comparison.items(), key=lambda x: x[1]['metric_value'])
        comparison['ranked'] = [name for name, _ in sorted_models]

        self.results['model_comparison'] = comparison
        return comparison

    def cross_validation(self, U, Y, n_folds=5, method='ERA', **kwargs):
        """
        交叉验证

        参数:
            U: 输入数据
            Y: 输出数据
            n_folds: 折数
            method: 辨识方法 ('ERA', 'ARX')
            **kwargs: 辨识参数

        返回:
            cv_results: 交叉验证结果
        """
        from .identification import ERA, ARX

        n_samples = len(U)
        fold_size = n_samples // n_folds

        cv_results = {
            'fold_metrics': [],
            'mean_metrics': None,
            'std_metrics': None
        }

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size

            U_val = U[val_start:val_end]
            Y_val = Y[val_start:val_end]

            U_train = np.vstack([U[:val_start], U[val_end:]])
            Y_train = np.vstack([Y[:val_start], Y[val_end:]])

            if method == 'ERA':
                identifier = ERA(dt=self.model.dt if self.model else 1.0)
                model = identifier.identify(Y_train, U_train, **kwargs)
            elif method == 'ARX':
                identifier = ARX(dt=self.model.dt if self.model else 1.0)
                model = identifier.identify(U_train, Y_train, **kwargs)
            else:
                raise ValueError(f"不支持的辨识方法: {method}")

            original_model = self.model
            self.model = model

            Y_pred, _ = self.one_step_prediction(U_val, Y_val)
            metrics = self.compute_metrics(Y_val, Y_pred)

            cv_results['fold_metrics'].append(metrics['overall'])
            self.model = original_model

        metric_names = list(cv_results['fold_metrics'][0].keys())
        cv_results['mean_metrics'] = {
            name: np.mean([fold[name] for fold in cv_results['fold_metrics']])
            for name in metric_names
        }
        cv_results['std_metrics'] = {
            name: np.std([fold[name] for fold in cv_results['fold_metrics']])
            for name in metric_names
        }

        self.results['cross_validation'] = cv_results
        return cv_results

    def print_report(self):
        """
        打印验证报告
        """
        print("=" * 70)
        print("模型验证报告")
        print("=" * 70)

        if 'metrics' in self.results:
            print("\n误差指标:")
            print("-" * 50)
            metrics = self.results['metrics']
            for name, vals in metrics.items():
                if name == 'overall':
                    continue
                print(f"\n{name}:")
                print(f"  RMSE: {vals['RMSE']:.6f}")
                print(f"  MAE:  {vals['MAE']:.6f}")
                print(f"  R²:   {vals['R2']:.6f}")
                print(f"  VAF:  {vals['VAF']:.6f}")

            print(f"\n整体指标:")
            for k, v in metrics['overall'].items():
                print(f"  {k}: {v:.6f}")

        if 'stability' in self.results:
            print("\n稳定性分析:")
            print("-" * 50)
            stab = self.results['stability']
            print(f"模型稳定: {'是' if stab['stable'] else '否'}")
            print(f"最大极点幅值: {stab['max_magnitude']:.6f}")
            print(f"不稳定模式数: {stab['unstable_modes']}")

        if 'frequency_domain' in self.results:
            print("\n频域验证:")
            print("-" * 50)
            fd = self.results['frequency_domain']
            print(f"平均相对误差: {fd['mean_relative_error']:.2f}%")

        print("\n" + "=" * 70)

    def frequency_band_analysis(self, U, Y, dt=None, n_bands=5):
        """
        分频段拟合质量分析

        将频率范围分成多个频段，分别计算每个频段的拟合误差，
        用于检测高频段拟合质量诊断。

        参数:
            U: 输入数据 (n_samples, n_inputs)
            Y: 输出数据 (n_samples, n_outputs)
            dt: 采样时间
            n_bands: 频段数量

        返回:
            band_analysis: 分频段分析结果
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        freq_analysis = self.frequency_domain_validation(U, Y, dt)
        omega = freq_analysis['omega']
        H_model = freq_analysis['H_model']
        H_empirical = freq_analysis['H_empirical']

        f = omega / (2 * np.pi)
        f_max = np.max(f)
        band_edges = np.linspace(0, f_max, n_bands + 1)

        band_analysis = {
            'bands': [],
            'band_edges': band_edges,
            'band_errors': []
        }

        print("\n" + "=" * 70)
        print("分频段拟合质量分析")
        print("=" * 70)
        print(f"{'频段':<15} {'频率范围(Hz)':<20} {'平均误差(%)':<15} {'质量评估':<15}")
        print("-" * 70)

        for i in range(n_bands):
            f_start = band_edges[i]
            f_end = band_edges[i + 1]
            mask = (f >= f_start) & (f <= f_end)

            if np.sum(mask) == 0:
                continue

            error_band = freq_analysis['relative_error'][mask, :, :]
            mean_error = np.mean(error_band)

            if mean_error < 5:
                quality = "优秀"
            elif mean_error < 15:
                quality = "良好"
            elif mean_error < 30:
                quality = "一般"
            else:
                quality = "较差"

            band_info = {
                'band_index': i,
                'f_start': f_start,
                'f_end': f_end,
                'mean_error': mean_error,
                'quality': quality
            }
            band_analysis['bands'].append(band_info)

            print(f"频段 {i + 1:<5} {f_start:>8.2f} - {f_end:<8.2f} {mean_error:>12.2f} {quality:>12}")

        high_freq_threshold = f_max * 0.7
        high_freq_mask = f >= high_freq_threshold
        if np.sum(high_freq_mask) > 0:
            high_error = np.mean(freq_analysis['relative_error'][high_freq_mask, :, :])
            band_analysis['high_frequency_error'] = high_error

            print("-" * 70)
            if high_error > 20:
                print(f"⚠ 警告：高频段(> {high_freq_threshold:.1f}Hz)平均误差 {high_error:.1f}%")
                print("  可能原因：")
                print("  1. 采样频率不足（建议5~10倍过采样）")
                print("  2. 高频信号噪声过大")
                print("  3. 模型阶数不足以捕捉高频动态")
                print("  建议：使用antialiasing_filter()进行抗混叠滤波")
            else:
                print(f"✓ 高频段拟合质量良好")

        print("=" * 70 + "\n")

        return band_analysis

    def diagnose_high_frequency_issues(self, U, Y, dt=None):
        """
        综合诊断高频段拟合问题

        参数:
            U: 输入数据 (n_samples, n_inputs)
            Y: 输出数据 (n_samples, n_outputs)
            dt: 采样时间

        返回:
            diagnosis: 诊断结果字典
        """
        from aerorom.identification import estimate_max_frequency

        dt = dt if dt is not None else self.model.dt
        fs = 1.0 / dt
        nyquist = fs / 2

        diagnosis = {
            'issues': [],
            'recommendations': []
        }

        max_freq_U = estimate_max_frequency(U, fs)
        max_freq_Y = estimate_max_frequency(Y, fs)
        max_freq = max(max_freq_U, max_freq_Y)

        diagnosis['signal_max_frequency'] = max_freq
        diagnosis['sampling_frequency'] = fs
        diagnosis['nyquist_frequency'] = nyquist
        diagnosis['oversampling_ratio'] = fs / max_freq if max_freq > 0 else float('inf')

        print("\n" + "=" * 70)
        print("高频拟合问题诊断")
        print("=" * 70)
        print(f"信号最高频率: {max_freq:.2f} Hz")
        print(f"采样频率: {fs:.2f} Hz")
        print(f"奈奎斯特频率: {nyquist:.2f} Hz")
        print(f"过采样比: {diagnosis['oversampling_ratio']:.1f}x")
        print("-" * 50)

        if max_freq > nyquist * 0.5:
            issue = "信号频率接近奈奎斯特频率"
            diagnosis['issues'].append(issue)
            print(f"⚠ {issue}")
            diagnosis['recommendations'].append(
                "使用 DataPreprocessor.antialiasing_filter() 进行抗混叠滤波"
            )
            diagnosis['recommendations'].append(
                "降低输入信号的最高频率或提高采样频率"
            )

        if diagnosis['oversampling_ratio'] < 5:
            issue = "过采样比不足"
            diagnosis['issues'].append(issue)
            print(f"⚠ {issue} (建议5~10倍)")
            diagnosis['recommendations'].append(
                "提高采样频率到信号最高频率的5~10倍")

        freq_analysis = self.frequency_domain_validation(U, Y, dt)
        high_freq_mask = freq_analysis['omega'] / (2 * np.pi) > max_freq * 0.7
        if np.sum(high_freq_mask) > 0:
            high_error = np.mean(freq_analysis['relative_error'][high_freq_mask, :, :])
            diagnosis['high_frequency_error'] = high_error

            if high_error > 20:
                issue = "高频段拟合误差过高"
                diagnosis['issues'].append(issue)
                print(f"⚠ {issue}: {high_error:.1f}%)")
                diagnosis['recommendations'].append(
                    "增加模型阶数以捕捉高频动态")
                diagnosis['recommendations'].append(
                    "检查输入信号的高频信噪比")

        if not diagnosis['issues']:
            print("✓ 未发现明显的高频拟合问题")
        else:
            print("-" * 50)
            print("建议措施:")
            for i, rec in enumerate(diagnosis['recommendations'], 1):
                print(f"  {i}. {rec}")

        print("=" * 70 + "\n")

        return diagnosis
