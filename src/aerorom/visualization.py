"""
可视化模块

提供数据和模型结果的可视化功能，包括：
- 原始数据和处理后数据的时域图
- 模型预测与真实值对比
- 频率响应函数（Bode图）
- 奇异值分布（阶数选择）
- 残差分析
- 极零点图
- 稳定性分析图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

try:
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    rcParams['axes.unicode_minus'] = False
except:
    pass


class Plotter:
    """
    绘图类

    提供多种可视化方法，支持保存为图片文件。
    """

    def __init__(self, style='seaborn-v0_8-whitegrid', figsize=(10, 6), dpi=100):
        """
        初始化绘图器

        参数:
            style: matplotlib样式
            figsize: 默认图像大小
            dpi: 图像分辨率
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        self.figsize = figsize
        self.dpi = dpi

    def plot_time_series(self, time, data, labels=None, title='时域信号',
                         xlabel='时间 (s)', ylabel='幅值',
                         show_legend=True, grid=True, save_path=None):
        """
        绘制时域信号

        参数:
            time: 时间向量 (n_samples,)
            data: 数据矩阵 (n_samples, n_channels) 或 (n_samples,)
            labels: 通道标签列表
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            show_legend: 是否显示图例
            grid: 是否显示网格
            save_path: 保存路径，如None则显示

        返回:
            fig: matplotlib Figure对象
            ax: matplotlib Axes对象
        """
        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T

        n_channels = data.shape[1]
        if labels is None:
            labels = [f'通道 {i+1}' for i in range(n_channels)]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for i in range(n_channels):
            ax.plot(time, data[:, i], label=labels[i], linewidth=1.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if show_legend:
            ax.legend(loc='best', fontsize=10)
        if grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, ax

    def plot_comparison(self, time, y_true, y_pred, labels=None,
                        title='模型预测对比', xlabel='时间 (s)',
                        ylabel='输出', show_error=True, save_path=None):
        """
        绘制模型预测与真实值对比

        参数:
            time: 时间向量
            y_true: 真实输出 (n_samples, n_outputs)
            y_pred: 预测输出 (n_samples, n_outputs)
            labels: 输出标签列表
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            show_error: 是否显示误差曲线
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            axes: matplotlib Axes对象列表
        """
        y_true = np.atleast_2d(y_true)
        y_pred = np.atleast_2d(y_pred)
        if y_true.shape[0] == 1:
            y_true = y_true.T
        if y_pred.shape[0] == 1:
            y_pred = y_pred.T

        n_outputs = y_true.shape[1]
        if labels is None:
            labels = [f'输出 {i+1}' for i in range(n_outputs)]

        n_rows = n_outputs * 2 if show_error else n_outputs
        fig, axes = plt.subplots(n_rows, 1, figsize=(self.figsize[0], 3 * n_rows),
                                 dpi=self.dpi, sharex=True)
        if n_rows == 1:
            axes = [axes]

        for i in range(n_outputs):
            idx = i * 2 if show_error else i

            axes[idx].plot(time, y_true[:, i], 'b-', label='真实值',
                          linewidth=2, alpha=0.8)
            axes[idx].plot(time, y_pred[:, i], 'r--', label='预测值',
                          linewidth=1.5, alpha=0.8)
            axes[idx].set_ylabel(f'{labels[i]}\n{ylabel}', fontsize=10)
            axes[idx].set_title(f'{labels[i]} 对比', fontsize=12, fontweight='bold')
            axes[idx].legend(loc='best', fontsize=9)
            axes[idx].grid(True, alpha=0.3)

            if show_error:
                error = y_true[:, i] - y_pred[:, i]
                axes[idx + 1].plot(time, error, 'g-', linewidth=1, alpha=0.8)
                axes[idx + 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                axes[idx + 1].set_ylabel('误差', fontsize=10)
                axes[idx + 1].set_xlabel(xlabel, fontsize=10)
                axes[idx + 1].grid(True, alpha=0.3)
            else:
                axes[idx].set_xlabel(xlabel, fontsize=10)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.002)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, axes

    def plot_bode(self, omega, H, outputs=None, inputs=None,
                  title='Bode图', save_path=None):
        """
        绘制Bode图（幅频和相频特性）

        参数:
            omega: 频率向量 (rad/s)
            H: 频率响应矩阵 (n_freq, n_outputs, n_inputs)
            outputs: 输出索引或名称列表
            inputs: 输入索引或名称列表
            title: 图表标题
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            axes: matplotlib Axes对象 (2, n_plots)
        """
        if outputs is None:
            outputs = list(range(H.shape[1]))
        if inputs is None:
            inputs = list(range(H.shape[2]))

        n_plots = len(outputs) * len(inputs)
        fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 7),
                                 dpi=self.dpi, sharex=True)
        if n_plots == 1:
            axes = axes.reshape(2, 1)

        plot_idx = 0
        for i in outputs:
            for j in inputs:
                H_ij = H[:, i, j]
                mag = 20 * np.log10(np.abs(H_ij) + 1e-20)
                phase = np.angle(H_ij, deg=True)

                axes[0, plot_idx].semilogx(omega, mag, 'b-', linewidth=1.5)
                axes[0, plot_idx].set_title(f'输出{i+1} → 输入{j+1}',
                                           fontsize=11, fontweight='bold')
                axes[0, plot_idx].set_ylabel('幅值 (dB)', fontsize=10)
                axes[0, plot_idx].grid(True, alpha=0.3, which='both')

                axes[1, plot_idx].semilogx(omega, phase, 'r-', linewidth=1.5)
                axes[1, plot_idx].set_xlabel('频率 (rad/s)', fontsize=10)
                axes[1, plot_idx].set_ylabel('相位 (°)', fontsize=10)
                axes[1, plot_idx].grid(True, alpha=0.3, which='both')

                plot_idx += 1

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.002)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, axes

    def plot_singular_values(self, singular_values, normalize=True,
                             suggested_order=None, title='奇异值分布',
                             xlabel='阶数', ylabel='奇异值',
                             save_path=None):
        """
        绘制奇异值分布（用于阶数选择）

        参数:
            singular_values: 奇异值数组
            normalize: 是否归一化
            suggested_order: 建议的阶数标记
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            ax: matplotlib Axes对象
        """
        S = np.asarray(singular_values)
        if normalize:
            S = S / S[0]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.semilogy(range(1, len(S) + 1), S, 'bo-', markersize=5,
                   linewidth=1.5, label='奇异值')

        if suggested_order is not None:
            ax.axvline(x=suggested_order, color='r', linestyle='--',
                      linewidth=2, label=f'建议阶数: {suggested_order}')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, ax

    def plot_residuals(self, time, errors, labels=None,
                       title='残差分析', save_path=None):
        """
        绘制残差分析图（时域+直方图+自相关）

        参数:
            time: 时间向量
            errors: 残差矩阵 (n_samples, n_outputs)
            labels: 输出标签列表
            title: 图表标题
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            axes: matplotlib Axes对象 (n_outputs, 3)
        """
        errors = np.atleast_2d(errors)
        if errors.shape[0] == 1:
            errors = errors.T

        n_outputs = errors.shape[1]
        if labels is None:
            labels = [f'输出 {i+1}' for i in range(n_outputs)]

        fig, axes = plt.subplots(n_outputs, 3, figsize=(15, 4 * n_outputs),
                                 dpi=self.dpi)
        if n_outputs == 1:
            axes = axes.reshape(1, 3)

        for i in range(n_outputs):
            err = errors[:, i]

            axes[i, 0].plot(time, err, 'b-', linewidth=1)
            axes[i, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[i, 0].set_title(f'{labels[i]} - 时域', fontsize=12, fontweight='bold')
            axes[i, 0].set_xlabel('时间 (s)', fontsize=10)
            axes[i, 0].set_ylabel('残差', fontsize=10)
            axes[i, 0].grid(True, alpha=0.3)

            axes[i, 1].hist(err, bins=50, density=True, alpha=0.7, color='steelblue',
                          edgecolor='black', linewidth=0.5)
            x = np.linspace(err.min(), err.max(), 100)
            from scipy.stats import norm
            mu, std = norm.fit(err)
            axes[i, 1].plot(x, norm.pdf(x, mu, std), 'r--', linewidth=2,
                          label=f'N({mu:.2e}, {std:.2e})')
            axes[i, 1].set_title(f'{labels[i]} - 直方图', fontsize=12, fontweight='bold')
            axes[i, 1].set_xlabel('残差值', fontsize=10)
            axes[i, 1].set_ylabel('概率密度', fontsize=10)
            axes[i, 1].legend(fontsize=9)
            axes[i, 1].grid(True, alpha=0.3)

            from scipy.signal import correlate
            corr = correlate(err - err.mean(), err - err.mean(), mode='full')
            corr = corr[len(corr) // 2:]
            corr = corr / corr[0]
            lags = np.arange(len(corr))
            axes[i, 2].stem(lags[:40], corr[:40], 'b', basefmt='k-')
            axes[i, 2].axhline(y=0, color='k', linestyle='-', alpha=0.5)
            axes[i, 2].set_title(f'{labels[i]} - 自相关', fontsize=12, fontweight='bold')
            axes[i, 2].set_xlabel('滞后', fontsize=10)
            axes[i, 2].set_ylabel('自相关系数', fontsize=10)
            axes[i, 2].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.002)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, axes

    def plot_pole_zero(self, poles, zeros=None, unit_circle=True,
                       title='零极点图', save_path=None):
        """
        绘制零极点图

        参数:
            poles: 极点数组 (复数)
            zeros: 零点数组 (复数) 或 零点列表
            unit_circle: 是否绘制单位圆
            title: 图表标题
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            ax: matplotlib Axes对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if unit_circle:
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.5)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        if zeros is not None:
            if isinstance(zeros, list):
                for z in zeros:
                    ax.plot(np.real(z), np.imag(z), 'o', markersize=8,
                           markeredgecolor='b', markerfacecolor='none',
                           markeredgewidth=1.5, label='零点' if '零点' not in [
                               l.get_label() for l in ax.get_lines()] else "")
            else:
                ax.plot(np.real(zeros), np.imag(zeros), 'o', markersize=8,
                       markeredgecolor='b', markerfacecolor='none',
                       markeredgewidth=1.5, label='零点')

        stable = np.abs(poles) < 1.0
        unstable = np.abs(poles) >= 1.0

        if np.any(stable):
            ax.plot(np.real(poles[stable]), np.imag(poles[stable]), 'x',
                   markersize=8, markeredgecolor='g', markeredgewidth=1.5,
                   label='稳定极点')

        if np.any(unstable):
            ax.plot(np.real(poles[unstable]), np.imag(poles[unstable]), 'x',
                   markersize=8, markeredgecolor='r', markeredgewidth=1.5,
                   label='不稳定极点')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('实部', fontsize=12)
        ax.set_ylabel('虚部', fontsize=12)
        ax.axis('equal')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, ax

    def plot_stability(self, fn, zeta, title='频率-阻尼分布',
                       save_path=None):
        """
        绘制固有频率和阻尼比分布

        参数:
            fn: 固有频率数组 (Hz)
            zeta: 阻尼比数组
            title: 图表标题
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            ax: matplotlib Axes对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        colors = zeta.copy()
        colors[zeta < 0] = 0

        sc = ax.scatter(fn, zeta, c=colors, s=100, cmap='viridis_r',
                       edgecolors='black', linewidth=0.5, zorder=5)

        ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7,
                  label='稳定性边界 (ζ=0)')

        for i, (f, z) in enumerate(zip(fn, zeta)):
            ax.annotate(f'模{i+1}\n{f:.2f}Hz\nζ={z:.3f}',
                       xy=(f, z), xytext=(10, 10),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', fc='yellow',
                                alpha=0.7))

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('阻尼比 ζ', fontsize=11)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('固有频率 (Hz)', fontsize=12)
        ax.set_ylabel('阻尼比 ζ', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, ax

    def plot_frequency_comparison(self, omega, H1, H2, label1='模型1',
                                  label2='模型2', output_idx=0, input_idx=0,
                                  title='频率响应对比', save_path=None):
        """
        绘制两个模型的频率响应对

        参数:
            omega: 频率向量
            H1: 模型1的频率响应 (n_freq, n_outputs, n_inputs)
            H2: 模型2的频率响应
            label1: 模型1标签
            label2: 模型2标签
            output_idx: 输出索引
            input_idx: 输入索引
            title: 图表标题
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            axes: matplotlib Axes对象 (2, 1)
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi, sharex=True)

        H1_ij = H1[:, output_idx, input_idx]
        H2_ij = H2[:, output_idx, input_idx]

        mag1 = 20 * np.log10(np.abs(H1_ij) + 1e-20)
        mag2 = 20 * np.log10(np.abs(H2_ij) + 1e-20)
        phase1 = np.angle(H1_ij, deg=True)
        phase2 = np.angle(H2_ij, deg=True)

        axes[0].semilogx(omega, mag1, 'b-', linewidth=2, label=label1)
        axes[0].semilogx(omega, mag2, 'r--', linewidth=1.5, label=label2)
        axes[0].set_ylabel('幅值 (dB)', fontsize=11)
        axes[0].set_title(f'{title} - 输出{output_idx+1} 输入{input_idx+1}',
                         fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=11)
        axes[0].grid(True, alpha=0.3, which='both')

        axes[1].semilogx(omega, phase1, 'b-', linewidth=2, label=label1)
        axes[1].semilogx(omega, phase2, 'r--', linewidth=1.5, label=label2)
        axes[1].set_xlabel('频率 (rad/s)', fontsize=11)
        axes[1].set_ylabel('相位 (°)', fontsize=11)
        axes[1].legend(loc='best', fontsize=11)
        axes[1].grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, axes

    def plot_metrics_comparison(self, model_names, metrics_dict, metric='RMSE',
                                title='模型性能对比', save_path=None):
        """
        绘制多个模型的性能指标对比柱状图

        参数:
            model_names: 模型名称列表
            metrics_dict: 指标字典 {metric_name: [values]}
            metric: 要显示的指标名
            title: 图表标题
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            ax: matplotlib Axes对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        x = np.arange(len(model_names))
        width = 0.35

        values = metrics_dict[metric]

        bars = ax.bar(x, values, width, color='steelblue',
                     edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, ax

    def plot_coherence(self, freqs, coherence, outputs=None, inputs=None,
                       title='相干函数', save_path=None):
        """
        绘制相干函数图

        参数:
            freqs: 频率向量 (Hz)
            coherence: 相干函数矩阵 (n_freq, n_outputs, n_inputs)
            outputs: 输出索引列表
            inputs: 输入索引列表
            title: 图表标题
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            ax: matplotlib Axes对象
        """
        if outputs is None:
            outputs = list(range(coherence.shape[1]))
        if inputs is None:
            inputs = list(range(coherence.shape[2]))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for i in outputs:
            for j in inputs:
                ax.plot(freqs, coherence[:, i, j],
                       label=f'输出{i+1}-输入{j+1}', linewidth=1.5)

        ax.axhline(y=0.7, color='r', linestyle='--', linewidth=1, alpha=0.7,
                  label='阈值 (0.7)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('频率 (Hz)', fontsize=12)
        ax.set_ylabel('相干函数 γ²', fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, ax

    def plot_vg(self, velocities, dampings, frequencies=None, flutter_speed=None,
                title='V-g 图', legend_loc='best', save_path=None):
        """
        绘制V-g图（速度-阻尼曲线）和V-f图（速度-频率曲线）

        参数:
            velocities: 速度数组 (n_vel,)
            dampings: 阻尼数组 (n_vel, n_modes)
            frequencies: 频率数组 (n_vel, n_modes)，可选
            flutter_speed: 颤振临界速度，可选
            title: 图表标题
            legend_loc: 图例位置
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            axes: matplotlib Axes对象数组
        """
        n_vel, n_modes = dampings.shape

        if frequencies is not None:
            fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5),
                                    dpi=self.dpi, sharex=True)
        else:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            axes = [ax]

        colors = plt.cm.tab10(np.linspace(0, 1, n_modes))

        for mode in range(n_modes):
            damp_mode = dampings[:, mode]
            valid_idx = ~np.isnan(damp_mode)

            if np.sum(valid_idx) < 2:
                continue

            vel_valid = velocities[valid_idx]
            damp_valid = damp_mode[valid_idx]

            label = f'模态 {mode + 1}'
            if frequencies is not None:
                freq_mode = frequencies[:, mode]
                freq_valid = freq_mode[valid_idx]
                if len(freq_valid) > 0 and not np.all(np.isnan(freq_valid)):
                    mean_freq = np.nanmean(freq_valid)
                    label += f' ({mean_freq:.1f} Hz)'

            axes[0].plot(vel_valid, damp_valid, 'o-', color=colors[mode],
                        label=label, linewidth=1.5, markersize=4)

            if frequencies is not None:
                axes[1].plot(vel_valid, freq_valid, 'o-', color=colors[mode],
                            label=label, linewidth=1.5, markersize=4)

        axes[0].axhline(y=0, color='k', linestyle='-', linewidth=1.5, alpha=0.8)
        axes[0].axhspan(-0.1, 0, facecolor='red', alpha=0.1, label='不稳定区')

        if flutter_speed is not None:
            axes[0].axvline(x=flutter_speed, color='r', linestyle='--', linewidth=2,
                           label=f'颤振速度 {flutter_speed:.1f} m/s')
            if frequencies is not None:
                axes[1].axvline(x=flutter_speed, color='r', linestyle='--', linewidth=2)

        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].set_ylabel('阻尼比 g', fontsize=12)
        axes[0].legend(loc=legend_loc, fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=10)

        if frequencies is not None:
            axes[1].set_xlabel('速度 (m/s)', fontsize=12)
            axes[1].set_ylabel('频率 (Hz)', fontsize=12)
            axes[1].legend(loc=legend_loc, fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(labelsize=10)
        else:
            axes[0].set_xlabel('速度 (m/s)', fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, axes

    def plot_root_locus(self, roots_list, velocities=None, flutter_speed=None,
                        title='根轨迹图', xlim=None, ylim=None,
                        save_path=None):
        """
        绘制根轨迹图（s平面极点随速度变化）

        参数:
            roots_list: 极点列表，每个元素对应一个速度下的特征值数组
            velocities: 对应的速度数组，可选
            flutter_speed: 颤振临界速度，可选
            title: 图表标题
            xlim: x轴范围 [xmin, xmax]
            ylim: y轴范围 [ymin, ymax]
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            ax: matplotlib Axes对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.axvline(x=0, color='k', linestyle='-', linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1.5, alpha=0.8)
        ax.axvspan(-10, 0, facecolor='green', alpha=0.05, label='稳定区')
        ax.axvspan(0, 10, facecolor='red', alpha=0.05, label='不稳定区')

        n_vel = len(roots_list)
        colors = plt.cm.jet(np.linspace(0, 1, n_vel))

        for i, roots in enumerate(roots_list):
            if len(roots) == 0:
                continue

            real_parts = np.real(roots)
            imag_parts = np.imag(roots)

            valid_mask = ~np.isnan(real_parts) & ~np.isnan(imag_parts)
            if not np.any(valid_mask):
                continue

            real_parts = real_parts[valid_mask]
            imag_parts = imag_parts[valid_mask]

            label = None
            if velocities is not None and (i == 0 or i == n_vel - 1 or i % 10 == 0):
                label = f'V={velocities[i]:.1f} m/s'

            ax.scatter(real_parts, imag_parts, color=colors[i], s=30,
                      alpha=0.6, label=label, edgecolors='k', linewidths=0.5)

        if flutter_speed is not None:
            ax.scatter([], [], color='r', s=100, marker='*',
                      label=f'颤振 V={flutter_speed:.1f} m/s')

        ax.set_xlabel('实部 σ (1/s)', fontsize=12)
        ax.set_ylabel('虚部 ω (rad/s)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, ax

    def plot_damping_derivatives(self, velocities, damping_derivatives,
                                 title='阻尼导数曲线', save_path=None):
        """
        绘制阻尼导数曲线（阻尼随速度的变化率）

        参数:
            velocities: 速度数组
            damping_derivatives: 阻尼导数数组 (n_vel, n_modes)
            title: 图表标题
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            ax: matplotlib Axes对象
        """
        n_vel, n_modes = damping_derivatives.shape

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, n_modes))

        for mode in range(n_modes):
            deriv_mode = damping_derivatives[:, mode]
            valid_idx = ~np.isnan(deriv_mode)

            if np.sum(valid_idx) < 2:
                continue

            vel_valid = velocities[valid_idx]
            deriv_valid = deriv_mode[valid_idx]

            ax.plot(vel_valid, deriv_valid, 'o-', color=colors[mode],
                   label=f'模态 {mode + 1}', linewidth=1.5, markersize=4)

        ax.axhline(y=0, color='k', linestyle='-', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('速度 (m/s)', fontsize=12)
        ax.set_ylabel('阻尼导数 dg/dV', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, ax

    def plot_flutter_summary(self, flutter_results, title='颤振分析汇总',
                            save_path=None):
        """
        绘制颤振分析汇总图（V-g图 + V-f图 + 根轨迹）

        参数:
            flutter_results: 颤振分析结果字典
            title: 图表标题
            save_path: 保存路径

        返回:
            fig: matplotlib Figure对象
            axes: matplotlib Axes对象数组
        """
        velocities = flutter_results['velocities']
        dampings = flutter_results['dampings']
        frequencies = flutter_results['frequencies']
        flutter_speed = flutter_results.get('flutter_speed')

        fig = plt.figure(figsize=(self.figsize[0] * 1.5, self.figsize[1] * 2),
                        dpi=self.dpi)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, :])

        n_modes = dampings.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, n_modes))

        for mode in range(n_modes):
            damp_mode = dampings[:, mode]
            freq_mode = frequencies[:, mode]
            valid_idx = ~np.isnan(damp_mode)

            if np.sum(valid_idx) < 2:
                continue

            vel_valid = velocities[valid_idx]
            damp_valid = damp_mode[valid_idx]
            freq_valid = freq_mode[valid_idx]

            mean_freq = np.nanmean(freq_valid)
            label = f'模态 {mode + 1} ({mean_freq:.1f} Hz)'

            ax1.plot(vel_valid, damp_valid, 'o-', color=colors[mode],
                    label=label, linewidth=1.5, markersize=4)
            ax2.plot(vel_valid, freq_valid, 'o-', color=colors[mode],
                    label=label, linewidth=1.5, markersize=4)

        if 'root_locus' in flutter_results:
            roots = flutter_results['root_locus']['roots']
            n_vel = len(roots)
            colors_rl = plt.cm.jet(np.linspace(0, 1, n_vel))

            ax3.axvline(x=0, color='k', linestyle='-', linewidth=1.5, alpha=0.8)
            ax3.axhline(y=0, color='k', linestyle='-', linewidth=1.5, alpha=0.8)

            for i, rts in enumerate(roots):
                if len(rts) == 0:
                    continue
                real_parts = np.real(rts)
                imag_parts = np.imag(rts)
                valid = ~np.isnan(real_parts) & ~np.isnan(imag_parts)
                if np.any(valid):
                    ax3.scatter(real_parts[valid], imag_parts[valid],
                               color=colors_rl[i], s=20, alpha=0.5)

            ax3.set_xlabel('实部 σ (1/s)', fontsize=11)
            ax3.set_ylabel('虚部 ω (rad/s)', fontsize=11)
            ax3.set_title('根轨迹', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.axis('off')
            ax3.text(0.5, 0.5, '根轨迹数据不可用\n请先调用 compute_root_locus()',
                    ha='center', va='center', fontsize=12)

        ax1.axhline(y=0, color='k', linestyle='-', linewidth=1.5, alpha=0.8)
        ax1.axhspan(-0.1, 0, facecolor='red', alpha=0.1)

        if flutter_speed is not None:
            ax1.axvline(x=flutter_speed, color='r', linestyle='--', linewidth=2,
                       label=f'颤振 {flutter_speed:.1f} m/s')
            ax2.axvline(x=flutter_speed, color='r', linestyle='--', linewidth=2)

        ax1.set_ylabel('阻尼比 g', fontsize=11)
        ax1.set_title('V-g 图 (速度-阻尼)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('速度 (m/s)', fontsize=11)
        ax2.set_ylabel('频率 (Hz)', fontsize=11)
        ax2.set_title('V-f 图 (速度-频率)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        return fig, [ax1, ax2, ax3]

    def show_all(self):
        """显示所有打开的图形"""
        plt.show()

    def close_all(self):
        """关闭所有图形"""
        plt.close('all')
