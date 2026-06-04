"""
数据预处理模块

提供CFD/试验数据的加载、清洗、标准化和分割功能，
为系统辨识准备高质量的数据。
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """
    数据预处理类

    功能：
        - 加载多种格式的数据（CSV, TXT, NPY）
        - 数据清洗：去噪、插值、异常值处理
        - 数据标准化/归一化
        - 训练集/验证集分割
        - 构造Hankel矩阵所需的输入输出数据
    """

    def __init__(self, dt=None):
        """
        初始化数据预处理器

        参数:
            dt: 采样时间间隔 (s)，如果为None则从数据中推断
        """
        self.dt = dt
        self.scaler_input = None
        self.scaler_output = None
        self.raw_data = None
        self.processed_data = None
        self.input_labels = None
        self.output_labels = None

    def load_data(self, filepath, file_format='auto', **kwargs):
        """
        加载数据文件

        参数:
            filepath: 数据文件路径
            file_format: 文件格式 ('auto', 'csv', 'txt', 'npy', 'pkl')
            **kwargs: 传递给加载函数的额外参数

        返回:
            self: 链式调用
        """
        if file_format == 'auto':
            ext = filepath.split('.')[-1].lower()
            format_map = {'csv': 'csv', 'txt': 'txt', 'npy': 'npy', 'pkl': 'pkl', 'pickle': 'pkl'}
            file_format = format_map.get(ext, 'csv')

        if file_format in ['csv', 'txt']:
            delimiter = kwargs.get('delimiter', ',' if file_format == 'csv' else r'\s+')
            self.raw_data = pd.read_csv(filepath, delimiter=delimiter, **kwargs)
        elif file_format == 'npy':
            data = np.load(filepath)
            self.raw_data = pd.DataFrame(data)
        elif file_format == 'pkl':
            self.raw_data = pd.read_pickle(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")

        if self.dt is None and 'time' in self.raw_data.columns:
            time = self.raw_data['time'].values
            self.dt = np.mean(np.diff(time))

        return self

    def set_input_output(self, input_cols, output_cols):
        """
        设置输入和输出列

        参数:
            input_cols: 输入列名列表
            output_cols: 输出列名列表

        返回:
            self: 链式调用
        """
        self.input_labels = input_cols
        self.output_labels = output_cols
        return self

    def remove_outliers(self, method='iqr', threshold=1.5, columns=None):
        """
        移除异常值

        参数:
            method: 异常值检测方法 ('iqr', 'zscore')
            threshold: 异常值判定阈值
            columns: 要处理的列名列表，默认处理所有数值列

        返回:
            self: 链式调用
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")

        data = self.raw_data.copy()
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns

        if method == 'iqr':
            for col in columns:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                data = data[(data[col] >= lower) & (data[col] <= upper)]
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < threshold]
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")

        self.raw_data = data.reset_index(drop=True)
        return self

    def interpolate_missing(self, method='linear', **kwargs):
        """
        插值缺失值

        参数:
            method: 插值方法 ('linear', 'time', 'polynomial', 'spline')
            **kwargs: 传递给插值函数的额外参数

        返回:
            self: 链式调用
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")

        self.raw_data = self.raw_data.interpolate(method=method, **kwargs)
        self.raw_data = self.raw_data.fillna(method='bfill').fillna(method='ffill')
        return self

    def denoise(self, method='butterworth', **kwargs):
        """
        数据去噪

        参数:
            method: 去噪方法 ('butterworth', 'gaussian', 'wavelet', 'savgol')
            **kwargs: 传递给去噪函数的额外参数

        返回:
            self: 链式调用
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")

        data = self.raw_data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if method == 'butterworth':
            order = kwargs.get('order', 4)
            cutoff = kwargs.get('cutoff', 0.1)
            fs = 1.0 / self.dt if self.dt else 1.0
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            for col in numeric_cols:
                data[col] = signal.filtfilt(b, a, data[col].values)
        elif method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            for col in numeric_cols:
                data[col] = data[col].rolling(window=5, win_type='gaussian', center=True).mean(std=sigma)
        elif method == 'savgol':
            window_length = kwargs.get('window_length', 51)
            polyorder = kwargs.get('polyorder', 3)
            for col in numeric_cols:
                data[col] = signal.savgol_filter(data[col].values, window_length, polyorder)
        else:
            raise ValueError(f"不支持的去噪方法: {method}")

        data = data.fillna(method='bfill').fillna(method='ffill')
        self.raw_data = data
        return self

    def resample(self, new_dt, method='linear'):
        """
        重采样数据到新的时间步长

        参数:
            new_dt: 新的采样时间间隔
            method: 插值方法

        返回:
            self: 链式调用
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")

        if 'time' not in self.raw_data.columns:
            self.raw_data.insert(0, 'time', np.arange(len(self.raw_data)) * self.dt)

        old_time = self.raw_data['time'].values
        new_time = np.arange(old_time[0], old_time[-1], new_dt)

        new_data = pd.DataFrame({'time': new_time})
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns.drop('time')

        for col in numeric_cols:
            f = interpolate.interp1d(old_time, self.raw_data[col].values, kind=method, fill_value='extrapolate')
            new_data[col] = f(new_time)

        self.raw_data = new_data
        self.dt = new_dt
        return self

    def standardize(self, method='standard', fit_on_train=True, train_ratio=0.8):
        """
        标准化/归一化数据

        参数:
            method: 标准化方法 ('standard', 'minmax')
            fit_on_train: 是否仅在训练集上拟合标准化器
            train_ratio: 训练集比例

        返回:
            self: 链式调用
        """
        if self.raw_data is None or self.input_labels is None or self.output_labels is None:
            raise ValueError("请先加载数据并设置输入输出列")

        data = self.raw_data.copy()
        input_data = data[self.input_labels].values
        output_data = data[self.output_labels].values

        if fit_on_train:
            n_train = int(len(input_data) * train_ratio)
            input_train = input_data[:n_train]
            output_train = output_data[:n_train]
        else:
            input_train = input_data
            output_train = output_data

        if method == 'standard':
            self.scaler_input = StandardScaler()
            self.scaler_output = StandardScaler()
        elif method == 'minmax':
            self.scaler_input = MinMaxScaler()
            self.scaler_output = MinMaxScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")

        self.scaler_input.fit(input_train)
        self.scaler_output.fit(output_train)

        input_scaled = self.scaler_input.transform(input_data)
        output_scaled = self.scaler_output.transform(output_data)

        for i, col in enumerate(self.input_labels):
            data[col] = input_scaled[:, i]
        for i, col in enumerate(self.output_labels):
            data[col] = output_scaled[:, i]

        self.processed_data = data
        return self

    def inverse_transform(self, data, data_type='output'):
        """
        反标准化数据

        参数:
            data: 要反标准化的数据
            data_type: 'input' 或 'output'

        返回:
            反标准化后的数据
        """
        scaler = self.scaler_output if data_type == 'output' else self.scaler_input
        if scaler is None:
            return data
        return scaler.inverse_transform(data)

    def split_data(self, train_ratio=0.8, shuffle=False):
        """
        分割训练集和验证集

        参数:
            train_ratio: 训练集比例
            shuffle: 是否打乱数据

        返回:
            (train_data, val_data): 训练集和验证集
        """
        data = self.processed_data if self.processed_data is not None else self.raw_data
        if data is None:
            raise ValueError("请先加载数据")

        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        n_train = int(len(data) * train_ratio)
        train_data = data.iloc[:n_train]
        val_data = data.iloc[n_train:]

        return train_data, val_data

    def get_input_output(self, data=None):
        """
        获取输入输出数据数组

        参数:
            data: 输入的DataFrame，默认使用processed_data

        返回:
            (U, Y): 输入矩阵 (n_samples, n_inputs)，输出矩阵 (n_samples, n_outputs)
        """
        if data is None:
            data = self.processed_data if self.processed_data is not None else self.raw_data
        if data is None or self.input_labels is None or self.output_labels is None:
            raise ValueError("请先加载数据并设置输入输出列")

        U = data[self.input_labels].values
        Y = data[self.output_labels].values
        return U, Y

    def build_hankel_matrices(self, U, Y, block_rows):
        """
        构建Hankel矩阵（用于ERA算法）

        参数:
            U: 输入数据 (n_samples, n_inputs)
            Y: 输出数据 (n_samples, n_outputs)
            block_rows: Hankel矩阵的块行数

        返回:
            (Hankel_U, Hankel_Y): 输入和输出的Hankel矩阵
        """
        n_samples, n_inputs = U.shape
        n_outputs = Y.shape[1]
        n_cols = n_samples - 2 * block_rows + 1

        if n_cols <= 0:
            raise ValueError(f"block_rows太大，样本数不足。需要至少{2 * block_rows}个样本")

        Hankel_U = np.zeros((block_rows * n_inputs, n_cols))
        Hankel_Y = np.zeros((block_rows * n_outputs, n_cols))

        for i in range(block_rows):
            U_block = U[i:i + n_cols].T
            Y_block = Y[i:i + n_cols].T
            Hankel_U[i * n_inputs:(i + 1) * n_inputs, :] = U_block
            Hankel_Y[i * n_outputs:(i + 1) * n_outputs, :] = Y_block

        return Hankel_U, Hankel_Y

    def build_arx_data(self, U, Y, input_order, output_order):
        """
        构建ARX模型的回归矩阵

        参数:
            U: 输入数据 (n_samples, n_inputs)
            Y: 输出数据 (n_samples, n_outputs)
            input_order: 输入阶数
            output_order: 输出阶数

        返回:
            (Phi, Y_reg): 回归矩阵和输出向量
        """
        n_samples, n_inputs = U.shape
        n_outputs = Y.shape[1]
        max_order = max(input_order, output_order)
        n_reg = n_samples - max_order

        if n_reg <= 0:
            raise ValueError(f"阶数太大，样本数不足")

        Phi = np.zeros((n_reg, output_order * n_outputs + input_order * n_inputs))
        Y_reg = Y[max_order:]

        for i in range(output_order):
            Y_block = Y[max_order - i - 1:max_order - i - 1 + n_reg]
            Phi[:, i * n_outputs:(i + 1) * n_outputs] = Y_block

        for i in range(input_order):
            U_block = U[max_order - i - 1:max_order - i - 1 + n_reg]
            Phi[:, output_order * n_outputs + i * n_inputs:output_order * n_outputs + (i + 1) * n_inputs] = U_block

        return Phi, Y_reg

    def get_time_vector(self, data=None):
        """
        获取时间向量

        参数:
            data: 输入的DataFrame

        返回:
            time: 时间向量
        """
        if data is None:
            data = self.processed_data if self.processed_data is not None else self.raw_data
        if data is None:
            raise ValueError("请先加载数据")

        if 'time' in data.columns:
            return data['time'].values
        else:
            return np.arange(len(data)) * self.dt

    def summary(self):
        """
        打印数据摘要信息

        返回:
            self: 链式调用
        """
        if self.raw_data is not None:
            print("=" * 60)
            print("数据摘要")
            print("=" * 60)
            print(f"样本数: {len(self.raw_data)}")
            print(f"采样时间 dt: {self.dt:.6f} s")
            print(f"采样频率: {1.0/self.dt:.2f} Hz" if self.dt else "采样时间未设置")
            print(f"\n列名: {list(self.raw_data.columns)}")
            print(f"\n输入列: {self.input_labels}")
            print(f"输出列: {self.output_labels}")
            print(f"\n统计信息:")
            print(self.raw_data.describe())
            print("=" * 60)
        return self

    def compute_spectrum(self, data=None, fs=None, window='hann', nperseg=None):
        """
        计算信号的功率谱密度（PSD）

        参数:
            data: 输入数据，如为None则使用已加载的输入输出数据
            fs: 采样频率 (Hz)，如为None则使用dt计算
            window: 窗函数类型
            nperseg: 每段长度

        返回:
            freqs: 频率向量
            psd_dict: 各通道的PSD字典
        """
        if self.dt is None and fs is None:
            raise ValueError("请先设置采样时间dt或提供fs")

        fs = fs or 1.0 / self.dt
        if nperseg is None:
            nperseg = min(256, len(self.raw_data) // 4)

        if data is None:
            U, Y = self.get_input_output()
            data = np.hstack([U, Y])
            labels = (self.input_labels or [f'input_{i}' for i in range(U.shape[1])]) + \
                     (self.output_labels or [f'output_{i}' for i in range(Y.shape[1])])
        else:
            data = np.atleast_2d(data).T if data.ndim == 1 else data
            labels = [f'channel_{i}' for i in range(data.shape[1])]

        psd_dict = {}
        freqs = None

        for i, label in enumerate(labels):
            freqs, psd = signal.welch(data[:, i], fs=fs, window=window,
                                       nperseg=nperseg, scaling='density')
            psd_dict[label] = psd

        return freqs, psd_dict

    def estimate_max_frequency(self, data=None, threshold=0.05, method='psd'):
        """
        估计信号的最高有效频率

        参数:
            data: 输入数据
            threshold: 能量阈值（相对于峰值的比例）
            method: 估计方法 ('psd', 'fft')

        返回:
            max_freq: 最高有效频率 (Hz)
            freq_analysis: 详细分析结果
        """
        if self.dt is None:
            raise ValueError("请先设置采样时间dt")

        fs = 1.0 / self.dt

        if data is None:
            U, Y = self.get_input_output()
            data = np.hstack([U, Y])

        data = np.atleast_2d(data).T if data.ndim == 1 else data

        max_freqs = []
        freq_analysis = {'fs': fs, 'nyquist_freq': fs / 2}

        for i in range(data.shape[1]):
            if method == 'psd':
                freqs, psd = signal.welch(data[:, i], fs=fs, nperseg=256)
                psd_norm = psd / np.max(psd)
                significant_indices = np.where(psd_norm > threshold)[0]
                if len(significant_indices) > 0:
                    max_f = freqs[significant_indices[-1]]
                else:
                    max_f = 0.0
            else:
                n = len(data[:, i])
                fft_vals = np.abs(np.fft.fft(data[:, i]))
                freqs = np.fft.fftfreq(n, 1 / fs)
                positive_mask = freqs >= 0
                freqs_pos = freqs[positive_mask]
                fft_pos = fft_vals[positive_mask]
                fft_norm = fft_pos / np.max(fft_pos)
                significant_indices = np.where(fft_norm > threshold)[0]
                if len(significant_indices) > 0:
                    max_f = freqs_pos[significant_indices[-1]]
                else:
                    max_f = 0.0

            max_freqs.append(max_f)
            freq_analysis[f'channel_{i}'] = max_f

        max_freq = max(max_freqs)
        freq_analysis['max_frequency'] = max_freq
        freq_analysis['nyquist_ratio'] = max_freq / (fs / 2) if fs > 0 else 0

        return max_freq, freq_analysis

    def check_sampling_requirements(self, max_freq=None, min_oversampling=5.0):
        """
        检查采样频率是否满足系统辨识要求

        辨识理论要求：
        - 奈奎斯特采样定理: fs > 2 * f_max
        - 良好辨识通常需要: fs > 5 * f_max

        参数:
            max_freq: 信号最高频率 (Hz)，如为None则自动估计
            min_oversampling: 最小过采样倍数（推荐5~10倍）

        返回:
            is_valid: 是否满足要求
            analysis: 分析结果字典
        """
        if self.dt is None:
            raise ValueError("请先设置采样时间dt")

        fs = 1.0 / self.dt
        nyquist_freq = fs / 2

        if max_freq is None:
            max_freq, _ = self.estimate_max_frequency()

        analysis = {
            'fs': fs,
            'dt': self.dt,
            'nyquist_frequency': nyquist_freq,
            'signal_max_frequency': max_freq,
            'min_required_fs': 2.0 * max_freq,
            'recommended_fs': min_oversampling * max_freq,
            'nyquist_satisfied': fs > 2.0 * max_freq,
            'recommendation_satisfied': fs >= min_oversampling * max_freq,
            'oversampling_ratio': fs / max_freq if max_freq > 0 else float('inf')
        }

        is_valid = analysis['nyquist_satisfied']

        print("\n" + "=" * 60)
        print("采样频率诊断")
        print("=" * 60)
        print(f"采样频率: {fs:.2f} Hz")
        print(f"奈奎斯特频率: {nyquist_freq:.2f} Hz")
        print(f"信号最高频率: {max_freq:.2f} Hz")
        print(f"最小采样频率要求: {2.0 * max_freq:.2f} Hz")
        print(f"推荐采样频率 ({min_oversampling}x): {min_oversampling * max_freq:.2f} Hz")
        print(f"过采样比: {analysis['oversampling_ratio']:.1f}x")
        print("-" * 60)

        if is_valid:
            print("✓ 满足奈奎斯特采样定理")
        else:
            print("✗ 不满足奈奎斯特采样定理！可能发生混叠！")
            print("  建议：降低信号最高频率或提高采样频率")

        if analysis['recommendation_satisfied']:
            print("✓ 满足辨识推荐采样频率")
        else:
            print("⚠ 建议使用更高的采样频率以获得更好的辨识结果")

        if analysis['oversampling_ratio'] < 3:
            print("⚠ 警告：过采样比较低，高频辨识精度可能不足")
        print("=" * 60 + "\n")

        return is_valid, analysis

    def antialiasing_filter(self, cutoff_freq, filter_order=4, ftype='butter', data=None):
        """
        应用抗混叠滤波器

        参数:
            cutoff_freq: 截止频率 (Hz)
            filter_order: 滤波器阶数
            ftype: 滤波器类型 ('butter', 'cheby1', 'cheby2', 'ellip')
            data: 要滤波的数据，如为None则处理输入输出

        返回:
            self: 链式调用（data为None时）或滤波后的数据
        """
        if self.dt is None:
            raise ValueError("请先设置采样时间dt")

        fs = 1.0 / self.dt
        nyquist = fs / 2

        if cutoff_freq >= nyquist:
            raise ValueError(f"截止频率({cutoff_freq}Hz)必须小于奈奎斯特频率({nyquist}Hz)")

        normal_cutoff = cutoff_freq / nyquist

        if ftype == 'butter':
            b, a = signal.butter(filter_order, normal_cutoff, btype='low', analog=False)
        elif ftype == 'cheby1':
            b, a = signal.cheby1(filter_order, 1, normal_cutoff, btype='low')
        elif ftype == 'cheby2':
            b, a = signal.cheby2(filter_order, 20, normal_cutoff, btype='low')
        elif ftype == 'ellip':
            b, a = signal.ellip(filter_order, 1, 20, normal_cutoff, btype='low')
        else:
            raise ValueError(f"不支持的滤波器类型: {ftype}")

        if data is not None:
            data = np.atleast_2d(data).T if data.ndim == 1 else data
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
            return filtered_data
        else:
            if self.processed_data is None:
                self.processed_data = self.raw_data.copy()

            for col in self.input_labels + self.output_labels:
                if col in self.processed_data.columns:
                    self.processed_data[col] = signal.filtfilt(
                        b, a, self.processed_data[col].values
                    )

            return self

    def recommend_identification_params(self, max_freq=None):
        """
        基于信号频率特性推荐辨识参数

        参数:
            max_freq: 信号最高频率，如为None则自动估计

        返回:
            recommendations: 推荐参数字典
        """
        if self.dt is None:
            raise ValueError("请先设置采样时间dt")

        fs = 1.0 / self.dt

        if max_freq is None:
            max_freq, _ = self.estimate_max_frequency()

        fn_nyquist = fs / 2
        fn_max = min(max_freq * 1.2, fn_nyquist * 0.8)

        recommendations = {
            'sampling_frequency': fs,
            'signal_max_frequency': max_freq,
            'effective_frequency_range': [0, fn_max],
            'recommended_model_order_range': [int(fn_max * 0.5), int(fn_max * 2.0)],
            'era_block_rows': min(max(20, int(fs / fn_max) * 2), 100),
            'arx_time_constant': 1.0 / (2 * np.pi * max_freq) if max_freq > 0 else 1.0,
            'should_filter': max_freq > fn_nyquist * 0.5,
            'recommended_cutoff': fn_nyquist * 0.4
        }

        print("\n" + "=" * 60)
        print("辨识参数推荐")
        print("=" * 60)
        print(f"有效辨识频率范围: 0 - {fn_max:.1f} Hz")
        print(f"推荐模型阶数范围: {recommendations['recommended_model_order_range']}")
        print(f"推荐ERA Hankel块行数: {recommendations['era_block_rows']}")
        if recommendations['should_filter']:
            print(f"⚠ 建议先进行抗混叠滤波，截止频率: {recommendations['recommended_cutoff']:.1f} Hz")
        print("=" * 60 + "\n")

        return recommendations
