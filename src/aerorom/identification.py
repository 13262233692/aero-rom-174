"""
系统辨识算法模块

提供多种系统辨识算法，用于从CFD或试验数据中辨识气动力状态空间模型。

主要算法：
    - ERA (Eigensystem Realization Algorithm): 特征系统实现算法
    - ARX (AutoRegressive with eXogenous inputs): 带外生输入的自回归模型
    - StateSpaceModel: 通用状态空间模型类
"""

import numpy as np
import scipy
from scipy.linalg import svd, lstsq, solve_discrete_lyapunov
from scipy import signal
import warnings


def estimate_max_frequency(data, fs, threshold=0.05):
    """
    估计数据的最高有效频率

    参数:
        data: 输入数据 (n_samples,) 或 (n_samples, n_channels)
        fs: 采样频率 (Hz)
        threshold: PSD能量阈值（相对于峰值的比例）

    返回:
        max_freq: 最高有效频率 (Hz)
    """
    data = np.atleast_2d(data).T if data.ndim == 1 else data
    max_freqs = []

    for i in range(data.shape[1]):
        try:
            nperseg = min(256, len(data) // 4)
            if nperseg < 64:
                nperseg = min(64, len(data))
            freqs, psd = signal.welch(data[:, i], fs=fs, nperseg=nperseg)
            psd_max = np.max(psd)
            if psd_max > 0:
                psd_norm = psd / psd_max
                significant = np.where(psd_norm > threshold)[0]
                if len(significant) > 0:
                    max_freqs.append(freqs[significant[-1]])
        except:
            pass

    return max(max_freqs) if max_freqs else 0.0


def check_frequency_characteristics(Y, U, dt, method_name='Identification'):
    """
    检查输入输出数据的频率特性，验证是否满足辨识要求

    参数:
        Y: 输出数据 (n_samples, n_outputs)
        U: 输入数据 (n_samples, n_inputs)
        dt: 采样时间 (s)
        method_name: 辨识方法名称

    返回:
        freq_info: 频率信息字典
    """
    fs = 1.0 / dt if dt else None
    if fs is None:
        warnings.warn("采样时间dt未设置，无法进行频率特性检查")
        return {'valid': False}

    nyquist_freq = fs / 2

    max_freq_U = estimate_max_frequency(U, fs)
    max_freq_Y = estimate_max_frequency(Y, fs)
    max_freq = max(max_freq_U, max_freq_Y)

    freq_info = {
        'fs': fs,
        'nyquist_frequency': nyquist_freq,
        'max_frequency_U': max_freq_U,
        'max_frequency_Y': max_freq_Y,
        'max_frequency': max_freq,
        'oversampling_ratio': fs / max_freq if max_freq > 0 else float('inf'),
        'warnings': []
    }

    if max_freq > nyquist_freq * 0.5:
        warning_msg = (
            f"信号最高频率({max_freq:.1f}Hz)接近奈奎斯特频率({nyquist_freq:.1f}Hz)！\n"
            f"  这可能导致高频段辨识失真。建议：\n"
            f"  1. 提高采样频率(当前 {fs:.1f}Hz)\n"
            f"  2. 使用抗混叠滤波器降低信号带宽\n"
            f"  3. 在预处理模块中调用 antialiasing_filter() 进行滤波"
        )
        warnings.warn(f"\n[{method_name}] 警告：{warning_msg}")
        freq_info['warnings'].append('high_frequency_content')

    oversampling_ratio = fs / max_freq if max_freq > 0 else float('inf')
    if oversampling_ratio < 5:
        warning_msg = (
            f"过采样比({oversampling_ratio:.1f}x)较低！\n"
            f"  系统辨识通常建议5~10倍过采样以保证辨识精度。\n"
            f"  高频段辨识结果可能不可靠。"
        )
        warnings.warn(f"\n[{method_name}] 警告：{warning_msg}")
        freq_info['warnings'].append('low_oversampling')

    return freq_info


class StateSpaceModel:
    """
    离散时间状态空间模型类

    模型形式:
        x(k+1) = A x(k) + B u(k)
        y(k)   = C x(k) + D u(k)

    其中:
        x ∈ R^n  - 状态向量
        u ∈ R^p  - 输入向量
        y ∈ R^m  - 输出向量
    """

    def __init__(self, A=None, B=None, C=None, D=None, dt=1.0):
        """
        初始化状态空间模型

        参数:
            A: 状态矩阵 (n, n)
            B: 输入矩阵 (n, p)
            C: 输出矩阵 (m, n)
            D: 直馈矩阵 (m, p)
            dt: 采样时间间隔
        """
        self.A = np.array(A) if A is not None else None
        self.B = np.array(B) if B is not None else None
        self.C = np.array(C) if C is not None else None
        self.D = np.array(D) if D is not None else None
        self.dt = dt
        self.name = 'StateSpaceModel'
        self.info = {}

    @property
    def n_states(self):
        """状态维度"""
        return self.A.shape[0] if self.A is not None else 0

    @property
    def n_inputs(self):
        """输入维度"""
        return self.B.shape[1] if self.B is not None else 0

    @property
    def n_outputs(self):
        """输出维度"""
        return self.C.shape[0] if self.C is not None else 0

    def simulate(self, U, x0=None):
        """
        仿真模型输出

        参数:
            U: 输入序列 (n_samples, n_inputs)
            x0: 初始状态 (n_states,)，默认零初始状态

        返回:
            Y: 输出序列 (n_samples, n_outputs)
            X: 状态序列 (n_samples, n_states)
        """
        if self.A is None:
            raise ValueError("模型未辨识，请先辨识模型")

        U = np.atleast_2d(U)
        if U.ndim == 1:
            U = U.reshape(-1, 1)

        n_samples = U.shape[0]
        n_states = self.n_states
        n_outputs = self.n_outputs

        X = np.zeros((n_samples, n_states))
        Y = np.zeros((n_samples, n_outputs))

        if x0 is not None:
            X[0] = x0

        for k in range(n_samples):
            Y[k] = self.C @ X[k] + self.D @ U[k]
            if k < n_samples - 1:
                X[k + 1] = self.A @ X[k] + self.B @ U[k]

        return Y, X

    def frequency_response(self, omega=None):
        """
        计算频率响应函数

        参数:
            omega: 频率点向量 (rad/s)，默认自动生成

        返回:
            omega: 频率点向量
            H: 频率响应矩阵 (n_freq, n_outputs, n_inputs)
        """
        if self.A is None:
            raise ValueError("模型未辨识，请先辨识模型")

        if omega is None:
            omega = np.logspace(-2, np.log10(np.pi / self.dt), 100)

        I = np.eye(self.n_states)
        H = np.zeros((len(omega), self.n_outputs, self.n_inputs), dtype=complex)

        for i, w in enumerate(omega):
            z = np.exp(1j * w * self.dt)
            H[i] = self.C @ np.linalg.solve(z * I - self.A, self.B) + self.D

        return omega, H

    def eigenvalues(self):
        """
        计算系统特征值

        返回:
            eigenvalues: 系统特征值（离散域）
        """
        if self.A is None:
            raise ValueError("模型未辨识，请先辨识模型")
        return np.linalg.eigvals(self.A)

    def natural_frequencies_damping(self):
        """
        计算固有频率和阻尼比

        返回:
            fn: 固有频率 (Hz)
            zeta: 阻尼比
        """
        eig = self.eigenvalues()
        eps = 1e-20
        abs_eig = np.abs(eig) + eps
        sigma = np.log(abs_eig) / self.dt
        omega_d = np.abs(np.angle(eig)) / self.dt
        omega_n = np.sqrt(sigma**2 + omega_d**2 + eps)
        zeta = np.where(omega_n > eps, -sigma / omega_n, 0.0)
        fn = omega_n / (2 * np.pi)
        return fn, zeta

    def to_continuous(self, method='zoh'):
        """
        转换为连续时间模型

        参数:
            method: 转换方法 ('zoh', 'bilinear')

        返回:
            Ac, Bc, Cc, Dc: 连续时间系统矩阵
        """
        if method == 'zoh':
            M = np.zeros((self.n_states + self.n_inputs, self.n_states + self.n_inputs))
            M[:self.n_states, :self.n_states] = self.A
            M[:self.n_states, self.n_states:] = self.B
            M[self.n_states:, self.n_states:] = np.eye(self.n_inputs)

            Ms = scipy.linalg.logm(M) / self.dt
            Ac = Ms[:self.n_states, :self.n_states]
            Bc = Ms[:self.n_states, self.n_states:]
        elif method == 'bilinear':
            I = np.eye(self.n_states)
            Ac = (2 / self.dt) * (self.A - I) @ np.linalg.inv(self.A + I)
            Bc = np.sqrt(2 / self.dt) * np.linalg.inv(self.A + I) @ self.B
        else:
            raise ValueError(f"不支持的转换方法: {method}")

        return Ac, Bc, self.C.copy(), self.D.copy()

    def reduce_order(self, n_reduced):
        """
        模型降阶（平衡截断法）

        参数:
            n_reduced: 降阶后的状态数

        返回:
            reduced_model: 降阶后的状态空间模型
        """
        if self.A is None:
            raise ValueError("模型未辨识，请先辨识模型")

        if n_reduced >= self.n_states:
            warnings.warn("降阶数大于等于原阶数，返回原模型")
            return self

        A = self.A
        B = self.B
        C = self.C
        D = self.D

        Wc = solve_discrete_lyapunov(A, B @ B.T)
        Wo = solve_discrete_lyapunov(A.T, C.T @ C)

        L = np.linalg.cholesky(Wc)
        U, S, Vt = svd(L.T @ Wo @ L)
        Sigma = np.diag(np.sqrt(S))

        T = L @ U @ np.linalg.inv(Sigma)
        T_inv = Sigma @ U.T @ np.linalg.inv(L)

        Ar = T_inv[:n_reduced] @ A @ T[:, :n_reduced]
        Br = T_inv[:n_reduced] @ B
        Cr = C @ T[:, :n_reduced]
        Dr = D

        reduced_model = StateSpaceModel(Ar, Br, Cr, Dr, self.dt)
        reduced_model.name = f'Reduced_{self.name}'
        return reduced_model

    def summary(self):
        """打印模型摘要"""
        print("=" * 60)
        print(f"模型: {self.name}")
        print("=" * 60)
        print(f"状态数: {self.n_states}")
        print(f"输入数: {self.n_inputs}")
        print(f"输出数: {self.n_outputs}")
        print(f"采样时间: {self.dt:.6f} s")
        if self.A is not None:
            fn, zeta = self.natural_frequencies_damping()
            print(f"\n固有频率 (Hz): {fn}")
            print(f"阻尼比: {zeta}")
        print("=" * 60)
        return self


class ERA:
    """
    特征系统实现算法 (Eigensystem Realization Algorithm)

    基于脉冲响应数据辨识离散时间状态空间模型。
    """

    def __init__(self, dt=1.0):
        """
        初始化ERA辨识器

        参数:
            dt: 采样时间间隔
        """
        self.dt = dt
        self.model = None
        self.singular_values = None

    def identify(self, Y, U=None, n_states=None, block_rows=None, tolerance=1e-6,
                 check_frequency=True):
        """
        辨识状态空间模型

        参数:
            Y: 输出数据 (n_samples, n_outputs) 或 脉冲响应 (n_samples, n_outputs, n_inputs)
            U: 输入数据，如非脉冲响应则需提供 (n_samples, n_inputs)
            n_states: 模型阶数，如为None则自动选择
            block_rows: Hankel矩阵块行数，默认n_samples//3
            tolerance: 奇异值截断容差
            check_frequency: 是否进行频率特性检查

        返回:
            model: 辨识得到的状态空间模型
        """
        Y = np.array(Y)

        if U is not None:
            U = np.array(U)
            if Y.ndim != 2:
                raise ValueError("当提供U时，Y的维度应为2 (n_samples, n_outputs)")
            n_samples, n_outputs = Y.shape
            n_inputs = U.shape[1]

            if check_frequency:
                check_frequency_characteristics(Y, U, self.dt, 'ERA')

            Y = self._deconvolve(Y, U)
        else:
            if Y.ndim == 2:
                n_samples, n_outputs = Y.shape
                n_inputs = 1
                Y = Y.reshape(n_samples, n_outputs, n_inputs)
            elif Y.ndim == 3:
                n_samples, n_outputs, n_inputs = Y.shape
            else:
                raise ValueError("Y的维度应为2或3")

        if block_rows is None:
            block_rows = min(n_samples // 3, 50)

        block_cols = n_samples - 2 * block_rows + 1
        if block_cols <= 0:
            block_rows = min(n_samples // 3, 20)
            block_cols = n_samples - 2 * block_rows + 1

        H = np.zeros((block_rows * n_outputs, block_cols * n_inputs))
        H2 = np.zeros((block_rows * n_outputs, block_cols * n_inputs))

        for i in range(block_rows):
            for j in range(block_cols):
                idx = i + j
                if idx < n_samples:
                    H_block = Y[idx]
                    H[i * n_outputs:(i + 1) * n_outputs, j * n_inputs:(j + 1) * n_inputs] = H_block
                if idx + 1 < n_samples:
                    H2_block = Y[idx + 1]
                    H2[i * n_outputs:(i + 1) * n_outputs, j * n_inputs:(j + 1) * n_inputs] = H2_block

        U_svd, S_svd, Vt_svd = svd(H, full_matrices=False)
        self.singular_values = S_svd

        if n_states is None:
            S_normalized = S_svd / S_svd[0]
            n_states = np.sum(S_normalized > tolerance)
            n_states = max(1, min(n_states, len(S_svd) - 1))

        if n_states > len(S_svd):
            warnings.warn(f"请求的阶数{n_states}大于可用秩{len(S_svd)}，已调整")
            n_states = len(S_svd)

        S_sqrt = np.sqrt(S_svd[:n_states])
        S_inv = np.diag(1.0 / S_sqrt)
        U_n = U_svd[:, :n_states]
        V_n = Vt_svd[:n_states, :].T

        P = U_n @ S_inv
        Q = S_inv @ V_n.T

        A = P.T @ H2 @ Q.T
        B = Q[:, :n_inputs]
        C = P[:n_outputs, :]
        D = Y[0].copy() if n_samples > 0 else np.zeros((n_outputs, n_inputs))

        self.model = StateSpaceModel(A, B, C, D, self.dt)
        self.model.name = f'ERA_{n_states}states'
        self.model.info['singular_values'] = S_svd
        return self.model

    def _deconvolve(self, Y, U):
        """
        从任意输入输出数据计算脉冲响应

        参数:
            Y: 输出数据 (n_samples, n_outputs)
            U: 输入数据 (n_samples, n_inputs)

        返回:
            impulse_response: 脉冲响应 (n_samples, n_outputs, n_inputs)
        """
        n_samples, n_outputs = Y.shape
        n_inputs = U.shape[1]

        impulse_response = np.zeros((n_samples, n_outputs, n_inputs))

        for i in range(n_outputs):
            for j in range(n_inputs):
                yi = Y[:, i]
                uj = U[:, j]
                hi, _ = deconvolution(yi, uj, self.dt)
                n_h = len(hi)
                if n_h >= n_samples:
                    impulse_response[:, i, j] = hi[:n_samples]
                else:
                    impulse_response[:n_h, i, j] = hi

        return impulse_response

    def select_order(self, max_states=50, plot=False):
        """
        基于奇异值选择模型阶数

        参数:
            max_states: 最大考虑的阶数
            plot: 是否绘制奇异值图

        返回:
            suggested_order: 建议的模型阶数
        """
        if self.singular_values is None:
            raise ValueError("请先调用identify方法或提供脉冲响应数据")

        S = self.singular_values[:max_states]
        S_normalized = S / S[0]

        dS = np.diff(np.log10(S_normalized + 1e-20))
        elbow_idx = np.argmin(dS) + 1

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.semilogy(S_normalized, 'bo-', markersize=4)
            plt.axvline(elbow_idx, color='r', linestyle='--', label=f'建议阶数: {elbow_idx}')
            plt.xlabel('阶数')
            plt.ylabel('归一化奇异值')
            plt.title('奇异值分布图')
            plt.legend()
            plt.grid(True)
            plt.show()

        return elbow_idx


class ARX:
    """
    ARX模型辨识 (AutoRegressive with eXogenous inputs)

    模型形式:
        y(k) + a1*y(k-1) + ... + ana*y(k-na) = b1*u(k-1) + ... + bnb*u(k-nb) + e(k)
    """

    def __init__(self, dt=1.0):
        """
        初始化ARX辨识器

        参数:
            dt: 采样时间间隔
        """
        self.dt = dt
        self.model = None
        self.theta = None

    def identify(self, U, Y, input_order, output_order, check_frequency=True):
        """
        辨识ARX模型

        参数:
            U: 输入数据 (n_samples, n_inputs)
            Y: 输出数据 (n_samples, n_outputs)
            input_order: 输入阶数 (nb)
            output_order: 输出阶数 (na)
            check_frequency: 是否进行频率特性检查

        返回:
            model: 辨识得到的状态空间模型
        """
        U = np.atleast_2d(U)
        Y = np.atleast_2d(Y)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n_samples, n_inputs = U.shape
        n_outputs = Y.shape[1]

        if check_frequency:
            check_frequency_characteristics(Y, U, self.dt, 'ARX')

        max_order = max(input_order, output_order)
        n_reg = n_samples - max_order

        if n_reg <= 0:
            raise ValueError(f"样本数不足，需要至少{max_order + 1}个样本")

        n_params = output_order * n_outputs + input_order * n_inputs
        Theta = np.zeros((n_outputs, n_params))

        for i in range(n_outputs):
            Phi_i = np.zeros((n_reg, n_params))
            y_i = Y[max_order:, i]

            for j in range(output_order):
                Phi_i[:, j * n_outputs:(j + 1) * n_outputs] = -Y[max_order - j - 1:max_order - j - 1 + n_reg]

            for j in range(input_order):
                Phi_i[:, output_order * n_outputs + j * n_inputs:output_order * n_outputs + (j + 1) * n_inputs] = U[max_order - j - 1:max_order - j - 1 + n_reg]

            theta_i, residuals, rank, sv = lstsq(Phi_i, y_i)
            Theta[i] = theta_i

        self.theta = Theta

        A, B, C, D = self._arx_to_ss(Theta, n_inputs, n_outputs, input_order, output_order)
        self.model = StateSpaceModel(A, B, C, D, self.dt)
        self.model.name = f'ARX_{input_order}_{output_order}'
        return self.model

    def _arx_to_ss(self, Theta, n_inputs, n_outputs, nb, na):
        """
        将ARX参数转换为状态空间形式（可控标准型）

        参数:
            Theta: ARX参数矩阵 (n_outputs, n_params)
            n_inputs: 输入维度
            n_outputs: 输出维度
            nb: 输入阶数
            na: 输出阶数

        返回:
            A, B, C, D: 状态空间矩阵
        """
        n_states = na * n_outputs + nb * n_inputs

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_inputs))
        C = np.zeros((n_outputs, n_states))
        D = np.zeros((n_outputs, n_inputs))

        for i in range(n_outputs):
            theta = Theta[i]

            a_params = theta[:na * n_outputs].reshape(na, n_outputs)
            b_params = theta[na * n_outputs:].reshape(nb, n_inputs)

            for j in range(na):
                if j == 0:
                    A[j * n_outputs:(j + 1) * n_outputs, j * n_outputs:(j + 1) * n_outputs] = -a_params[j]
                else:
                    A[j * n_outputs:(j + 1) * n_outputs, j * n_outputs:(j + 1) * n_outputs] = -a_params[j]
                    A[j * n_outputs:(j + 1) * n_outputs, (j - 1) * n_outputs:j * n_outputs] = np.eye(n_outputs)

            for j in range(nb):
                B[na * n_outputs + j * n_inputs:na * n_outputs + (j + 1) * n_inputs, :] = np.eye(n_inputs)
                if j == 0:
                    B[:n_outputs, :] += b_params[j]
                else:
                    B[j * n_outputs:(j + 1) * n_outputs, :] += b_params[j]

            C[i, :n_outputs] = 1.0

        return A, B, C, D

    def simulate(self, U, y0=None):
        """
        仿真ARX模型

        参数:
            U: 输入序列 (n_samples, n_inputs)
            y0: 初始输出条件

        返回:
            Y: 输出序列 (n_samples, n_outputs)
        """
        if self.model is None:
            raise ValueError("模型未辨识，请先辨识模型")
        return self.model.simulate(U)


def deconvolution(y, u, dt, method='wiener', **kwargs):
    """
    解卷积计算脉冲响应

    参数:
        y: 输出信号
        u: 输入信号
        dt: 采样时间
        method: 方法 ('wiener', 'least_squares')

    返回:
        h: 脉冲响应
        residuals: 残差
    """
    y = np.asarray(y)
    u = np.asarray(u)
    N = len(y)

    if method == 'wiener':
        Yf = np.fft.fft(y)
        Uf = np.fft.fft(u)
        snr = kwargs.get('snr', 100.0)
        Hf = Yf * np.conj(Uf) / (Uf * np.conj(Uf) + 1.0 / snr)
        h = np.fft.ifft(Hf).real
        residuals = y - np.convolve(u, h, mode='same') * dt
    elif method == 'least_squares':
        n = kwargs.get('n', min(100, N // 2))
        U_matrix = np.zeros((N, n))
        for i in range(n):
            U_matrix[i:, i] = u[:N - i]
        h, residuals, rank, sv = lstsq(U_matrix, y)
        residuals = y - U_matrix @ h
    else:
        raise ValueError(f"不支持的解卷积方法: {method}")

    return h, residuals
