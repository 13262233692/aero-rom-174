"""
模型导出模块

提供将辨识得到的降阶模型导出为多种格式的功能，
包括MATLAB格式、Python pickle格式、JSON格式、
Simulink模型、以及FEM/NASTRAN格式等，方便用于颤振分析。
"""

import numpy as np
import json
import pickle
import os
from datetime import datetime


class ModelExporter:
    """
    模型导出类

    支持多种导出格式：
    - Python pickle (.pkl)
    - MATLAB (.mat)
    - JSON (.json)
    - NumPy (.npz)
    - CSV (.csv)
    - Simulink S函数 (.m)
    - Abaqus/NASTRAN格式
    """

    def __init__(self, model=None):
        """
        初始化模型导出器

        参数:
            model: 状态空间模型
        """
        self.model = model

    def set_model(self, model):
        """
        设置待导出的模型

        参数:
            model: 状态空间模型

        返回:
            self: 链式调用
        """
        self.model = model
        return self

    def export_pickle(self, filepath, include_metadata=True):
        """
        导出为Python pickle格式

        参数:
            filepath: 输出文件路径
            include_metadata: 是否包含元数据

        返回:
            self: 链式调用
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        data = {
            'A': self.model.A,
            'B': self.model.B,
            'C': self.model.C,
            'D': self.model.D,
            'dt': self.model.dt,
            'name': self.model.name,
            'n_states': self.model.n_states,
            'n_inputs': self.model.n_inputs,
            'n_outputs': self.model.n_outputs,
        }

        if include_metadata:
            data['metadata'] = self._get_metadata()

        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return self

    def export_numpy(self, filepath, include_metadata=True):
        """
        导出为NumPy .npz格式

        参数:
            filepath: 输出文件路径
            include_metadata: 是否包含元数据

        返回:
            self: 链式调用
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        data = {
            'A': self.model.A,
            'B': self.model.B,
            'C': self.model.C,
            'D': self.model.D,
            'dt': np.array([self.model.dt]),
            'n_states': np.array([self.model.n_states]),
            'n_inputs': np.array([self.model.n_inputs]),
            'n_outputs': np.array([self.model.n_outputs]),
        }

        if include_metadata:
            fn, zeta = self.model.natural_frequencies_damping()
            eig = self.model.eigenvalues()
            data.update({
                'natural_frequencies': fn,
                'damping_ratios': zeta,
                'eigenvalues': eig,
            })

        if not filepath.endswith('.npz'):
            filepath += '.npz'

        np.savez(filepath, **data)
        return self

    def export_matlab(self, filepath, variable_name='rom_model'):
        """
        导出为MATLAB .mat格式

        参数:
            filepath: 输出文件路径
            variable_name: MATLAB工作区变量名

        返回:
            self: 链式调用
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("需要安装scipy才能导出MATLAB格式")

        fn, zeta = self.model.natural_frequencies_damping()
        eig = self.model.eigenvalues()

        data = {
            'A': self.model.A,
            'B': self.model.B,
            'C': self.model.C,
            'D': self.model.D,
            'dt': self.model.dt,
            'n_states': self.model.n_states,
            'n_inputs': self.model.n_inputs,
            'n_outputs': self.model.n_outputs,
            'name': self.model.name,
            'eigenvalues': eig,
            'natural_frequencies': fn,
            'damping_ratios': zeta,
            'export_time': datetime.now().isoformat(),
        }

        if not filepath.endswith('.mat'):
            filepath += '.mat'

        sio.savemat(filepath, {variable_name: data})
        return self

    def export_json(self, filepath, indent=2, include_matrices=True):
        """
        导出为JSON格式

        参数:
            filepath: 输出文件路径
            indent: JSON缩进空格数
            include_matrices: 是否包含完整矩阵

        返回:
            self: 链式调用
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        fn, zeta = self.model.natural_frequencies_damping()
        eig = self.model.eigenvalues()

        data = {
            'name': self.model.name,
            'dt': self.model.dt,
            'n_states': int(self.model.n_states),
            'n_inputs': int(self.model.n_inputs),
            'n_outputs': int(self.model.n_outputs),
            'metadata': self._get_metadata(),
            'dynamics': {
                'natural_frequencies_hz': fn.tolist(),
                'damping_ratios': zeta.tolist(),
                'eigenvalues_real': eig.real.tolist(),
                'eigenvalues_imag': eig.imag.tolist(),
            }
        }

        if include_matrices:
            data.update({
                'A': self.model.A.tolist(),
                'B': self.model.B.tolist(),
                'C': self.model.C.tolist(),
                'D': self.model.D.tolist(),
            })

        if not filepath.endswith('.json'):
            filepath += '.json'

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        return self

    def export_csv(self, directory):
        """
        导出为CSV格式（每个矩阵一个文件）

        参数:
            directory: 输出目录路径

        返回:
            self: 链式调用
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        os.makedirs(directory, exist_ok=True)

        import pandas as pd

        pd.DataFrame(self.model.A).to_csv(
            os.path.join(directory, 'matrix_A.csv'), index=False, header=False
        )
        pd.DataFrame(self.model.B).to_csv(
            os.path.join(directory, 'matrix_B.csv'), index=False, header=False
        )
        pd.DataFrame(self.model.C).to_csv(
            os.path.join(directory, 'matrix_C.csv'), index=False, header=False
        )
        pd.DataFrame(self.model.D).to_csv(
            os.path.join(directory, 'matrix_D.csv'), index=False, header=False
        )

        fn, zeta = self.model.natural_frequencies_damping()
        info = pd.DataFrame({
            'parameter': ['dt', 'n_states', 'n_inputs', 'n_outputs'] +
                       [f'fn_{i}_Hz' for i in range(len(fn))] +
                       [f'damping_{i}' for i in range(len(zeta))],
            'value': [self.model.dt, self.model.n_states,
                     self.model.n_inputs, self.model.n_outputs] +
                    fn.tolist() + zeta.tolist()
        })
        info.to_csv(os.path.join(directory, 'model_info.csv'), index=False)

        return self

    def export_simulink_sfunction(self, filepath, function_name='rom_sfunction'):
        """
        导出为Simulink S函数（M文件）

        参数:
            filepath: 输出文件路径
            function_name: S函数名称

        返回:
            self: 链式调用
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        A_str = self._matrix_to_matlab_str(self.model.A)
        B_str = self._matrix_to_matlab_str(self.model.B)
        C_str = self._matrix_to_matlab_str(self.model.C)
        D_str = self._matrix_to_matlab_str(self.model.D)

        code = f"""function [sys,x0,str,ts] = {function_name}(t,x,u,flag)
% {function_name} - State-space ROM S-function
% Generated by AeroROM Toolbox
% Model: {self.model.name}
% States: {self.model.n_states}, Inputs: {self.model.n_inputs}, Outputs: {self.model.n_outputs}
% Sample time: {self.model.dt} s

% System matrices
A = {A_str};
B = {B_str};
C = {C_str};
D = {D_str};

n = {self.model.n_states};
p = {self.model.n_inputs};
m = {self.model.n_outputs};

switch flag,
    case 0,
        sizes = simsizes;
        sizes.NumContStates  = 0;
        sizes.NumDiscStates  = n;
        sizes.NumOutputs     = m;
        sizes.NumInputs      = p;
        sizes.DirFeedthrough = ~isempty(find(D, 1));
        sizes.NumSampleTimes = 1;
        sys = simsizes(sizes);
        x0  = zeros(n, 1);
        str = [];
        ts  = [{self.model.dt} 0];

    case 2,
        sys = A * x + B * u;

    case 3,
        sys = C * x + D * u;

    case {{1, 4, 9}},
        sys = [];

    otherwise
        error(['Unhandled flag = ',num2str(flag)]);
end
"""

        if not filepath.endswith('.m'):
            filepath += '.m'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

        return self

    def export_fortran(self, filepath, module_name='rom_model'):
        """
        导出为Fortran模块

        参数:
            filepath: 输出文件路径
            module_name: Fortran模块名

        返回:
            self: 链式调用
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        n = self.model.n_states
        p = self.model.n_inputs
        m = self.model.n_outputs

        A_str = self._matrix_to_fortran_str(self.model.A)
        B_str = self._matrix_to_fortran_str(self.model.B)
        C_str = self._matrix_to_fortran_str(self.model.C)
        D_str = self._matrix_to_fortran_str(self.model.D)

        code = f"""module {module_name}
! Generated by AeroROM Toolbox
! Model: {self.model.name}
! States: {n}, Inputs: {p}, Outputs: {m}
! Sample time: {self.model.dt} s

  implicit none

  integer, parameter :: n_states = {n}
  integer, parameter :: n_inputs = {p}
  integer, parameter :: n_outputs = {m}
  double precision, parameter :: dt = {self.model.dt}d0

  double precision, dimension(n_states, n_states) :: A = reshape([ &
{A_str} &
  ], [n_states, n_states])

  double precision, dimension(n_states, n_inputs) :: B = reshape([ &
{B_str} &
  ], [n_states, n_inputs])

  double precision, dimension(n_outputs, n_states) :: C = reshape([ &
{C_str} &
  ], [n_outputs, n_states])

  double precision, dimension(n_outputs, n_inputs) :: D = reshape([ &
{D_str} &
  ], [n_outputs, n_inputs])

contains

  subroutine rom_state_update(x, u, x_next)
    implicit none
    double precision, dimension(n_states), intent(in) :: x
    double precision, dimension(n_inputs), intent(in) :: u
    double precision, dimension(n_states), intent(out) :: x_next
    x_next = matmul(A, x) + matmul(B, u)
  end subroutine rom_state_update

  subroutine rom_output(x, u, y)
    implicit none
    double precision, dimension(n_states), intent(in) :: x
    double precision, dimension(n_inputs), intent(in) :: u
    double precision, dimension(n_outputs), intent(out) :: y
    y = matmul(C, x) + matmul(D, u)
  end subroutine rom_output

end module {module_name}
"""

        if not filepath.endswith('.f90'):
            filepath += '.f90'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

        return self

    def export_python_module(self, filepath, module_name='rom_model'):
        """
        导出为独立Python模块

        参数:
            filepath: 输出文件路径
            module_name: 模块名

        返回:
            self: 链式调用
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        n = self.model.n_states
        p = self.model.n_inputs
        m = self.model.n_outputs

        A_str = np.array2string(self.model.A, separator=', ',
                               formatter={'float_kind': lambda x: f'{x:.15e}'})
        B_str = np.array2string(self.model.B, separator=', ',
                               formatter={'float_kind': lambda x: f'{x:.15e}'})
        C_str = np.array2string(self.model.C, separator=', ',
                               formatter={'float_kind': lambda x: f'{x:.15e}'})
        D_str = np.array2string(self.model.D, separator=', ',
                               formatter={'float_kind': lambda x: f'{x:.15e}'})

        code = f'''"""
{module_name}.py - Reduced Order Model Module
Generated by AeroROM Toolbox

Model: {self.model.name}
States: {n}, Inputs: {p}, Outputs: {m}
Sample time: {self.model.dt} s
"""

import numpy as np

n_states = {n}
n_inputs = {p}
n_outputs = {m}
dt = {self.model.dt}

A = np.array({A_str})
B = np.array({B_str})
C = np.array({C_str})
D = np.array({D_str})


def state_update(x, u):
    """
    更新状态: x(k+1) = A*x(k) + B*u(k)

    Parameters
    ----------
    x : np.ndarray (n_states,)
        Current state vector
    u : np.ndarray (n_inputs,)
        Input vector

    Returns
    -------
    x_next : np.ndarray (n_states,)
        Next state vector
    """
    return A @ x + B @ u


def output(x, u):
    """
    计算输出: y(k) = C*x(k) + D*u(k)

    Parameters
    ----------
    x : np.ndarray (n_states,)
        State vector
    u : np.ndarray (n_inputs,)
        Input vector

    Returns
    -------
    y : np.ndarray (n_outputs,)
        Output vector
    """
    return C @ x + D @ u


def simulate(U, x0=None):
    """
    仿真整个输入序列

    Parameters
    ----------
    U : np.ndarray (n_samples, n_inputs)
        Input sequence
    x0 : np.ndarray (n_states,), optional
        Initial state, defaults to zero

    Returns
    -------
    Y : np.ndarray (n_samples, n_outputs)
        Output sequence
    X : np.ndarray (n_samples, n_states)
        State sequence
    """
    U = np.atleast_2d(U)
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    n_samples = U.shape[0]
    X = np.zeros((n_samples, n_states))
    Y = np.zeros((n_samples, n_outputs))

    if x0 is not None:
        X[0] = x0

    for k in range(n_samples):
        Y[k] = output(X[k], U[k])
        if k < n_samples - 1:
            X[k + 1] = state_update(X[k], U[k])

    return Y, X
'''

        if not filepath.endswith('.py'):
            filepath += '.py'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

        return self

    def export_nastran_dmi(self, filepath):
        """
        导出为NASTRAN DMI格式（用于颤振分析）

        参数:
            filepath: 输出文件路径

        返回:
            self: 链式调用
        """
        if self.model is None:
            raise ValueError("请先设置模型")

        lines = []
        lines.append('$ NASTRAN DMI file for Aerodynamic ROM')
        lines.append(f'$ Model: {self.model.name}')
        lines.append(f'$ States: {self.model.n_states}')
        lines.append(f'$ Inputs: {self.model.n_inputs}')
        lines.append(f'$ Outputs: {self.model.n_outputs}')
        lines.append(f'$ Sample time: {self.model.dt} s')
        lines.append('')

        lines.extend(self._matrix_to_dmi(self.model.A, 'A'))
        lines.append('')
        lines.extend(self._matrix_to_dmi(self.model.B, 'B'))
        lines.append('')
        lines.extend(self._matrix_to_dmi(self.model.C, 'C'))
        lines.append('')
        lines.extend(self._matrix_to_dmi(self.model.D, 'D'))

        if not filepath.endswith('.bdf') and not filepath.endswith('.dat'):
            filepath += '.bdf'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return self

    def _get_metadata(self):
        """获取模型元数据"""
        return {
            'export_time': datetime.now().isoformat(),
            'model_name': self.model.name,
            'n_states': int(self.model.n_states),
            'n_inputs': int(self.model.n_inputs),
            'n_outputs': int(self.model.n_outputs),
            'dt': float(self.model.dt),
        }

    @staticmethod
    def _matrix_to_matlab_str(matrix):
        """将矩阵转换为MATLAB字符串"""
        rows = []
        for row in matrix:
            rows.append(' '.join([f'{x:.15e}' for x in row]))
        return '[' + '; '.join(rows) + ']'

    @staticmethod
    def _matrix_to_fortran_str(matrix):
        """将矩阵转换为Fortran字符串（列优先）"""
        flat = matrix.T.flatten()
        values = []
        for i, v in enumerate(flat):
            values.append(f'{v:.15e}d0')
            if (i + 1) % 5 == 0 and (i + 1) < len(flat):
                values[-1] += ', &'
            elif (i + 1) < len(flat):
                values[-1] += ', '
        return '    ' + '\n    '.join(values)

    @staticmethod
    def _matrix_to_dmi(matrix, name):
        """将矩阵转换为NASTRAN DMI格式"""
        nrows, ncols = matrix.shape
        lines = [f'DMI,{name},1,REAL,{nrows},{ncols},0,0']

        for j in range(ncols):
            for i in range(nrows):
                value = matrix[i, j]
                lines.append(f'{value:.12e}')

        return lines
