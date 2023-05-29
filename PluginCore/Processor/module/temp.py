import numpy as np
import pandas as pd

def convert_to_uV(x):
    return x*1e6

def exponential_moving_standardize(
        data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    r"""Perform exponential moving standardization.

    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential moving variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{->v_t}, eps)`.


    Parameters
    ----------
    data: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: np.ndarray (n_channels, n_times)
        Standardized data.
    """
    if data.ndim==3:
        res = np.ones_like(data)
        for i_sample in range(len(data)):
            res[i_sample] = exponential_moving_standardize(data[i_sample], factor_new=factor_new,
                                                           init_block_size=init_block_size,eps=eps)
        return res

    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_block_standardized = (
            data[0:init_block_size] - init_mean) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T

def exponential_moving_demean(data, factor_new=0.001, init_block_size=None):
    r"""Perform exponential moving demeanining.

    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Deman the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t)`.


    Parameters
    ----------
    data: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Demean data before to this index with regular demeaning.

    Returns
    -------
    demeaned: np.ndarray (n_channels, n_times)
        Demeaned data.
    """
    if data.ndim==3:
        res = np.ones_like(data)
        for i_sample in range(len(data)):
            res[i_sample] = exponential_moving_demean(data[i_sample], factor_new=factor_new,
                                                           init_block_size=init_block_size)
        return res
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        demeaned[0:init_block_size] = data[0:init_block_size] - init_mean
    return demeaned.T

def zscore(data):
    """Zscore normalize continuous or windowed data in-place.

    Parameters
    ----------
    data: np.ndarray (n_channels, n_times) or (n_windows, n_channels, n_times)
        continuous or windowed signal

    Returns
    -------
    zscored: np.ndarray (n_channels x n_times) or (n_windows x n_channels x
    n_times)
        normalized continuous or windowed data

    ..note:
        If this function is supposed to preprocess continuous data, it should be
        given to raw.apply_function().
    """
    zscored = data - np.mean(data, keepdims=True, axis=-1)
    zscored = zscored / np.std(zscored, keepdims=True, axis=-1)
    # TODO: the overriding of protected '_data' should be implemented in the
    # TODO: dataset when transforms are applied to windows
    if hasattr(data, '_data'):
        data._data = zscored
    return zscored

def scale(data, factor):
    """Scale continuous or windowed data in-place

    Parameters
    ----------
    data: np.ndarray (n_channels x n_times) or (n_windows x n_channels x
    n_times)
        continuous or windowed signal
    factor: float
        multiplication factor

    Returns
    -------
    scaled: np.ndarray (n_channels x n_times) or (n_windows x n_channels x
    n_times)
        normalized continuous or windowed data
    ..note:
        If this function is supposed to preprocess continuous data, it should be
        given to raw.apply_function().
    """
    scaled = np.multiply(data, factor)
    # TODO: the overriding of protected '_data' should be implemented in the
    # TODO: dataset when transforms are applied to windows
    if hasattr(data, '_data'):
        data._data = scaled
    return scaled