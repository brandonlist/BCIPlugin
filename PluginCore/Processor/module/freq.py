import scipy
import scipy.signal
import numpy as np
from warnings import warn

def HighpassCnt(data, low_cut_hz, fs, filt_order=8, axis=0):
    """
     Highpass signal applying **causal** butterworth filter of given order.

    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    fs: float
    filt_order: int

    Returns
    -------
    highpassed_data: 2d-array
        Data after applying highpass filter.
    """
    if (low_cut_hz is None) or (low_cut_hz == 0):
        print('Not doing any highpass, since low 0 or None')
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, low_cut_hz / (fs / 2.0), btype="highpass"
    )
    assert filter_is_stable(a)
    data_highpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_highpassed

def LowpassCnt(data, high_cut_hz, fs, filt_order=8, axis=0):
    """
     Lowpass signal applying **causal** butterworth filter of given order.

    Parameters
    ----------
    data: 2d-array
        Time x channels
    high_cut_hz: float
    fs: float
    filt_order: int

    Returns
    -------
    lowpassed_data: 2d-array
        Data after applying lowpass filter.
    """
    if (high_cut_hz is None) or (high_cut_hz == fs / 2.0):
        'Not doing any lowpass, since high cut hz is None or nyquist freq.'
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, high_cut_hz / (fs / 2.0), btype="lowpass"
    )
    assert filter_is_stable(a)
    data_lowpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_lowpassed

def BandpassCnt(
        data, low_cut_hz, high_cut_hz, fs, filt_order=8, axis=0, filtfilt=False
):
    """
     Bandpass signal applying **causal** butterworth filter of given order.
    Can be used as low-pass / high-pass filters by setting low_cut_hz=0 / high_cut_hz=None or Nyquist
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    high_cut_hz: float
    fs: float
    filt_order: int
    filtfilt: bool
        Whether to use filtfilt instead of lfilter

    Returns
    -------
    bandpassed_data: 2d-array
        Data after applying bandpass filter.
    """
    if (low_cut_hz == 0 or low_cut_hz is None) and (
            high_cut_hz == None or high_cut_hz == fs / 2.0
    ):
        print('low_cut cant be 0 or None, and high_cut_hz cant be None or higher than Nyquist')
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        print('Using lowpass filter since low cut hz is 0 or None')
        return LowpassCnt(
            data, high_cut_hz, fs, filt_order=filt_order, axis=axis
        )
    if high_cut_hz == None or high_cut_hz == (fs / 2.0):
        print('Using highpass filter since high cut hz is None or nyquist freq')
        return HighpassCnt(
            data, low_cut_hz, fs, filt_order=filt_order, axis=axis
        )

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype="bandpass")
    assert filter_is_stable(a), 'Filter should be stable...'
    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed

def filter_is_stable(a):
    """
    Check if filter coefficients of IIR filter are stable.

    Parameters
    ----------
    a: list or 1darray of number
        Denominator filter coefficients a.

    Returns
    -------
    is_stable: bool
        Filter is stable or not.
    Notes
    ----
    Filter is stable if absolute value of all  roots is smaller than 1,
    see [1]_.

    References
    ----------
    .. [1] HYRY, "SciPy 'lfilter' returns only NaNs" StackOverflow,
       http://stackoverflow.com/a/8812737/1469195
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a))
    )
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a)) < 1)

def filterbank(raw, frequency_bands, drop_original_signals=True,
               order_by_frequency_band=False, **mne_filter_kwargs):
    """Applies multiple bandpass filters to the signals in raw. The raw will be
    modified in-place and number of channels in raw will be updated to
    len(frequency_bands) * len(raw.ch_names) (-len(raw.ch_names) if
    drop_original_signals).

    Parameters
    ----------
    raw: mne.io.Raw
        The raw signals to be filtered.
    frequency_bands: list(tuple)
        The frequency bands to be filtered for (e.g. [(4, 8), (8, 13)]).
    drop_original_signals: bool
        Whether to drop the original unfiltered signals
    order_by_frequency_band: bool
        If True will return channels odered by frequency bands, so if there
        are channels Cz, O1 and filterbank ranges [(4,8), (8,13)], returned
        channels will be [Cz_4-8, O1_4-8, Cz_8-13, O1_8-13]. If False, order
        will be [Cz_4-8, Cz_8-13, O1_4-8, O1_8-13].
    mne_filter_kwargs: dict
        Keyword arguments for filtering supported by mne.io.Raw.filter().
        Please refer to mne for a detailed explanation.
    """
    if not frequency_bands:
        raise ValueError(f"Expected at least one frequency band, got"
                         f" {frequency_bands}")
    if not all([len(ch_name) < 8 for ch_name in raw.ch_names]):
        warn("Try to use shorter channel names, since frequency band "
             "annotation requires an estimated 4-8 chars depending on the "
             "frequency ranges. Will truncate to 15 chars (mne max).")
    original_ch_names = raw.ch_names
    all_filtered = []
    for (l_freq, h_freq) in frequency_bands:
        filtered = raw.copy()
        filtered.filter(l_freq=l_freq, h_freq=h_freq, **mne_filter_kwargs)
        # mne automatically changes the highpass/lowpass info values
        # when applying filters and channels cant be added if they have
        # different such parameters. Not needed when making picks as
        # high pass is not modified by filter if pick is specified
        filtered.info["highpass"] = raw.info["highpass"]
        filtered.info["lowpass"] = raw.info["lowpass"]
        # add frequency band annotation to channel names
        # truncate to a max of 15 characters, since mne does not allow for more
        filtered.rename_channels({
            old_name: (old_name + f"_{l_freq}-{h_freq}")[-15:]
            for old_name in filtered.ch_names})
        all_filtered.append(filtered)
    raw.add_channels(all_filtered)
    if not order_by_frequency_band:
        # order channels by name and not by frequency band:
        # index the list with a stepsize of the number of channels for each of
        # the original channels
        chs_by_freq_band = []
        for i in range(len(original_ch_names)):
            chs_by_freq_band.extend(raw.ch_names[i::len(original_ch_names)])
        raw.reorder_channels(chs_by_freq_band)
    if drop_original_signals:
        raw.drop_channels(original_ch_names)
