import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter


def load_audio(path):
    """Load a mono or stereo WAV file.

    Parameters
    ----------
    path : str or Path
        Path to the WAV file.

    Returns
    -------
    data : numpy.ndarray of dtype float32
        Audio samples normalized to [-1.0, 1.0]. If the file contains
        two channels, the shape is (n_samples, 2).
    sample_rate : int
        Sampling frequency of the audio file.
    """
    sample_rate, data = wavfile.read(path)
    dtype = data.dtype

    # Convert to float32
    data = np.asarray(data, dtype=np.float32)

    if np.issubdtype(dtype, np.integer):
        max_val = float(np.iinfo(dtype).max)
        if dtype == np.int16:
            max_val = 32768.0
        elif dtype == np.int32:
            max_val = 2147483648.0
        elif dtype == np.uint8:
            data = data - 128.0
            max_val = 128.0
        data /= max_val
    else:
        max_abs = np.max(np.abs(data))
        if max_abs > 1.0:
            data /= max_abs

    return data, sample_rate


def butter_filter(signal, cutoff_freq_hz, fs, type):
    """Apply a Butterworth filter to a signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal. Can be mono ``(n_samples,)`` or multi-channel
        ``(n_samples, n_channels)``. Filtering is done along the first
        axis.
    cutoff_freq_hz : float or sequence of float
        Cutoff frequency/frequencies in Hertz.
    fs : int or float
        Sampling rate of ``signal``.
    type : {'low', 'high', 'bandpass', 'bandstop'}
        Type of Butterworth filter to apply.

    Returns
    -------
    numpy.ndarray
        Filtered signal with the same shape as ``signal``.
    """

    nyq = 0.5 * fs
    normal_cutoff = np.asarray(cutoff_freq_hz, dtype=float) / nyq
    b, a = butter(8, normal_cutoff, btype=type)
    return lfilter(b, a, signal, axis=0)


def lowpass8(signal, cutoff_freq_hz, fs):
    """8th-order low-pass Butterworth filter."""
    return butter_filter(signal, cutoff_freq_hz, fs, "low")


def highpass8(signal, cutoff_freq_hz, fs):
    """8th-order high-pass Butterworth filter."""
    return butter_filter(signal, cutoff_freq_hz, fs, "high")
