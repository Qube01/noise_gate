import math


def db_to_linear(db: float) -> float:
    """Convert decibel value to linear scale."""
    return 10 ** (db / 20)


def linear_to_db(x: float) -> float:
    """Convert linear value to decibels.

    The input value is clamped to a minimum of 1e-10 to avoid log(0).
    """
    x = max(x, 1e-10)
    return 20 * math.log10(x)


def compute_envelope(
    signal,
    sample_rate: int,
    threshold_db: float,
    floor_db: float,
    attack_ms: float,
    hold_ms: float,
    decay_ms: float,
    silence_flag: bool = False,
) -> list:
    """Return a gate envelope for ``signal``.

    Parameters
    ----------
    signal : ``numpy.ndarray`` or sequence of floats
        Input audio signal. Can be mono ``(n_samples,)`` or multi-channel
        ``(n_samples, n_channels)``.
    sample_rate : int
        Sampling rate of ``signal`` in Hertz.
    threshold_db : float
        Amplitude threshold in decibels above which the gain stays at ``1``.
    floor_db : float
        Minimum gain in decibels when the gate is fully closed.
    attack_ms : float
        Lookahead time in milliseconds. The gate will anticipate incoming
        signal by this amount.
    hold_ms : float
        Duration in milliseconds to keep the gain at ``1`` after the signal
        falls below ``threshold_db``.
    decay_ms : float
        Time in milliseconds for the gain to exponentially decay from ``1``
        to ``floor_db`` after the hold period.
    silence_flag : bool, optional
        If ``True`` the initial gain starts at ``floor`` instead of ``1``.

    Returns
    -------
    list of float
        Gain envelope with values clamped to ``[floor, 1.0]``.
    """

    import numpy as np

    signal = np.asarray(signal, dtype=float)

    if signal.ndim > 1:
        follower = np.max(np.abs(signal), axis=1)
    else:
        follower = np.abs(signal)

    threshold = db_to_linear(threshold_db)
    floor = db_to_linear(floor_db)

    attack_samples = int(max(attack_ms, 0) * sample_rate / 1000)
    hold_samples = int(max(hold_ms, 0) * sample_rate / 1000)
    decay_samples = int(max(decay_ms, 0) * sample_rate / 1000)

    if attack_samples > 0:
        pad = np.full(attack_samples, follower[-1], dtype=float)
        follower = np.concatenate([follower[attack_samples:], pad])

    if decay_samples > 0:
        decay_coeff = floor ** (1.0 / decay_samples)
    else:
        decay_coeff = floor

    env = np.empty(len(follower), dtype=float)
    gain = 1.0 if not silence_flag else floor
    hold_counter = 0

    for i, amp in enumerate(follower):
        if amp >= threshold:
            gain = 1.0
            hold_counter = hold_samples
        else:
            if hold_counter > 0:
                hold_counter -= 1
                gain = 1.0
            else:
                gain *= decay_coeff
                if gain < floor:
                    gain = floor
        env[i] = gain

    return env.tolist()
