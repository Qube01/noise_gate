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
    stereo_link: bool = True,
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
    stereo_link : bool, optional
        If ``True`` (default), a single envelope is computed from the maximum
        absolute value across channels. If ``False``, each channel is processed
        independently and a multi-channel envelope is returned.

    Returns
    -------
    list of float
        Gain envelope with values clamped to ``[floor, 1.0]``.
    """

    import numpy as np

    signal = np.asarray(signal, dtype=float)

    if signal.ndim == 1:
        followers = [np.abs(signal)]
    else:
        if stereo_link:
            followers = [np.max(np.abs(signal), axis=1)]
        else:
            followers = [np.abs(signal[:, ch]) for ch in range(signal.shape[1])]

    threshold = db_to_linear(threshold_db)
    floor = db_to_linear(floor_db)

    attack_samples = int(max(attack_ms, 0) * sample_rate / 1000)
    hold_samples = int(max(hold_ms, 0) * sample_rate / 1000)
    decay_samples = int(max(decay_ms, 0) * sample_rate / 1000)

    if decay_samples > 0:
        decay_coeff = floor ** (1.0 / decay_samples)
    else:
        decay_coeff = floor

    def process(follower):
        if attack_samples > 0:
            pad = np.full(attack_samples, follower[-1], dtype=float)
            follower = np.concatenate([follower[attack_samples:], pad])

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

        return env

    envs = [process(f) for f in followers]

    if len(envs) == 1:
        return envs[0].tolist()
    else:
        return np.stack(envs, axis=1).tolist()


def apply_noise_gate(signal, envelope, floor, silence_flag):
    """Apply a gain envelope to a signal.

    Parameters
    ----------
    signal : numpy.ndarray or sequence of floats
        Input signal. Can be mono ``(n_samples,)`` or multi-channel
        ``(n_samples, n_channels)``.
    envelope : sequence of float
        Gain values produced by :func:`compute_envelope`.
    floor : float
        Minimum gain value (linear) when the gate is fully closed.
    silence_flag : bool
        If ``True`` the output gain starts at ``floor`` instead of ``1``.

    Returns
    -------
    numpy.ndarray
        The gated signal with the same shape as ``signal``.
    """

    import numpy as np
    from .audio_utils import highpass8, lowpass8

    # Defaults for optional global parameters
    gate_freq = globals().get("GATE_FREQ", 0.0)
    sample_rate = globals().get("SAMPLE_RATE", 44100)

    signal = np.asarray(signal, dtype=float)
    env = np.asarray(envelope, dtype=float)
    gain = 1.0 if not silence_flag else floor

    if env.ndim == 1 and signal.ndim > 1:
        env = env[:, None]

    if gate_freq > 20:
        hp = highpass8(signal, gate_freq, sample_rate)
        lp = lowpass8(signal, gate_freq, sample_rate)
        gated_hp = hp * env
        output = gated_hp + lp
    else:
        output = signal * env

    output *= gain
    return output


def process_file(path_in, path_out, params_dict):
    """Apply a noise gate to an audio file and save the result.

    Parameters
    ----------
    path_in : str or Path
        Input audio file (WAV format).
    path_out : str or Path
        Destination path for the gated audio.
    params_dict : dict
        Dictionary containing gate parameters:

        - ``THRESHOLD`` (float): Threshold in dB.
        - ``LEVEL-REDUCTION`` (float): Amount of reduction in dB when the gate
          is closed. Positive values represent attenuation.
        - ``ATTACK`` (float): Attack/lookahead time in ms.
        - ``HOLD`` (float): Hold time in ms.
        - ``DECAY`` (float): Decay time in ms.
        - ``GATE-FREQ`` (float): Crossover frequency in kHz.
        - ``STEREO_LINK`` (bool): Process channels together if ``True``.
        - ``SILENCE_FLAG`` (bool): Start with gain at the floor if ``True``.

    The processed audio is written to ``path_out``.
    """

    from .audio_utils import load_audio, save_audio

    signal, fs = load_audio(path_in)

    threshold = float(params_dict.get("THRESHOLD", -40.0))
    level_red = float(params_dict.get("LEVEL-REDUCTION", 80.0))
    attack = float(params_dict.get("ATTACK", 0.0))
    hold = float(params_dict.get("HOLD", 0.0))
    decay = float(params_dict.get("DECAY", 0.0))
    gate_freq_khz = float(params_dict.get("GATE-FREQ", 0.0))
    stereo_link = bool(params_dict.get("STEREO_LINK", True))
    silence_flag = bool(params_dict.get("SILENCE_FLAG", False))

    floor_db = -abs(level_red) if level_red > 0 else level_red

    env = compute_envelope(
        signal,
        fs,
        threshold_db=threshold,
        floor_db=floor_db,
        attack_ms=attack,
        hold_ms=hold,
        decay_ms=decay,
        silence_flag=silence_flag,
        stereo_link=stereo_link,
    )

    global GATE_FREQ, SAMPLE_RATE
    GATE_FREQ = gate_freq_khz * 1000.0
    SAMPLE_RATE = fs

    gated = apply_noise_gate(signal, env, db_to_linear(floor_db), silence_flag)

    save_audio(path_out, gated, fs)

    return gated
