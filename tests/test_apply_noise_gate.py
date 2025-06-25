import numpy as np
import os
import sys
from scipy.io import wavfile

# Ensure the project root is on the import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_utils import load_audio
from noise_gate.utils import compute_envelope, apply_noise_gate, db_to_linear


def test_apply_noise_gate_noise_removed(tmp_path):
    data, fs = load_audio(os.path.join('samples', 'jfk.wav'))

    noise_duration = 1.0  # seconds
    noise_samples = int(noise_duration * fs)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, size=noise_samples)
    if data.ndim > 1:
        noise = noise[:, None]
    signal = np.concatenate([noise, data])

    threshold_db = -6.0
    floor_db = -50.0
    env = compute_envelope(
        signal,
        fs,
        threshold_db=threshold_db,
        floor_db=floor_db,
        attack_ms=10,
        hold_ms=50,
        decay_ms=100,
        silence_flag=True,
    )
    gated = apply_noise_gate(signal, env, db_to_linear(floor_db), False)

    out_path = tmp_path / 'gated_noise_voice.wav'
    wavfile.write(out_path, fs, np.int16(gated * 32767))

    before_noise = np.max(np.abs(signal[:noise_samples]))
    after_noise = np.max(np.abs(gated[:noise_samples]))
    before_voice = np.max(np.abs(signal[noise_samples:noise_samples + fs]))
    after_voice = np.max(np.abs(gated[noise_samples:noise_samples + fs]))

    assert after_noise < before_noise * 0.3
    assert after_voice >= before_voice * 0.8


def test_apply_noise_gate_full_silence(tmp_path):
    data, fs = load_audio(os.path.join('samples', 'jfk.wav'))

    threshold_db = 1.0  # higher than any amplitude in the file
    floor_db = -100.0
    env = compute_envelope(
        data,
        fs,
        threshold_db=threshold_db,
        floor_db=floor_db,
        attack_ms=0,
        hold_ms=0,
        decay_ms=0,
        silence_flag=False,
    )
    gated = apply_noise_gate(data, env, db_to_linear(floor_db), False)

    out_path = tmp_path / 'gated_silence.wav'
    wavfile.write(out_path, fs, np.int16(gated * 32767))

    assert np.max(np.abs(gated)) < db_to_linear(-99)
