import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from noise_gate.utils import compute_envelope, db_to_linear


def test_envelope_constant_above_threshold():
    fs = 1000
    signal = np.ones(100) * 0.6
    env = compute_envelope(
        signal,
        fs,
        threshold_db=-6,
        floor_db=-60,
        attack_ms=0,
        hold_ms=0,
        decay_ms=0,
        silence_flag=False,
    )
    assert np.allclose(env, 1.0)


def test_envelope_constant_below_threshold():
    fs = 1000
    signal = np.ones(50) * 0.05
    floor_db = -40
    env = compute_envelope(
        signal,
        fs,
        threshold_db=-20,
        floor_db=floor_db,
        attack_ms=0,
        hold_ms=0,
        decay_ms=0,
        silence_flag=False,
    )
    floor = db_to_linear(floor_db)
    assert np.allclose(env, floor)


def test_lookahead_and_hold_decay():
    fs = 1000
    # 30 ms silence, 10 ms signal above threshold, then silence
    sig = np.concatenate([np.zeros(30), np.ones(10), np.zeros(60)])
    env = compute_envelope(
        sig,
        fs,
        threshold_db=-6,
        floor_db=-60,
        attack_ms=10,  # lookahead
        hold_ms=20,
        decay_ms=10,
    )
    # Gate should open 10 ms before the signal rises (at sample 20)
    assert env[19] < 0.5 and env[20] == 1.0
    # Hold keeps gain at 1 for 10 ms after the burst ends (samples 40-49)
    assert np.allclose(env[40:50], 1.0)
    # After hold period, gain decays toward floor within 10 ms
    floor = db_to_linear(-60)
    assert env[60] <= floor + 1e-6


def test_envelope_stereo_link_true():
    fs = 1000
    sig = np.array([
        [0.5, 0.1],
        [0.6, 0.2],
        [-0.7, -0.8],
        [0.2, -0.1],
    ])
    env = compute_envelope(
        sig,
        fs,
        threshold_db=-6,
        floor_db=-60,
        attack_ms=0,
        hold_ms=0,
        decay_ms=0,
        stereo_link=True,
    )
    floor = db_to_linear(-60)
    expected = [floor, 1.0, 1.0, floor]
    assert np.allclose(env, expected)


def test_envelope_stereo_link_false():
    fs = 1000
    sig = np.array([
        [0.5, 0.1],
        [0.6, 0.2],
        [-0.7, -0.8],
        [0.2, -0.1],
    ])
    env = compute_envelope(
        sig,
        fs,
        threshold_db=-6,
        floor_db=-60,
        attack_ms=0,
        hold_ms=0,
        decay_ms=0,
        stereo_link=False,
    )
    floor = db_to_linear(-60)
    expected = np.array([
        [floor, floor],
        [1.0, floor],
        [1.0, 1.0],
        [floor, floor],
    ])
    assert np.allclose(env, expected)
