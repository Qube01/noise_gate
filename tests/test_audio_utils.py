import numpy as np
import os
import sys

# Ensure the project root is on the import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_utils import butter_filter, lowpass8, highpass8


def amplitude_at(signal, freq, fs):
    fft = np.fft.rfft(signal)
    idx = int(freq * len(signal) / fs)
    return 2 * np.abs(fft[idx]) / len(signal)


def generate_sine(freq, duration, fs):
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def test_lowpass8_fft():
    fs = 44100
    duration = 1.0
    low_freq = 200
    high_freq = 5000

    sig_low = generate_sine(low_freq, duration, fs)
    sig_high = generate_sine(high_freq, duration, fs)
    signal = sig_low + sig_high

    cutoff = 1000
    filtered = lowpass8(signal, cutoff, fs)

    amp_low_before = amplitude_at(signal, low_freq, fs)
    amp_low_after = amplitude_at(filtered, low_freq, fs)
    amp_high_before = amplitude_at(signal, high_freq, fs)
    amp_high_after = amplitude_at(filtered, high_freq, fs)

    assert amp_low_after / amp_low_before > 0.8
    assert amp_high_after / amp_high_before < 0.1


def test_highpass8_fft():
    fs = 44100
    duration = 1.0
    low_freq = 200
    high_freq = 5000

    sig_low = generate_sine(low_freq, duration, fs)
    sig_high = generate_sine(high_freq, duration, fs)
    signal = sig_low + sig_high

    cutoff = 1000
    filtered = highpass8(signal, cutoff, fs)

    amp_low_before = amplitude_at(signal, low_freq, fs)
    amp_low_after = amplitude_at(filtered, low_freq, fs)
    amp_high_before = amplitude_at(signal, high_freq, fs)
    amp_high_after = amplitude_at(filtered, high_freq, fs)

    assert amp_high_after / amp_high_before > 0.8
    assert amp_low_after / amp_low_before < 0.1


def test_butter_filter_lowpass_fft():
    fs = 44100
    duration = 1.0
    low_freq = 300
    high_freq = 4000

    sig_low = generate_sine(low_freq, duration, fs)
    sig_high = generate_sine(high_freq, duration, fs)
    signal = sig_low + sig_high

    cutoff = 1000
    filtered = butter_filter(signal, cutoff, fs, "low")

    amp_low_before = amplitude_at(signal, low_freq, fs)
    amp_low_after = amplitude_at(filtered, low_freq, fs)
    amp_high_before = amplitude_at(signal, high_freq, fs)
    amp_high_after = amplitude_at(filtered, high_freq, fs)

    assert amp_low_after / amp_low_before > 0.8
    assert amp_high_after / amp_high_before < 0.1
