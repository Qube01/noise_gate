# noise_gate
A noise gate in Python that mimics the behavior of Audacity's Noise Gate plugin.

## Command line usage

After installing the required dependencies, you can apply the gate from the
command line:

```bash
python -m noise_gate -i input.wav -o output.wav [options]
```

Available options:

* `-t/--threshold` – gate threshold in dB (default: -40)
* `-r/--reduction` – attenuation when the gate is closed in dB (default: 80)
* `--attack` – attack/lookahead time in ms
* `--hold` – hold time in ms
* `--decay` – decay time in ms
* `--gate-freq` – crossover frequency in kHz
* `--stereo-link/--no-stereo-link` – link or separate stereo channels
* `--start-silence` – start with the gate fully closed
```

