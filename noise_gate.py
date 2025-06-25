import argparse
from noise_gate.utils import process_file


def parse_args():
    parser = argparse.ArgumentParser(description="Apply a noise gate to a WAV file")
    parser.add_argument('-i', '--input', required=True, help='Input WAV file')
    parser.add_argument('-o', '--output', required=True, help='Output WAV file')
    parser.add_argument('-t', '--threshold', type=float, default=-40.0,
                        help='Gate threshold in dB')
    parser.add_argument('-r', '--reduction', type=float, default=80.0,
                        help='Amount of attenuation when gate is closed (dB)')
    parser.add_argument('--attack', type=float, default=0.0,
                        help='Lookahead/attack time in ms')
    parser.add_argument('--hold', type=float, default=0.0,
                        help='Hold time in ms after signal drops below threshold')
    parser.add_argument('--decay', type=float, default=0.0,
                        help='Decay time in ms to fade to floor')
    parser.add_argument('--gate-freq', type=float, default=0.0,
                        help='Crossover frequency in kHz for split gating')
    stereo = parser.add_mutually_exclusive_group()
    stereo.add_argument('--stereo-link', dest='stereo_link', action='store_true',
                        help='Link stereo channels (default)')
    stereo.add_argument('--no-stereo-link', dest='stereo_link', action='store_false',
                        help='Process channels independently')
    parser.set_defaults(stereo_link=True)
    parser.add_argument('--start-silence', dest='silence_flag', action='store_true',
                        help='Start gate in silent state')
    parser.set_defaults(silence_flag=False)
    return parser.parse_args()


def main():
    args = parse_args()
    params = {
        'THRESHOLD': args.threshold,
        'LEVEL-REDUCTION': args.reduction,
        'ATTACK': args.attack,
        'HOLD': args.hold,
        'DECAY': args.decay,
        'GATE-FREQ': args.gate_freq,
        'STEREO_LINK': args.stereo_link,
        'SILENCE_FLAG': args.silence_flag,
    }
    process_file(args.input, args.output, params)


if __name__ == '__main__':
    main()
