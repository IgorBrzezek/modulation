# audio_modulator.py
#
# --- Metadata Variables ---
__DATE__ = "2025-12-07"
__VERSION__ = "1.1.3" # Wersja z poprawioną słyszalnością modulacji FM
__AUTHOR__ = "Igor Brzeżek"
__EMAIL__ = "igor.brzezek@gmail.com"
__GITHUB__ = "/igorbrzezek"
# --------------------------

import argparse
import sys
import os
import numpy as np
from scipy.io import wavfile
from typing import Tuple, Optional

# --- Colorama Integration for better terminal compatibility ---
try:
    from colorama import init
    init() 
except ImportError:
    pass
# -----------------------------------------------------------

# --- Color Codes for --color option ---
class Color:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Signal Processing Functions ---

def load_wav(filepath: str) -> Optional[Tuple[int, np.ndarray]]:
    """Loads a WAV file and returns sample rate and data, or None on error."""
    try:
        sample_rate, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(data.dtype)
        return sample_rate, data
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error loading WAV file '{filepath}': {e}")
        return None

def save_wav(filepath: str, sample_rate: int, data: np.ndarray):
    """Saves a NumPy array as a WAV file."""
    try:
        if data.dtype != np.int16:
            if np.issubdtype(data.dtype, np.floating):
                max_abs = np.max(np.abs(data))
                if max_abs > 0:
                    data = data / max_abs * (2**15 - 1)
                else:
                    data = data * (2**15 - 1)
            data = data.astype(np.int16)

        wavfile.write(filepath, sample_rate, data)
        print(f"Successfully saved output to '{filepath}'")
    except Exception as e:
        print(f"Error saving WAV file '{filepath}': {e}")

def normalize_signals(carrier: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalizes both signals to the range [-1.0, 1.0] for processing."""
    def normalize(data: np.ndarray) -> np.ndarray:
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
        else:
            max_val = np.max(np.abs(data))

        if max_val == 0:
            return data.astype(np.float64)
        return data.astype(np.float64) / max_val

    min_len = min(len(carrier), len(signal))
    carrier_norm = normalize(carrier[:min_len])
    signal_norm = normalize(signal[:min_len])

    return carrier_norm, signal_norm

def apply_am(carrier_norm: np.ndarray, signal_norm: np.ndarray) -> np.ndarray:
    """Applies Amplitude Modulation (AM)."""
    modulated_signal = (1.0 + signal_norm) * carrier_norm
    return modulated_signal

def apply_fm(carrier: np.ndarray, signal: np.ndarray, fs: int) -> np.ndarray:
    """Applies Frequency Modulation (FM). Enhanced implementation for audibility."""
    
    # Krok 1: Normalizacja sygnału modulującego do pełnego zakresu [-1.0, 1.0]
    signal_norm = signal / np.max(np.abs(signal))

    # Krok 2: Ustalenie dużej dewiacji dla wyraźnego efektu (1000 Hz)
    freq_deviation = 1000.0 
    
    # Krok 3: Obliczenie fazy sygnału modulującego (integral sygnału modulującego)
    phase_signal = 2.0 * np.pi * freq_deviation * np.cumsum(signal_norm) / fs
    
    # Krok 4: Utworzenie syntetycznej nośnej (2000 Hz)
    t = np.arange(len(carrier)) / fs
    carrier_frequency = 2000.0 
    
    # Modulowany sygnał (FM: cos(2 * pi * fc * t + phase_signal))
    modulated_signal = np.cos(2.0 * np.pi * carrier_frequency * t + phase_signal)
    
    return modulated_signal

def apply_sum(carrier_norm: np.ndarray, signal_norm: np.ndarray) -> np.ndarray:
    """Applies simple summation (mixing) of the two signals."""
    summed_signal = carrier_norm + signal_norm
    max_abs = np.max(np.abs(summed_signal))
    if max_abs > 1.0:
        summed_signal /= max_abs
    return summed_signal

# --- Utility Functions ---

def print_stats(filepath: str, sample_rate: int, data: np.ndarray, is_color: bool):
    """Prints statistics for a loaded WAV file."""
    c = lambda text, color: f"{color}{text}{Color.RESET}" if is_color else text

    data_len = len(data)
    duration = data_len / sample_rate
    min_val = np.min(data)
    max_val = np.max(data)
    mean_abs = np.mean(np.abs(data))

    print(f"\n--- {c('Statistics for:', Color.HEADER)} {c(filepath, Color.BOLD)} ---")
    print(f"{c('Sample Rate (Hz):', Color.CYAN)} {sample_rate}")
    print(f"{c('Duration (s):', Color.CYAN)} {duration:.3f}")
    print(f"{c('Total Samples:', Color.CYAN)} {data_len}")
    print(f"{c('Min Amplitude:', Color.YELLOW)} {min_val}")
    print(f"{c('Max Amplitude:', Color.YELLOW)} {max_val}")
    print(f"{c('Average Absolute Amplitude:', Color.YELLOW)} {mean_abs:.3f}")
    print(f"{c('Data Type:', Color.CYAN)} {data.dtype}")
    print("-" * 40)

# --- Main Function ---

def main():
    """Main function to parse arguments and execute the audio processing."""
    
    # --- Custom Help Message Generation ---
    version_string = f"WAV File Modulator and Mixer v{__VERSION__} (Created: {__DATE__})"
    
    # Custom formatter to inject metadata into simple help (-h)
    class CustomHelpFormatter(argparse.HelpFormatter):
        def format_usage(self, usage, actions, groups, prefix):
            metadata_line = f"Metadata: Author: {__AUTHOR__} | Version: {__VERSION__} | Date: {__DATE__}\n"
            return metadata_line + super().format_usage(usage, actions, groups, prefix)

    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(
        description=f"{Color.BOLD}{version_string}{Color.RESET}",
        formatter_class=CustomHelpFormatter,
        add_help=False
    )

    # Required arguments (with short aliases)
    parser.add_argument('-c', '--carrier', type=str, required=True,
                        help='Path to the carrier WAV file.')
    parser.add_argument('-s', '--signal', type=str, required=True,
                        help='Path to the signal (modulating) WAV file.')
    parser.add_argument('-e', '--effect', type=str, required=True,
                        choices=['am', 'fm', 'sum'],
                        help='The modulation/mixing effect to apply:\n  am: Amplitude Modulation\n  fm: Frequency Modulation\n  sum: Simple Summation (Mixing)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to the output WAV file.')

    # Optional arguments
    parser.add_argument('--stat', action='store_true',
                        help='Show parameters (statistics) of the input files.')
    parser.add_argument('--color', action='store_true',
                        help='Show output with ANSI color codes.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the output file without asking.')
    parser.add_argument('-h', action='store_true', help='Show this simple help message.')
    parser.add_argument('--help', action='store_true', help='Show detailed help with examples and metadata.')

    # Parse initial arguments to check for help flags first
    if '-h' in sys.argv:
        parser.print_help()
        sys.exit(0)
        
    if '--help' in sys.argv:
        print(f"\n{Color.BOLD}Detailed Help and Examples for audio_modulator.py v{__VERSION__}{Color.RESET}\n")
        
        # Display Metadata
        print(f"{Color.HEADER}Metadata:{Color.RESET}")
        print(f"  {Color.CYAN}Version:{Color.RESET} {__VERSION__}")
        print(f"  {Color.CYAN}Date Created:{Color.RESET} {__DATE__}")
        print(f"  {Color.CYAN}Author:{Color.RESET} {__AUTHOR__}")
        print(f"  {Color.CYAN}Email:{Color.RESET} {__EMAIL__}")
        print(f"  {Color.CYAN}GitHub:{Color.RESET} {__GITHUB__}")
        
        print(f"\n{Color.HEADER}Description:{Color.RESET}")
        print("This script performs Amplitude Modulation (AM), Frequency Modulation (FM), or simple summation on two short WAV files.")
        print("The file specified by '-c'/'--carrier' is always the carrier wave.")
        print("The file specified by '-s'/'--signal' is always the modulating signal.")

        print(f"\n{Color.HEADER}Usage:{Color.RESET} python audio_modulator.py -c CARRIER_FILE -s SIGNAL_FILE -e EFFECT -o OUTPUT_FILE [OPTIONS]")

        print(f"\n{Color.HEADER}Required Arguments:{Color.RESET}")
        print("  -c, --carrier CARRIER_FILE  Path to the carrier WAV file.")
        print("  -s, --signal SIGNAL_FILE    Path to the signal (modulating) WAV file.")
        print("  -e, --effect {am, fm, sum}  The operation to perform (am, fm, or sum).")
        print("  -o, --output OUTPUT_FILE    Path to the resulting output WAV file.")

        print(f"\n{Color.HEADER}Optional Flags:{Color.RESET}")
        print("  --stat                      Show statistics for the input files.")
        print("  --color                     Use colored output for statistics and messages.")
        print("  --overwrite                 Overwrite the output file if it exists without prompting.")
        print("  -h                          Show simple help message.")
        print("  --help                      Show this detailed help with examples and metadata.")

        print(f"\n{Color.HEADER}Examples:{Color.RESET}")
        print(f"{Color.BLUE}1. Perform Amplitude Modulation (AM) and save output:{Color.RESET}")
        print("$ python audio_modulator.py -c carrier.wav -s signal.wav -e am -o modulated_am.wav")

        print(f"{Color.BLUE}2. Perform Summation (Mixing) with stats and force overwrite:{Color.RESET}")
        print("$ python audio_modulator.py -c tone_c.wav -s music_s.wav -e sum -o mixed.wav --stat --overwrite")
        
        print(f"{Color.BLUE}3. Perform Frequency Modulation (FM) (Requires tone for -c and slow tone for -s to hear effect):{Color.RESET}")
        print("$ python audio_modulator.py -c 2000hz_tone.wav -s 10hz_tone.wav -e fm -o modulated_fm.wav")

        sys.exit(0)

    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(1)

    # --- Overwrite Check ---
    if os.path.exists(args.output) and not args.overwrite:
        prompt = f"Output file '{args.output}' already exists. Overwrite? (y/N): "
        if args.color:
             prompt = f"{Color.YELLOW}WARNING:{Color.RESET} {prompt}"
        
        response = input(prompt).strip().lower()
        
        if response != 'y':
            print("Operation cancelled by user.")
            sys.exit(0)
        else:
            print("Overwriting file...")

    # --- Load Files ---
    carrier_data = load_wav(args.carrier)
    signal_data = load_wav(args.signal)

    if carrier_data is None or signal_data is None:
        sys.exit(1)

    carrier_rate, carrier_raw = carrier_data
    signal_rate, signal_raw = signal_data

    # Check for matching sample rates
    if carrier_rate != signal_rate:
        print(f"Error: Sample rates must match for processing. Carrier: {carrier_rate} Hz, Signal: {signal_rate} Hz.")
        sys.exit(1)

    fs = carrier_rate

    # --- Print Stats ---
    if args.stat:
        print_stats(args.carrier, fs, carrier_raw, args.color)
        print_stats(args.signal, fs, signal_raw, args.color)

    # --- Prepare Signals ---
    carrier_norm, signal_norm = normalize_signals(carrier_raw, signal_raw)

    # --- Apply Effect ---
    output_signal = None
    effect_name = args.effect.upper()

    print(f"\nApplying {effect_name} effect...")

    if args.effect == 'am':
        output_signal = apply_am(carrier_norm, signal_norm)
    elif args.effect == 'fm':
        output_signal = apply_fm(carrier_raw, signal_raw, fs)
    elif args.effect == 'sum':
        output_signal = apply_sum(carrier_norm, signal_norm)
    else:
        print(f"Error: Unknown effect '{args.effect}'")
        sys.exit(1)

    # --- Save Output ---
    if output_signal is not None:
        # Final normalization and conversion to int16
        max_abs = np.max(np.abs(output_signal))
        if max_abs > 0:
            output_signal = output_signal / max_abs * (2**15 - 1)
        else:
            output_signal = output_signal * (2**15 - 1)
        output_signal = output_signal.astype(np.int16)

        save_wav(args.output, fs, output_signal)
    else:
        print("Error: Output signal was not generated.")
        sys.exit(1)

if __name__ == '__main__':
    # Check for required libraries
    try:
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        print("\nRequired Python libraries (numpy, scipy) not found.")
        print("Please install them using: pip install numpy scipy")
        sys.exit(1)

    # Note: If colors still don't work, ensure 'colorama' is installed: pip install colorama
    main()