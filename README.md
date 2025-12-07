# Audio Modulator and Mixer

`audio_modulator.py` is a Python script designed for performing various audio signal processing operations, including Amplitude Modulation (AM), Frequency Modulation (FM), and simple signal summation (mixing) on WAV files. It provides a command-line interface for easy use and includes options for displaying file statistics and colored output.

## Table of Contents

- [Features](#features)
- [Metadata](#metadata)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Arguments](#arguments)
  - [Examples](#examples)
- [Modulation Types](#modulation-types)
  - [Amplitude Modulation (AM)](#amplitude-modulation-am)
  - [Frequency Modulation (FM)](#frequency-modulation-fm)
  - [Simple Summation (Mixing)](#simple-summation-mixing)

## Features

*   **Amplitude Modulation (AM):** Modulate a carrier wave's amplitude with a signal wave.
*   **Frequency Modulation (FM):** Modulate a carrier wave's frequency with a signal wave.
*   **Simple Summation (Mixing):** Combine two audio signals by adding their amplitudes.
*   **WAV File Support:** Loads and saves standard WAV audio files.
*   **Signal Normalization:** Automatically normalizes input signals for consistent processing.
*   **Command-Line Interface:** Easy to use via command-line arguments.
*   **Detailed Help:** Provides comprehensive help messages with examples.
*   **File Statistics:** Option to display detailed statistics for input WAV files.
*   **Colored Output:** Supports ANSI colored output for better readability in compatible terminals.
*   **Overwrite Protection:** Prevents accidental overwriting of existing output files.

## Metadata

*   **Author:** Igor Brzeżek
*   **Email:** igor.brzezek@gmail.com
*   **GitHub:** [https://github.com/igorbrzezek](https://github.com/igorbrzezek)
*   **Version:** 1.1.3 (Improved FM modulation audibility)
*   **Date Created:** 2025-12-07

## Requirements

*   Python 3.x
*   `numpy`
*   `scipy`
*   `colorama` (optional, for colored terminal output)

## Installation

You can install the required Python libraries using pip:

```bash
pip install numpy scipy colorama
```

## Usage

### Basic Usage

```bash
python modulation.py -c CARRIER_FILE -s SIGNAL_FILE -e EFFECT -o OUTPUT_FILE [OPTIONS]
```

### Arguments

*   `-c`, `--carrier` CARRIER_FILE (Required): Path to the carrier WAV file.
*   `-s`, `--signal` SIGNAL_FILE (Required): Path to the signal (modulating) WAV file.
*   `-e`, `--effect` {am, fm, sum} (Required): The modulation/mixing effect to apply.
    *   `am`: Amplitude Modulation
    *   `fm`: Frequency Modulation
    *   `sum`: Simple Summation (Mixing)
*   `-o`, `--output` OUTPUT_FILE (Required): Path to the resulting output WAV file.
*   `--stat`: Show parameters (statistics) of the input files.
*   `--color`: Show output with ANSI color codes.
*   `--overwrite`: Overwrite the output file without asking.
*   `-h`: Show simple help message.
*   `--help`: Show detailed help with examples and metadata.

### Examples

1.  **Perform Amplitude Modulation (AM) and save output:**

    ```bash
    python modulation.py -c carrier.wav -s signal.wav -e am -o modulated_am.wav
    ```

2.  **Perform Summation (Mixing) with statistics and force overwrite:**

    ```bash
    python modulation.py -c tone_c.wav -s music_s.wav -e sum -o mixed.wav --stat --overwrite
    ```

3.  **Perform Frequency Modulation (FM):**
    *(Requires a tone for the carrier and a slow tone for the signal to clearly hear the effect.)*

    ```bash
    python modulation.py -c 2000hz_tone.wav -s 10hz_tone.wav -e fm -o modulated_fm.wav
    ```

4.  **Display detailed help:**

    ```bash
    python modulation.py --help
    ```

## Modulation Types

### Amplitude Modulation (AM)

Amplitude Modulation varies the amplitude of a high-frequency carrier wave in accordance with the instantaneous amplitude of the modulating signal. This script implements a standard AM formula: `(1.0 + signal_norm) * carrier_norm`.

### Frequency Modulation (FM)

Frequency Modulation varies the frequency of a carrier wave in accordance with the instantaneous amplitude of the modulating signal. The implementation in this script is enhanced for audibility, using a significant frequency deviation and a synthetic carrier wave for clearer results.

### Simple Summation (Mixing)

Simple Summation, or mixing, combines two audio signals by directly adding their normalized amplitudes. The resulting signal is then re-normalized to prevent clipping. This is useful for layering sounds.