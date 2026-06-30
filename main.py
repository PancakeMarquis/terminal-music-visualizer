import os
from time import sleep
import numpy as np
import soundfile as sf
import pygame
from colorama import Fore
import sys

PEAK_HOLD_FRAMES = 20
PEAK_FALL_SPEED = 0.1
SMOOTHING_FACTOR = .3
CHUNK_SIZE = 1024
UI_RESERVED_LINES = 5
UI_RESERVED_COLUMN = 2
WAIT_TO_SKIP = .005

def main():
    pygame.init()
    audio_file= get_audio_file()
    

    audio, sample_rate = load_audio(audio_file)
    

    pygame.mixer.init(frequency= sample_rate)

    play_audio(audio_file)

    last_chunk = -1
    terminal_size = os.get_terminal_size()
    
    usable_width = terminal_size.columns - UI_RESERVED_COLUMN
    num_bands = usable_width // 2

    display_height = [0.0]*num_bands
    peak_height = [0.0]*num_bands
    peak_hold = [0] *num_bands
    
    try:
        
        while pygame.mixer.music.get_busy():
            position_ms = pygame.mixer.music.get_pos()
            terminal_size = os.get_terminal_size()
            usable_width = terminal_size.columns - UI_RESERVED_COLUMN
            new_num_bands = usable_width // 2
            max_height = terminal_size.lines - UI_RESERVED_LINES

            if new_num_bands != num_bands:
                num_bands = new_num_bands
                display_height = resize_list(display_height, num_bands)
                peak_height = resize_list(peak_height, num_bands)
                peak_hold = resize_list(peak_hold, num_bands, 0)

            if position_ms < 0:
                continue

            elapsed = position_ms / 1000.0

            chunk_index = int(elapsed * sample_rate / CHUNK_SIZE)

            if chunk_index == last_chunk:
                sleep(WAIT_TO_SKIP)
                continue

            last_chunk = chunk_index

            start = chunk_index * CHUNK_SIZE
            end = start + CHUNK_SIZE

            if start >= len(audio):
                break

            chunk = audio[start:end]

            if chunk.ndim == 2:
                chunk = chunk.mean(axis=1)

            new_height = analyze_chunk(chunk, sample_rate, num_bands, max_height)

            for i in range(len(display_height)):
                display_height[i] += (new_height[i] - display_height[i]) * SMOOTHING_FACTOR
                if display_height[i] >= peak_height[i]:
                    peak_height[i] = display_height[i]
                    peak_hold[i] = PEAK_HOLD_FRAMES
                else:
                    if peak_hold[i] > 0:
                        peak_hold[i] -= 1
                    else:
                        peak_height[i] -= PEAK_FALL_SPEED
                        peak_height[i] = max(peak_height[i], display_height[i])
            run_visualizer(display_height, peak_height, audio_file, max_height)
    except KeyboardInterrupt:
        print(f"Stopped playing {audio_file}")
    finally:
        pygame.mixer.music.stop()
        pygame.quit()

    
def get_audio_file():
    file: str = input("Choose an audio file: ")
    return file

def load_audio(file: str):
    try:
        audio, sample_rate = sf.read(file, dtype="float32")
        return audio, sample_rate
    except RuntimeError:
        print("Couldn't open audio file.")
        sys.exit(1)

def resize_list(lst, new_size, fill=0.0):
    if len(lst) < new_size:
        return lst + [fill] * (new_size - len(lst))
    return lst[:new_size]    

def analyze_chunk(chunk, sample_rate, num_bands, max_height):
    if chunk is None:
        return None
    chunk = chunk.copy()
    chunk -= np.mean(chunk)

    window = np.hanning(len(chunk))
    chunk *= window

    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), d=1/sample_rate)

    max_mag = np.max(spectrum)

    if max_mag > 0:
        spectrum /= max_mag

    bands = np.logspace(np.log10(20), np.log10(20000), num_bands + 1)

    bars = []
    for low, high in zip(bands[:-1], bands[1:]):
            mask = (freqs >= low) & (freqs < high)

            band = spectrum[mask]

            if len(band):
                magnitude = np.sqrt(np.mean(band ** 2))
            else:
                magnitude = 0

            bars.append(min(int((magnitude ** 0.5) * max_height), max_height))

    return bars

def play_audio(file):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

def run_visualizer(bars, peak_height, audio_file_name, max_height):
    sys.stdout.write("\033[H")   # Home
    sys.stdout.write("\033[J")   # Clear to end
    draw_bars(bars, peak_height, audio_file_name, max_height)
    sys.stdout.flush()
    
    
def draw_bars(bars, peak_height, audio_file_name, max_height):
    colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN]


    for row in range(max_height, 0, -1):
        for i, height in enumerate(bars):
            color = colors[i % len(colors)]
            if height >= row:
                print(f"{color}█ ", end="")
            elif row == round(peak_height[i]):
                print(f"{color}▲ ", end="")
            else:
                print("  ", end="")
        print()
    print("─" * (len(bars) * 2))
    print(f"Now Playing: {audio_file_name}")


if __name__ == "__main__":
    main()
