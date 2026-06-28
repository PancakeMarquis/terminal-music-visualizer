import os
from time import sleep
import numpy as np
import soundfile as sf
import pygame
from colorama import Fore, Back, Style

def main():
    pygame.init()
    file = get_audio_file()
    

    audio, sample_rate = load_audio(file)
    pygame.mixer.init(frequency= sample_rate)

    play_audio(file)

    chunk_size = 1024
    last_chunk = -1

    chunk_duration = chunk_size / sample_rate

    try:
        
        while pygame.mixer.music.get_busy():
            position_ms = pygame.mixer.music.get_pos()

            if position_ms < 0:
                continue

            elapsed = position_ms / 1000.0

            chunk_index = int(elapsed * sample_rate / chunk_size)

            if chunk_index == last_chunk:
                sleep(.005)
                continue

            last_chunk = chunk_index

            start = chunk_index * chunk_size
            end = start + chunk_size

            if start >= len(audio):
                break

            chunk = audio[start:end]

            if chunk.ndim == 2:
                chunk = chunk.mean(axis=1)

            bars = analyze_chunk(chunk, sample_rate)

            run_visualizer(bars)
    except KeyboardInterrupt:
        print(f"Stopped playing {file}")
    finally:
        pygame.mixer.music.stop()
        pygame.quit()

    
def get_audio_file() -> str:
    file: str = input("Choose an audio file: ")
    return file

def load_audio(file: str):
    audio, sample_rate = sf.read(file, dtype="float32")
    return audio, sample_rate
    

def analyze_chunk(chunk, sample_rate):
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

    bands = [
        (20, 250),
        (250, 1000),
        (1000, 4000),
        (4000, 20000)
    ]

    bars = []
    for low, high in bands:
            mask = (freqs >= low) & (freqs < high)

            band = spectrum[mask]

            if len(band):
                magnitude = np.sqrt(np.mean(band ** 2))
            else:
                magnitude = 0

            bars.append(min(int((magnitude ** 0.5) * 20), 20))

    return bars

def play_audio(file):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

def run_visualizer(bars):
    print("\033[2J\033[H", end="")
    draw_bars(bars)
    
    
def draw_bars(bars):
    labels = ["Bass", "Low", "Mid", "High"]
    colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN]

    for label, height, color in zip(labels, bars, colors):
        print(f"{label:5}{color}{'█' * height}")


if __name__ == "__main__":
    main()
