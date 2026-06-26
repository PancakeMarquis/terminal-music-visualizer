import os
from time import sleep
import wave
import numpy as np
from playsound3 import playsound
from time import perf_counter

def main():
    is_running: bool = True
    
   
    file = get_audio_file()
    play_audio(file)
    sleep(.8)

    audio, sample_rate = load_audio(file)

    chunk_size = 1024
    chunk_duration = chunk_size / sample_rate

    start_time = perf_counter()
    chunk_index = 0

    try:
        
        while is_running:
            chunk = get_next_chunk(audio, chunk_size)
            if chunk is None:
                break

            bars = analyze_chunk(chunk, sample_rate)

            run_visualizer(bars)

            chunk_index += 1

            target_time = chunk_index * chunk_duration
            elapsed = perf_counter() - start_time

            if target_time > elapsed:
                sleep(target_time - elapsed)

        audio.close()
        

    except KeyboardInterrupt:
        print(f"Stopped playing {file}")
    
def get_audio_file() -> str:
    file: str = input("Choose an audio file: ")
    return file

def load_audio(file: str):
    audio = wave.open(file)
    sample_rate = audio.getframerate()
    return audio, sample_rate
    

def get_next_chunk(audio, chunk_size):
    frames = audio.readframes(chunk_size)
    if len(frames) == 0:
        return None
    
    samples = np.frombuffer(frames, dtype=np.int16)

    return samples


def analyze_chunk(chunk, sample_rate):
    if chunk is None:
        return None
    
    chunk = chunk.astype(np.float32)
    chunk -= np.mean(chunk)

    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), d=1/sample_rate)

    bands = [
        (20, 250),
        (250, 1000),
        (1000, 4000),
        (4000, 20000)
    ]

    bars = []
    for low, high in bands:
            mask = (freqs >= low) & (freqs < high)

            if np.any(mask):
                magnitude = np.mean(spectrum[mask])
            else:
                magnitude = 0

            bars.append(min(int(np.log10(magnitude + 1) * 3), 20))

    return bars

def play_audio(file):
    playsound(file, False)

def run_visualizer(bars):
    os.system("clear")
    draw_bars(bars)
    
    
    

def draw_bars(bars):
    labels = ["Bass", "Low", "Mid", "High"]
    for label, height in zip(labels, bars):
        print(f"{label:5}{'#' * height}")


if __name__ == "__main__":
    main()
