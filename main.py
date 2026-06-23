import random
import os
from time import sleep
import wave
import numpy as np


def main():
    is_running: bool = True
    
   
    file = get_audio_file()
    print(f"Playing {file}")

    audio = load_audio(file)
    try:
        
        while is_running:
            chunk = get_next_chunk(audio)
            if chunk == None:
                break
            bars = analyze_chunk(chunk)
            run_visualizer(bars)

            
            

    except KeyboardInterrupt:
        print(f"Stopped playing {file}")
    
def get_audio_file() -> str:
    file: str = input("Choose an audio file: ")
    return file

def load_audio(file: str):
    audio = wave.open(file)
    return audio
    

def get_next_chunk(audio):
    frames = audio.readframes(1024)

    if len(frames) == 0:
        return None
    
    samples = np.frombuffer(frames, dtype=np.int16)

    return samples


def analyze_chunk(chunk):
    if chunk is None:
        return None
    amplitude = np.mean(np.abs(chunk))

    bars = int(amplitude / 32767 * 20)

    return bars

def play_audio(file_data):
    pass

def run_visualizer(bars):
    os.system("clear")
    draw_bars(bars)
    sleep(.023)
    
    

def draw_bars(bars: int):
    bar = ""
    for num in range(bars):
        bar += "#"
    print(bar)


if __name__ == "__main__":
    main()
