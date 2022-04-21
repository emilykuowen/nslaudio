import pyaudio
from scipy import signal
import numpy as np
import time
import wave

wf = wave.open('audio_sources/sin_440.wav', 'rb')
p = pyaudio.PyAudio()

# https://people.csail.mit.edu/hubert/pyaudio/docs/#class-stream
def callback(in_data, frame_count, time_info, status):
    audio_data = wf.readframes(4096)
    # fulldata = np.append(fulldata, audio_data) #saves filtered data in an array
    return (audio_data, pyaudio.paContinue)

stream = p.open(format = p.get_format_from_width(wf.getsampwidth()), # 16-bit int
                channels = 1, 
                rate = wf.getframerate(), # 44100 Hz
                output = True,
                stream_callback=callback)

stream.start_stream()

while stream.is_active():
    stream.stop_stream()
stream.close()
