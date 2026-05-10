import base64
import math
import struct
import wave
from io import BytesIO


def alarm_wav_bytes(duration=0.35, frequency=880, rate=44100):
    buffer = BytesIO()
    samples = int(duration * rate)
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        for index in range(samples):
            value = int(16000 * math.sin(2 * math.pi * frequency * index / rate))
            file.writeframes(struct.pack("<h", value))
    return buffer.getvalue()


def autoplay_audio_html(audio_bytes, key=None):
    data = base64.b64encode(audio_bytes).decode("ascii")
    marker = f' data-key="{key}"' if key is not None else ""
    return f'<audio autoplay{marker}><source src="data:audio/wav;base64,{data}" type="audio/wav"></audio>'
