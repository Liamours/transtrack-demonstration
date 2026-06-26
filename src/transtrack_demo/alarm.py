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


def alarm_js_html(audio_bytes, uid=0):
    data = base64.b64encode(audio_bytes).decode("ascii")
    return f'<!--{uid}--><script>new Audio("data:audio/wav;base64,{data}").play();</script>'
