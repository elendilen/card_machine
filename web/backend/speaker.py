# speaker.py
import pyttsx3
import threading

class Speaker:
    def __init__(self, rate=175, volume=1.0):
        self.rate = rate
        self.volume = volume

    def say(self, text):
        print(f"[🔊] 播报: {text}")
        thread = threading.Thread(target=self._speak, args=(text,))
        thread.daemon = True
        thread.start()

    def _speak(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[❌ 播报失败]: {e}")
