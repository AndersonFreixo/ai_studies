#Simple GUI agent with speech recognition and synthesis
#(Voxpopuli is a provisory solution for the late)
#The logic is very simple: when the user click on the 'Record' button,
#a thread starts recording the microphone. When the user releases it,
#the audio file is saved on disk for ASR and the result is fed into the agent.
#For it to work you must either login() into HuggingFace or change
#InferenceClientModel to LiteLLMModel in order to run the model locally.

import threading
import tempfile
import queue
import sys
import sounddevice as sd
import soundfile as sf
import torch
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from threading import Thread
from transformers import pipeline
from huggingface_hub import login
from smolagents import tool, CodeAgent, InferenceClientModel, DuckDuckGoSearchTool
from voxpopuli import Voice
from scipy.io.wavfile import read, write
from io import BytesIO

class App:
    def __init__(self):
        #GUI setup related
        self.root = Tk()
        self.root.title("Gui Smolagent")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.img = ImageTk.PhotoImage(Image.open("agent.jpeg").resize((360, 360)))
        panel = ttk.Label(mainframe, image=self.img).grid(column=0,row=0)

        btn = ttk.Button(mainframe, text="Record")
        btn.grid(column=0, row=1)
        btn.bind("<ButtonPress>", self.onPress)
        btn.bind("<ButtonRelease>", self.onRelease)

        #Audio recording related
        device_info = sd.query_devices(sd.default.device[0], 'input')
        self.samplerate = int(device_info['default_samplerate'])
        self.filename = None
        self.queue = queue.Queue()
        self.is_recording = False

        #Speech to text related
        #From https://www.marktechpost.com/2025/09/17/how-to-build-an-advanced-end-to-end-voice-ai-agent-using-hugging-face-pipelines/
        self.asr = pipeline(
            "automatic-speech-recognition",
            model="dominguesm/whisper-tiny-pt",
            device=-1,
            chunk_length_s=30,
            return_timestamps=False
        )
        #Text to speech related
        self.voice = Voice(lang = "br", voice_id = 4, pitch = 70, speed = 100)

        #Agent related
        self.agent = CodeAgent(
            tools=[DuckDuckGoSearchTool],
            model = InferenceClientModel(),
            additional_authorized_imports=['datetime']
        )

    def onPress(self, event):
        self.is_recording = True
        Thread(target = self.record_audio, daemon = True).start()

    def onRelease(self, event):
        self.is_recording = False
        text = self.transcribe(self.filename)
        response = self.agent.run(text)
        self.say(response)

    def record_audio(self):
        self.filename = tempfile.mktemp(prefix='ada_', suffix='.wav', dir='recordings')
        with sf.SoundFile(self.filename, mode='x', samplerate=self.samplerate, channels=1) as file:
            with sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.record_callback):
                while self.is_recording:
                    file.write(self.queue.get())

    def record_callback(self, indata, frames, time, status):
        self.queue.put(indata.copy())

    #From https://www.marktechpost.com/2025/09/17/how-to-build-an-advanced-end-to-end-voice-ai-agent-using-hugging-face-pipelines/
    def transcribe(self, filepath):
        out = self.asr(filepath)
        text = out["text"].strip()
        return text

    def say(self, text):
        wave = self.voice.to_audio(text)
        rate, wave_array = read(BytesIO(wave))
        sd.play(wave_array, rate)
        sd.wait()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    App().run()




