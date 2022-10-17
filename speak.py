from asyncore import read
from typing import Counter
import speech_recognition as sr
import playsound
import os
from os import kill
import pyttsx3
import random
from gtts import gTTS

import playsound
from os import kill
import sys



r = sr.Recognizer()

engine=pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)
engine.setProperty('rate',200)

def jarvis(text):
    print(text)
    engine.say(text)
    engine.runAndWait()


def friday_speak(audio_string):
    #status('b')
    audio=str(audio_string)
    tts = gTTS(text=audio, lang='en')
    r = random.randint(1, 20000000)
    audio_file = 'audio-' + str(r) + '.mp3'
    tts.save(audio_file)
    playsound.playsound(audio_file)
    os.remove(audio_file)

tospeak=sys.argv[1]
# friday_speak(str(tospeak))

voicedata=open("voicedata.txt","r")
voicevalueprev=voicedata.read()
if voicevalueprev == tospeak:
    print("same")
else:
    jarvis(str(tospeak))
    voicedata=open("voicedata.txt","w")
    voicedata.write(str(tospeak))
    voicedata.close()