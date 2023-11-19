# file containing required functions for processing audio files 
# and extracting features from them
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

# trims the given audio length to 3 seconds
# audio - audio signal
# sr - sampling rate
def trimAudio(audio,sr, targetLength=3):
    # targetLength should be in seconds
    if audio is None:
      return None
    if len(audio)<=targetLength*sr:
      return audio
    
    return audio[0:targetLength*sr]

# perform required pre-processing like background noise removal, trimming
# and so on
def preProcessAudio(audio, sr):
    # time the audio to a 3 second length
    trimmed_audio = trimAudio(audio, sr, targetLength=3)
    
    if trimmed_audio is None:
      return None
    
    # perform backgroun noise removal
    S = nr.reduce_noise(y=trimmed_audio, sr=sr)
    return S

# function to calculate mfcc co-efficents
# n_mfcc - number of mfcc co-efficients
def mfcc(audio, sr, n_mfcc = 20):
    if audio is None:
      return None
    s = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return s
    

# function to calculate mel spectrogram of the input audio
# audio - aduio signal
# n_fft - length of the window used for fast fourier transform
def melSpectrogram(audio, sr, n_fft=300):
    if audio is None:
      return None
    s = librosa.feature.melspectrogram(y = audio, s=sr, n_fft= n_fft)
    return s

# function that takes the filename and returns the extracted feature
def extract_audio_feature(audioPath, feature = 'mfcc', n_fft =300, n_mfcc = 20):
    audio, sr = librosa.load(audioPath)
    
    if feature =='mfcc':
      return mfcc(audio, sr, n_mfcc)
    elif feature == "melSpectrogram":
      return melSpectrogram(audio, sr, n_fft)
    
    return None
  

# Todo - calculating delta mfcc and delta2 mfcc

