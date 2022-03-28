#import packages
import os 
from google.cloud import speech
import wave

#Shankashana Krishnakumar - 2019772
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'sdgp-voice-recognition-040344760f01.json' #Creating an envirnoment where a json file is assigned to it.
speech_client = speech.SpeechClient() #"speech.SpeechClient" is assigned to a local variable called speech_client.
client = speech.SpeechClient() #"speech.SpeechClient" is assigned to a local variable called client.
media_file_name_wav = 'Record_2.wav' #Assigning  the wav extension file (audio file) to a local variable for further use.

#Opening the Audio file.
with wave.open(media_file_name_wav, "rb") as wave_file:
    frame_rate = wave_file.getframerate() # getting the frame rate of the audio file and assigned to a local variable, "frame_rate".
    channels = wave_file.getnchannels() # getting the channels of the audio file and assigned to a local variable, "channels".

#Opening the Audio file.
with open(media_file_name_wav, 'rb') as f2:
    byte_data_wav = f2.read() #Reading the byte data.
    audio_wav = speech.RecognitionAudio(content=byte_data_wav) #Analyzing the data.

    #Configuring the speech
    config_wav = speech.RecognitionConfig(
    sample_rate_hertz=frame_rate,
    enable_automatic_punctuation=True,
    language_code='en-US',
    audio_channel_count=channels
)


response_standard_wav = speech_client.recognize(
    config=config_wav,
    audio=audio_wav
)

#Display the output
#print(response_standard_wav)

#response = client.recognize(config=config_wav, audio=audio_wav)

#For-loop is used to print accuracy and confidence.
#for result in response.results:
 #   print("Transcript: {}".format(result.alternatives[0].transcript))

  #  f = open("testfile.txt", "w+")
   # f.write(result.alternatives[0].transcript)
