import speech_recognition as sr;
r = sr.Recognizer()
with sr.AudioFile("") as source:
    audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        print(text)
    except sr.UnknownValueError:
        print("Audio cannot be translated.")
    except sr.RequestError as e:
        print("Invalid request : {0}".format(e))
    except sr.URLError:
        print("The link is invalid.")
    except sr.HTTPError:
        print("The site is invalid.")
    except sr.WaitTimeoutError:
        print("The conversion has crashed.")