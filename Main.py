
    def text_to_speech(self, text):
        """Convert text to speech."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def speech_to_text(self, audio):
        """Convert speech to text."""
        try:
            text = self.recognizer.recognize_google(audio)
            .recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from speech recognition service; {e}"