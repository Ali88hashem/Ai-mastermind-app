import os
import json
import requests
import pytesseract
import cv2
import pandas as pd
import speech_recognition as sr
import pyttsx3
from bs4 import BeautifulSoup
from transformers import pipeline
from PyPDF2 import PdfReader

class AIMastermind:
    def __init__(self):
        # Initialize AI components
        self.tts_engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.summarizer = pipeline("summarization")
        self.translator = pipeline("translation_en_to_fr")
        self.knowledge_base = {}

    def text_to_speech(self, text):
        """Convert text to speech."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def speech_to_text(self):
        """Convert speech to text."""
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
            try:
                return self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Could not understand audio."
            except sr.RequestError:
                return "API request failed."

    def web_scraping(self, url):
        """Scrape text content from a webpage."""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()
        except requests.RequestException as e:
            return f"Error scraping website: {e}"

    def summarize_text(self, text, max_length=50, min_length=10):
        """Summarize text using AI."""
        try:
            return self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        except Exception as e:
            return f"Summarization error: {e}"

    def translate_text(self, text):
        """Translate English text to French."""
        try:
            return self.translator(text)[0]['translation_text']
        except Exception as e:
            return f"Translation error: {e}"

    def extract_text_from_image(self, image_path):
        """Extract text from an image using OCR."""
        try:
            img = cv2.imread(image_path)
            return pytesseract.image_to_string(img)
        except Exception as e:
            return f"OCR error: {e}"

    def analyze_csv(self, csv_path):
        """Analyze CSV file and return a summary."""
        try:
            df = pd.read_csv(csv_path)
            return df.describe()
        except Exception as e:
            return f"CSV processing error: {e}"

    def process_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            return "".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            return f"PDF processing error: {e}"

    def store_knowledge(self, key, value):
        """Store information in memory."""
        self.knowledge_base[key] = value

    def get_knowledge(self, key):
        """Retrieve stored knowledge."""
        return self.knowledge_base.get(key, "No data found.")

    def backup_knowledge(self, backup_path="knowledge_backup.json"):
        """Backup knowledge base to a JSON file."""
        try:
            with open(backup_path, 'w') as file:
                json.dump(self.knowledge_base, file, indent=4)
        except Exception as e:
            return f"Backup error: {e}"

    def restore_knowledge(self, backup_path="knowledge_backup.json"):
        """Restore knowledge base from a backup file."""
        try:
            with open(backup_path, 'r') as file:
                self.knowledge_base = json.load(file)
        except Exception as e:
            return f"Restore error: {e}"

    def install_dependencies(self):
        """Install required dependencies."""
        dependencies = [
            "requests", "pytesseract", "opencv-python", "pandas",
            "speechrecognition", "pyttsx3", "transformers", "beautifulsoup4", "pypdf2"
        ]
        for package in dependencies:
            os.system(f"pip install {package}")

if __name__ == "__main__":
    ai = AIMastermind()
    ai.text_to_speech("Hello, I am AI Mastermind")
    print("Summarized Text:", ai.summarize_text(
        "Artificial Intelligence is transforming the world in multiple fields, including healthcare, finance, and education."
    ))
    print("Translated Text:", ai.translate_text("Hello, how are you?"))