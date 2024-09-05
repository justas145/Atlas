# voice_assistant/text_to_speech.py

from openai import OpenAI
from deepgram import DeepgramClient, SpeakOptions
import logging

# Add this variable to store the last spoken text
last_spoken_text = ""

def text_to_speech(model, api_key, text, output_file_path, voice=None, local_model_path=None):
    global last_spoken_text
    
    # Check if the text is a duplicate
    if text == last_spoken_text:
        logging.info("Duplicate text detected. Skipping TTS.")
        return
    
    # Update the last spoken text
    last_spoken_text = text
    
    try:
        if model == 'openai':
            client = OpenAI(api_key=api_key)
            speech_response = client.audio.speech.create(
                model="tts-1",
                voice="nova" if voice is None else voice,
                input=text,
                speed=1.5
            )
            print("Making audio file with openai")

            speech_response.stream_to_file(output_file_path)
            # with open(output_file_path, "wb") as audio_file:
            #     audio_file.write(speech_response['data'])  # Ensure this correctly accesses the binary content
        elif model == 'deepgram':
            client = DeepgramClient(api_key=api_key)
            options = SpeakOptions(
                model="aura-arcas-en" if voice is None else voice,
                encoding="linear16",
                container="wav"
            )
            SPEAK_OPTIONS = {"text": text}
            response = client.speak.rest.v("1").save(output_file_path, SPEAK_OPTIONS, options)
        elif model == 'local':
            # Placeholder for local TTS model
            with open(output_file_path, "wb") as f:
                f.write(b"Local TTS audio data")
        else:
            raise ValueError("Unsupported TTS model")
    except Exception as e:
        logging.error(f"Failed to convert text to speech: {e}")

def reset_last_spoken_text():
    global last_spoken_text
    last_spoken_text = ""
