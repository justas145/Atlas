# voice_assistant/audio.py

import speech_recognition as sr
import pygame
import time
import logging
from pydub import AudioSegment
import simpleaudio as sa

def record_audio(file_path, timeout=10, phrase_time_limit=5, retries=3):
    """
    Record audio from the microphone and save it as a WAV file.
    
    Args:
    file_path (str): The path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    retries (int): Number of retries if recording fails.
    """
    recognizer = sr.Recognizer()
    for attempt in range(retries):
        try:
            with sr.Microphone() as source:
                logging.info("Recording started")
                # Listen for the first phrase and extract it into audio data
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logging.info("Recording complete")
                # Save the recorded audio data to a WAV file
                with open(file_path, "wb") as audio_file:
                    audio_file.write(audio_data.get_wav_data())
                return
        except sr.WaitTimeoutError:
            logging.warning(f"Listening timed out, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            logging.error(f"Failed to record audio: {e}")
            break
    else:
        logging.error("Recording failed after all retries")
# def play_audio(file_path):
#     """
#     Play an audio file using pygame with minimal delay.

#     Args:
#     file_path (str): The path to the audio file to play.
#     """
#     try:
#         pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
#         pygame.mixer.music.load(file_path)
#         pygame.mixer.music.set_volume(1.0)
#         pygame.mixer.music.play(start=0.0)

#         # Wait for the audio to finish playing
#         while pygame.mixer.music.get_busy():
#             pygame.time.Clock().tick(10)

#         pygame.mixer.quit()
#     except pygame.error as e:
#         logging.error(f"Failed to play audio: {e}")
#     except Exception as e:
#         logging.error(f"An unexpected error occurred while playing audio: {e}")


def play_audio(file_path):
    """
    Play an audio file with minimal delay using pydub and simpleaudio.

    Args:
    file_path (str): The path to the audio file to play.
    """
    try:
        # Load an audio file from path
        audio = AudioSegment.from_file(file_path)

        # Add 100ms of silence at the beginning
        silence = AudioSegment.silent(duration=2000)
        audio = silence + audio

        # Convert audio to raw data bytes
        audio_data = audio.raw_data

        # Get the number of channels and sample width from the audio
        num_channels = audio.channels
        bytes_per_sample = audio.sample_width

        # Create a simpleaudio playback object
        play_obj = sa.play_buffer(
            audio_data, num_channels, bytes_per_sample, audio.frame_rate
        )

        # Wait for the audio to finish playing
        play_obj.wait_done()

    except Exception as e:
        print(f"An error occurred while playing audio: {e}")
