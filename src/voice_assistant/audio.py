# voice_assistant/audio.py

import speech_recognition as sr
import pygame
import time
import logging
import sys
import threading
from pydub import AudioSegment
import simpleaudio as sa

def animate_recording():
    animation = "|/-\\"
    idx = 0
    while recording:
        print(f"\rRecording {animation[idx % len(animation)]}", end="", flush=True)
        idx += 1
        time.sleep(0.1)

def record_audio(file_path, timeout=10, phrase_time_limit=5, retries=3):
    global recording
    recognizer = sr.Recognizer()
    for attempt in range(retries):
        try:
            with sr.Microphone() as source:
                print("Recording started")
                logging.info("Recording started")
                
                recording = True
                animation_thread = threading.Thread(target=animate_recording)
                animation_thread.start()
                
                # Listen for the first phrase and extract it into audio data
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                
                recording = False
                animation_thread.join()
                
                print("\nRecording complete")
                logging.info("Recording complete")
                
                # Save the recorded audio data to a WAV file
                with open(file_path, "wb") as audio_file:
                    audio_file.write(audio_data.get_wav_data())
                print(f"Recording saved to {file_path}")
                logging.info(f"Recording saved to {file_path}")
                return True
        except sr.WaitTimeoutError:
            print(f"\nListening timed out, retrying... ({attempt + 1}/{retries})")
            logging.warning(f"Listening timed out, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            print(f"\nFailed to record audio: {e}")
            logging.error(f"Failed to record audio: {e}")
            if "PyAudio" in str(e):
                print("PyAudio is not installed. Please install it using: pip install pyaudio")
            break
    print("Recording failed after all retries")
    logging.error("Recording failed after all retries")
    return False

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
