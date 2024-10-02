import re
import logging
from voice_assistant.api_key_manager import get_tts_api_key
from voice_assistant.config import Config
from voice_assistant.text_to_speech import text_to_speech
from voice_assistant.audio import play_audio
from voice_assistant.utils import delete_file

def speak_text(text, voice_mode, voice=None):
    if voice_mode in ["1-way", "2-way"]:
        try:
            output_file = "output.wav"
            text = re.sub(r"FLIGHT(\d+)", r"FLIGHT \1", text)
            text = text.replace("FLIGHT", "flight")
            text = re.sub(r"\bft\b", "feet", text, flags=re.IGNORECASE)
            text = re.sub(r"\bnm\b", "nautical miles", text, flags=re.IGNORECASE)
            text = re.sub(r"\bdeg\b", "degrees", text, flags=re.IGNORECASE)
            text = re.sub(r"\bfpm\b", "feet per minute", text, flags=re.IGNORECASE)
            text = re.sub(r"\bkts\b", "knots", text, flags=re.IGNORECASE)
            text = re.sub(r"\bFL\b", "flight level", text, flags=re.IGNORECASE)
            text = " . . " + text
            tts_api_key = get_tts_api_key()
            # print(f"using voice {voice}")
            text_to_speech(
                Config.TTS_MODEL,
                tts_api_key,
                text,
                output_file,
                voice,
                Config.LOCAL_MODEL_PATH,
            )
            # need to sleep for 1 sec, else there is audio clipping at the beginning
            logging.info(f"Audio file generated: {output_file}")
            # time.sleep(1)
            play_audio(output_file)
            logging.info("Audio playback completed")
            delete_file(output_file)
            logging.info(f"Audio file deleted: {output_file}")
        except Exception as e:
            logging.error(f"Error in speak_text: {e}")
