# Create a function to convert the conversation to a text to speech audio
# By default, there are two speakers, but it can be extended to more speakers.
# For TTS, we use the API of fish.audio
# The conversation is in a text file, and each line is a message with speaker: message
# Example:
# John: Hello, how are you?
# Jane: I'm good, thank you.
# John: What is your name?
# Jane: My name is Jane.

# The converstation is in a file with the above format.
# Need to first figure out the speakers, and then convert each line to audio.
# Each line may generate a different audio file.
# Keep the order of the lines in the conversation file. 
# The output is a folder with the audio files.


"""
Example: reference_id.py
from fish_audio_sdk import Session, TTSRequest, ReferenceAudio

session = Session("your_api_key")

# Option 1: Using a reference_id
with open("output1.mp3", "wb") as f:
    for chunk in session.tts(TTSRequest(
        reference_id="MODEL_ID_UPLOADED_OR_CHOSEN_FROM_PLAYGROUND",
        text="Hello, world!"
    )):
        f.write(chunk)
"""

from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from pathlib import Path
import logging
import time

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ConversationLine:
    """Represents a single line in the conversation"""
    speaker: str
    message: str
    line_number: int


class ConversationParser:
    """Handles parsing of conversation files"""
    
    @staticmethod
    def parse_line(line: str) -> Optional[Tuple[str, str]]:
        """Parse a single line into speaker and message"""
        logger.debug(f"Parsing line: {line.strip()}")
        match = re.match(r'(.*?):\s*(.*)', line.strip())
        if not match:
            logger.debug(f"No speaker found, using default speaker for line: {line.strip()}")
            return ("DEFAULT_SPEAKER", line.strip())
        return match.groups()
    
    @classmethod
    def parse_file(cls, file_path: str) -> List[ConversationLine]:
        """Parse entire conversation file"""
        logger.info(f"Starting to parse conversation file: {file_path}")
        lines = []
        total_lines = 0
        valid_lines = 0
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                total_lines += 1
                if not line.strip():
                    logger.debug(f"Skipping empty line at position {i+1}")
                    continue
                    
                result = cls.parse_line(line)
                if result:
                    speaker, message = result
                    lines.append(ConversationLine(
                        speaker=speaker,
                        message=message,
                        line_number=i
                    ))
                    valid_lines += 1
                    logger.debug(f"Successfully parsed line {i+1}: Speaker={speaker}")
        
        logger.info(f"Finished parsing file. Total lines: {total_lines}, Valid conversations: {valid_lines}")
        return lines


class AudioGenerator:
    """Handles text-to-speech conversion"""
    
    def __init__(self, api_key: str):
        logger.info("Initializing Audio Generator")
        self.session = Session(api_key)
    
    def generate_audio(self, text: str, model_id: str, output_path: str):
        """Generate audio file from text using specified model"""
        logger.info(f"Generating audio for text: '{text[:50]}...' using model: {model_id}")
        start_time = time.time()
        
        try:
            with open(output_path, "wb") as f:
                if model_id:
                    for chunk in self.session.tts(TTSRequest(
                        reference_id=model_id,
                        text=text,
                        request_headers={"Content-Type": "application/json", "model": "speech-1.6"}
                    ), backend="speech-1.6"):
                        f.write(chunk)
                else:
                    for chunk in self.session.tts(
                        TTSRequest(
                            reference_id="0ab6ea77e6684c76b57bada57c414698",
                            text=text,
                            request_headers={"Content-Type": "application/json", "model": "speech-1.6"}
                        ),
                        backend="speech-1.6"
                    ):
                        f.write(chunk)
            
            duration = time.time() - start_time
            logger.info(f"Audio generation completed in {duration:.2f} seconds: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate audio: {str(e)}")
            raise


class ConversationToSpeech:
    """Main class coordinating the conversion process"""
    
    def __init__(self, api_key: str, speaker_models: Dict[str, str]):
        logger.info("Initializing Conversation To Speech converter")
        self.speaker_models = speaker_models
        self.audio_generator = AudioGenerator(api_key)
        self.parser = ConversationParser()
        logger.info(f"Configured for speakers: {', '.join(speaker_models.keys())}")
    
    def validate_speaker(self, speaker: str) -> bool:
        """Check if we have a model for the speaker"""
        is_valid = speaker in self.speaker_models
        if not is_valid:
            logger.warning(f"No model found for speaker: {speaker}")
        return is_valid
    
    def get_output_filename(self, idx: int, line: ConversationLine, output_dir: Path) -> Path:
        """Generate output filename for a conversation line"""
        return output_dir / f"{idx}_{line.speaker}.mp3"
    
    def process_conversation(self, conversation_file: str, output_dir: str) -> None:
        """Process entire conversation file"""
        logger.info(f"Starting conversation processing")
        logger.info(f"Input file: {conversation_file}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Parse conversation
        lines = self.parser.parse_file(conversation_file)
        logger.info(f"Found {len(lines)} lines to process")
        
        # Process each line
        successful_conversions = 0
        failed_conversions = 0
        
        idx = 0
        for line in lines:
            progress_pct = (idx / len(lines)) * 100
            logger.info(f"Processing line {idx+1}/{len(lines)} ({progress_pct:.1f}%): Speaker={line.speaker}")

            # Skip empty lines
            if not line.message.strip():
                logger.debug(f"Skipping empty line at position {idx}")
                continue
            
            if not self.validate_speaker(line.speaker):
                failed_conversions += 1
                logger.error(f"Skipping line {idx+1}/{len(lines)} ({progress_pct:.1f}%): Speaker={line.speaker} not found in speaker models")

            output_file = self.get_output_filename(idx, line, output_path)
            model_id = self.speaker_models.get(line.speaker) or ""
            
            try:
                self.audio_generator.generate_audio(
                    text=line.message,
                    model_id=model_id,
                    output_path=str(output_file)
                )
                successful_conversions += 1
                logger.info(f"Successfully generated audio for line {idx+1}/{len(lines)} ({progress_pct:.1f}%)")
            except Exception as e:
                failed_conversions += 1
                logger.error(f"Error generating audio for line {idx+1}/{len(lines)} ({progress_pct:.1f}%): {e}")
            finally:
                idx += 1
        
        # Final summary
        logger.info("=== Conversion Complete ===")
        logger.info(f"Total lines processed: {len(lines)}")
        logger.info(f"Successful conversions: {successful_conversions}")
        logger.info(f"Failed conversions: {failed_conversions}")
        logger.info(f"Output files can be found in: {output_dir}")


def main():
    logger.info("=== Starting Text-to-Speech Conversion ===")
    
    # Configuration
    API_KEY = "ba286c071de8436197dd53e0c49849e7"  # Using your existing API key
    
    # You'll need to replace these with actual model IDs from Fish Audio
    SPEAKER_MODELS = {
        "求索者": "0ab6ea77e6684c76b57bada57c414698",  # 齐静春 https://fish.audio/m/0ab6ea77e6684c76b57bada57c414698/
        # "求索者": "29a7eea9ed484ef6b175da9bcfb49979",  # 自然流男 https://fish.audio/m/29a7eea9ed484ef6b175da9bcfb49979/
        # "引导者": "d8456df652bb4dc6ba687120065d0be2",   # 道士 https://fish.audio/m/d8456df652bb4dc6ba687120065d0be2/
        "引导者": "7b422cadfad046fe9970596366324f40",  # 老子 https://fish.audio/m/7b422cadfad046fe9970596366324f40/
        
    }

    conversation_file = "narrative-self.txt"
    output_dir = "output_audio"

    conversation_file = os.path.expanduser("~/Downloads/" + conversation_file)
    output_dir = os.path.expanduser("~/Downloads/" + output_dir)

    if not os.path.exists(output_dir):
        logger.info("Output directory does not exist, creating it")
        os.makedirs(output_dir)
    
    try:
        # Initialize converter
        converter = ConversationToSpeech(
            api_key=API_KEY,
            speaker_models=SPEAKER_MODELS
        )
        
        # Process conversation
        converter.process_conversation(
            conversation_file=conversation_file,
            output_dir=output_dir
        )
        
        logger.info("=== Processing Completed Successfully ===")
    except Exception as e:
        logger.error(f"=== Processing Failed: {str(e)} ===")
        raise


if __name__ == "__main__":
    main()


