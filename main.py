import sys
from pathlib import Path

from transcribe_video import convert_to_audio, transcribe_audio
from en_to_sin import translate_file
from sin_to_roman import romanize
from sinhala_tts import sinhala_audio
from final_video import join_video_audio

def get_input() -> tuple[Path, str]:
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> <transcribe_model_name>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    model_name = sys.argv[2]

    if not video_path.exists():
        print(f"Error: {video_path} does not exist.")
        sys.exit(1)sinhala_audio

    return video_path, model_name

def main():
    # Get user input for video path and model(audio to english txt) name
    video_path, model_name = get_input()

    # Convert video to audio
    audio_path = convert_to_audio(video_path, video_path.with_suffix('.mp3'))

    # Generate speech-to-text transcription
    transcribe_path = transcribe_audio(audio_path, model_name)

    # Translate transcription to Sinhala
    translated_path = translate_file(transcribe_path)

    # Romanize Sinhala text
    romanized_path = romanize(translated_path)

    # Generate sinhala audio using tts model
    sinhala_wav = sinhala_audio(romanized_path)

    # Join the original video with the new Sinhala audio
    join_video_audio(video_path, sinhala_wav)


if __name__ == "__main__":
    main()