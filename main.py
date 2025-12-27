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
        sys.exit(1)

    return video_path, model_name

def main():
    # Get user input for video path and model(audio to english txt) name
    video_path, model_name = get_input()
    print(f"Video Path: {video_path}")
    print(f"Model Name: {model_name}")

    # Convert video to audio
    print("Converting video to audio...")
    audio_path = convert_to_audio(video_path, video_path.with_suffix('.mp3'))
    print(f"Audio Path: {audio_path}")

    # Generate speech-to-text transcription
    print("Transcribing audio...")
    transcribe_path = transcribe_audio(audio_path, model_name)
    print(f"Transcribe Path: {transcribe_path}")

    # Translate transcription to Sinhala
    print("Translating transcription to Sinhala...")
    translated_path = translate_file(transcribe_path)
    print(f"Translated Path: {translated_path}")

    # Romanize Sinhala text
    print("Romanizing Sinhala text...")
    romanized_path = romanize(translated_path)
    print(f"Romanized Path: {romanized_path}")

    # Generate sinhala audio using tts model
    print("Generating Sinhala audio...")
    sinhala_wav = sinhala_audio(romanized_path)
    print(f"Sinhala Audio Path: {sinhala_wav}")

    # Join the original video with the new Sinhala audio
    print("Joining original video with Sinhala audio...")
    sinhala_video_path = join_video_audio(video_path, sinhala_wav)
    print(f"Sinhala Video Path: {sinhala_video_path}")



if __name__ == "__main__":
    main()