import sys
from pathlib import Path
import whisper
import subprocess

model_names = ["tiny.en", "base.en", "small.en", "medium.en", "tiny", "base", "small", "medium", "large", "turbo"]

def transcribe_audio(audio_path: Path, model_name: str) -> str:
    print("Loading model...")
    model = whisper.load_model(model_name)

    print("Transcribing audio...")
    result = model.transcribe(str(audio_path))
    return result["text"]

def convert_to_audio(video_path: Path, audio_path: Path) -> Path:
    print("Converting video to audio...")
    command = "ffmpeg -i {} -vn -ar 44100 -ac 2 -b:a 192k {}".format(video_path, audio_path)
    subprocess.run(command, shell=True)
    return audio_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_video.py <video_file> <model_name>")
        sys.exit(1)

    else:
        if sys.argv[2] not in model_names:
            print(f"Error: Model name must be one of {model_names}")
            sys.exit(1)

    video_file = Path(sys.argv[1])
    model_name = sys.argv[2]

    if not video_file.exists():
        print(f"Error: The file {video_file} does not exist.")
        sys.exit(1)

    # audio_path = convert_to_audio(video_file, video_file.with_suffix('.mp3')) #comment this if the audio is already extracted
    audio_path = video_file.with_suffix('.mp3') #comment this if the audio is not already extracted

    transcribe = transcribe_audio(audio_path, model_name)

    with open(video_file.stem + "_" + model_name + ".txt", 'w') as f:
        f.write(transcribe)
    print("Transcription completed and saved.")


if __name__ == "__main__":
    main()