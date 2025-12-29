import sys
from pathlib import Path
import whisper_timestamped as whisper_ts
import subprocess
import json

model_names = ["tiny.en", "base.en", "small.en", "medium.en", "tiny", "base", "small", "medium", "large", "turbo"]

def transcribe_audio(audio_path: Path, model_name: str = "medium.en") -> Path:
    print("Loading model...")
    model = whisper_ts.load_model(model_name)

    print("Transcribing audio (sentence-level timestamps)...")
    result = whisper_ts.transcribe(
        model,
        str(audio_path),
        language="en",
        vad=True,                 # improves segment boundaries
        detect_disfluencies=False
    )

    # Ensure output folder exists
    stt_folder = Path("segment_metadata")
    stt_folder.mkdir(exist_ok=True)

    segments_file = stt_folder / "segment_metadata.json"

    segments = []

    for seg in result["segments"]:
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip(),
            "translation": "",
            "roman": "",
            "audio": "",
        })

    with open(segments_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"Saved:")
    print(f" - Transcript: {text_file}")
    print(f" - Segments:   {segments_file}")

    # Return segment JSON (perfect input for translate → TTS → ffmpeg)
    return segments_file

def convert_to_audio(video_path: Path, audio_path: Path) -> Path:
    print("Converting video to audio...")
    # Ensure audios folder exists
    audios_folder = Path("audios")
    audios_folder.mkdir(exist_ok=True)
    
    # Save audio in audios folder
    audio_path = audios_folder / audio_path.name
    
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

    audio_path = convert_to_audio(video_file, video_file.with_suffix('.mp3')) #comment this if the audio is already extracted
    # audio_path = video_file.with_suffix('.mp3') #comment this if the audio is not already extracted

    transcribe = transcribe_audio(audio_path, model_name)


if __name__ == "__main__":
    main()