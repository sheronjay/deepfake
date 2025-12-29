import sys
from pathlib import Path
import subprocess
import json
import whisperx

model_names = ["tiny.en", "base.en", "small.en", "medium.en", "tiny", "base", "small", "medium", "large", "turbo"]

def transcribe_audio(audio_path: Path, model_name: str = "medium.en") -> Path:
    device = "cpu"
    compute_type = "int8"   # fastest/most stable on CPU

    print(f"Loading WhisperX model '{model_name}' on CPU ({compute_type})...")
    model = whisperx.load_model(model_name, device=device, compute_type=compute_type,  vad_method="silero")

    print("Loading audio...")
    audio = whisperx.load_audio(str(audio_path))

    print("Transcribing (rough segments)...")
    result = model.transcribe(audio, language="en")

    print("Loading alignment model...")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)

    print("Aligning (refining timestamps)...")
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device=device,
        return_char_alignments=False
    )

    out_dir = Path("segment_metadata")
    out_dir.mkdir(exist_ok=True)
    segments_file = out_dir / "segment_metadata.json"

    segments = []
    for seg in aligned["segments"]:
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "target_duration": float(seg["end"] - seg["start"]),
            "text": seg["text"].strip(),
            "translation": "",
            "roman": "",
            "audio": "",
            "tts_duration": "",
            "duration_ratio": "",
        })

    segments_file.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {segments_file}")
    return segments_file

def convert_to_audio(video_path: Path) -> Path:
    print("Converting video to audio...")
    # Ensure audios folder exists
    audios_folder = Path("audios")
    audios_folder.mkdir(exist_ok=True)
    
    # Save audio in audios folder
    audio_path = audios_folder / (video_path.stem + ".wav")
    
    command = (
        f'ffmpeg -y -fflags +genpts -i "{video_path}" '
        f'-vn -ac 1 -ar 16000 -c:a pcm_s16le "{audio_path}"'
    )
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