import sys
import json
from pathlib import Path
from TTS.api import TTS


# ---- CONFIG ----
MODEL_PATH = "tts_model/checkpoint_80000.pth"
CONFIG_PATH = "tts_model/config.json"
OUTPUT_WAV = "output.wav"
# ----------------

def sinhala_audio(input_file):
    # Load TTS model
    tts = TTS(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        gpu=False  # set True if you have CUDA
    )

    # Read romanized text
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except Exception as e:
        print("Error reading romanized text file:", e)
        sys.exit(1)

    audios_folder = Path("sinhala_audio_segments")
    audios_folder.mkdir(exist_ok=True)

    audio_segments = []
    for segment in segments:
        text = segment['text']
        output_filename = f"{Path(input_file).stem}_segment_{segment['start']}_{segment['end']}.wav"
        output_path = audios_folder / output_filename

        tts.tts_to_file(
        text=text,
        file_path=str(output_path),
        length_scale=0.85 
        )

        segment['audio'] = str(output_path)

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    return input_file

def main():
    if len(sys.argv) != 2:
        print("Usage: python sinhala_tts.py <text_file.txt>")
        sys.exit(1)

    text_file = sys.argv[1]

    if not Path(text_file).exists():
        print(f"Error: File {text_file} does not exist.")
        sys.exit(1)

    print("Generating Sinhala audio...")
    output_path = sinhala_audio(text_file)
    print(f"Audio generated successfully: {output_path}")

if __name__ == "__main__":
    main()
