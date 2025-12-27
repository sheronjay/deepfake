import sys
from pathlib import Path
from TTS.api import TTS

# ---- CONFIG ----
MODEL_PATH = "tts_model/checkpoint_80000.pth"
CONFIG_PATH = "tts_model/config.json"
OUTPUT_WAV = "output.wav"
# ----------------

def sinhala_audio(romanized_path):
    # Load TTS model
    tts = TTS(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        gpu=False  # set True if you have CUDA
    )

    # Read romanized text
    try:
        with open(romanized_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        print("Error reading romanized text file:", e)
        sys.exit(1)

    audios_folder = Path("sinhala_audio")
    audios_folder.mkdir(exist_ok=True)

    output_filename = Path(romanized_path).stem + "_sinhala_tts"
    output_path = audios_folder / output_filename

    # Generate audio
    tts.tts_to_file(
        text=text,
        file_path=output_path
    )

    print(f"Audio generated and saved as: {output_path}")

    return output_path + '.wav'

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
