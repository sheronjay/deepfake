import sys
from TTS.api import TTS

# ---- CONFIG ----
MODEL_PATH = "tts_model/checkpoint_80000.pth"
CONFIG_PATH = "tts_model/config.json"
OUTPUT_WAV = "output.wav"
# ----------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python sinhala_tts.py <text_file.txt>")
        sys.exit(1)

    text_file = sys.argv[1]

    # Read input text
    try:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        print("Error reading text file:", e)
        sys.exit(1)

    if not text:
        print("Text file is empty.")
        sys.exit(1)

    print("trying to load model")
    # Load TTS model
    tts = TTS(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        gpu=False  # set True if you have CUDA
    )
    print("TTS model loaded successfully.")

    # Generate audio
    tts.tts_to_file(
        text=text,
        file_path=OUTPUT_WAV
    )

    print(f"Audio generated and saved as: {OUTPUT_WAV}")

if __name__ == "__main__":
    main()
