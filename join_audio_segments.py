import json
import subprocess
from pathlib import Path

def join_segments(json_path: Path, output_audio: Path):
    with open(json_path, "r") as f:
        segments = json.load(f)

    inputs = []
    filters = []
    mix_inputs = []

    for i, seg in enumerate(segments):
        audio = seg["audio"]
        delay_ms = int(seg["start"] * 1000)

        inputs.append(f'-i "{audio}"')
        filters.append(f'[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]')
        mix_inputs.append(f'[a{i}]')

    filter_complex = "; ".join(filters) + ";" + "".join(mix_inputs) + f"amix=inputs={len(segments)}:dropout_transition=0"

    cmd = f"""
    ffmpeg -y {' '.join(inputs)} \
    -filter_complex "{filter_complex}" \
    -c:a aac -b:a 192k "{output_audio}"
    """

    subprocess.run(cmd, shell=True)


jsonfile = Path("sinhala_audio") / "abella-danger-interview_medium.en_segments_translated_romanized_sinhala_tts_metadata.json"
outputfile = Path("sinhala_audio") / "abella-danger-interview_sinhala_full.m4a"

join_segments(jsonfile, outputfile)
