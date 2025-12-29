import json
import subprocess
from pathlib import Path

def join_segments(json_path: Path):

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

    stt_folder = Path("joined_sinhala_audio")
    stt_folder.mkdir(exist_ok=True)

    # Generate output path in the same directory as input
    output_audio = stt_folder / f"{json_path.stem}_final_sinhala_audio.m4a"

    cmd = f"""
    ffmpeg -y {' '.join(inputs)} \
    -filter_complex "{filter_complex}" \
    -c:a aac -b:a 192k "{output_audio}"
    """

    subprocess.run(cmd, shell=True)
    
    return output_audio

