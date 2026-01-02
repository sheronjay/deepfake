import json
import subprocess
from pathlib import Path

def join_segments(json_path: Path, max_speedup: float = 1.25):
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    inputs = []
    filters = []
    mix_inputs = []

    for i, seg in enumerate(segments):
        audio = seg["converted_audio"]
        start = float(seg["start"])
        delay_ms = int(start * 1000)

        # Use precomputed fields from JSON
        ratio = float(seg.get("duration_ratio", 1.0))  # tts_duration / target_duration

        # If TTS is longer than target, ratio > 1.0 => need to speed up
        # Cap to keep speech natural
        speed = min(max(ratio, 0.5), max_speedup)  # atempo supports >=0.5; we cap speed-up

        inputs.append(f'-i "{audio}"')

        # Apply atempo only if needed
        if abs(speed - 1.0) > 0.01:
            filters.append(f'[{i}:a]atempo={speed:.5f},adelay={delay_ms}|{delay_ms}[a{i}]')
        else:
            filters.append(f'[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]')

        mix_inputs.append(f'[a{i}]')

        # Optional: store what we applied (useful for debugging)
        seg["atempo_applied"] = speed

    filter_complex = "; ".join(filters) + ";" + "".join(mix_inputs) + \
        f"amix=inputs={len(segments)}:dropout_transition=0"

    out_dir = Path("joined_sinhala_audio")
    out_dir.mkdir(exist_ok=True)
    output_audio = out_dir / f"{json_path.stem}_final_sinhala_audio.m4a"

    cmd = f"""
    ffmpeg -y {' '.join(inputs)} \
    -filter_complex "{filter_complex}" \
    -c:a aac -b:a 192k "{output_audio}"
    """

    subprocess.run(cmd, shell=True)

    # Save updated JSON with atempo_applied (optional)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    return output_audio

