import subprocess
from pathlib import Path

def join_video_audio(video_path, audio_path):

    output_filename =  Path(video_path).stem + "_sinhala.mp4"
    output_path = Path("sinhala_video") / output_filename

    command = [
    "ffmpeg",
    "-y",                       # Overwrite output if exists
    "-i", video_path,
    "-i", audio_path,
    "-map", "0:v:0",            # Take video from the first input
    "-map", "1:a:0",            # Take audio from the second input
    "-c:v", "copy",             # Copy video stream without re-encoding
    "-c:a", "aac",              # Encode audio to AAC
    output_path
    ]

    subprocess.run(command, check=True)
