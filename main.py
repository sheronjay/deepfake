import sys
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv

from transcribe_video import convert_to_audio, transcribe_audio
from en_to_sin import translate_file
from sin_to_roman import romanize
from sinhala_tts import sinhala_audio
from final_video import join_video_audio
from join_audio_segments import join_segments

# Load environment variables
load_dotenv()

def get_input() -> tuple[Path, str]:
    if len(sys.argv) < 1:
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"Error: {video_path} does not exist.")
        sys.exit(1)

    return video_path

def main():
    # Get user input for video path
    video_path = get_input()
    print(f"Video Path: {video_path}")

    # Convert video to audio
    print("Converting video to audio...")
    audio_path = convert_to_audio(video_path)
    print(f"Audio Path: {audio_path}")

    # Generate speech-to-text transcription
    print("Transcribing audio...")
    transcribe_path = transcribe_audio(audio_path)
    print(f"Transcribe Path: {transcribe_path}")

    # Translate transcription to Sinhala
    print("Translating transcription to Sinhala...")
    translated_path = translate_file(transcribe_path)
    print(f"Translated Path: {translated_path}")

    # Romanize Sinhala text
    print("Romanizing Sinhala text...")
    romanized_path = romanize(translated_path)
    print(f"Romanized Path: {romanized_path}")

    # Generate sinhala audio using tts model
    print("Generating Sinhala audio...")
    sinhala_wav_segments, sinhala_audio_folder = sinhala_audio(romanized_path)
    print(f"Sinhala Audio Path: {sinhala_wav_segments}")

    # Convert voice of Sinhala audio segments using rvc venv
    print("Converting voice using RVC model...")
    project_root = Path(__file__).parent
    rvc_python = project_root / "rvc" / "bin" / "python"
    convert_script = project_root / "convert_voice.py"
    
    # Run convert_voice.py in the rvc virtual environment
    # Arguments: <metadata_json> <model_name> [<index_file_name>]
    # Provide the index file path under assets/indices so the convert script can locate it
    index_file = project_root / "assets" / "indices" / "added_IVF1281_Flat_nprobe_1_mahindasiri_thero_4_v1.index"
    result = subprocess.run(
        [
            str(rvc_python),
            str(convert_script),
            str(sinhala_wav_segments),  # metadata JSON path
            "mahindasiri_thero_4.pth",  # model name
            str(index_file),  # index file (optional)
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )
    
    if result.returncode != 0:
        print(f"Error during voice conversion: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    voice_converted_audio_folder = Path(sinhala_audio_folder) / "voice_converted_sinhala_audio_segments"

    # Join audio segments
    print("Joining audio segments...")
    sinhala_m4a = join_segments(sinhala_wav_segments)

    # Join the original video with the new Sinhala audio
    print("Joining original video with Sinhala audio...")
    sinhala_video_path = join_video_audio(video_path, sinhala_m4a)
    print(f"Sinhala Video Path: {sinhala_video_path}")

    # Add lip sync using Wav2Lip
    print("Adding lip sync...")
    wav2lip_dir = Path(os.getenv("wav2lip_path"))
    wav2lip_venv_path = Path(os.getenv("wav2lip_venv_path"))
    inference_script = wav2lip_dir / "inference.py"
    checkpoint_path = wav2lip_dir / "checkpoints" / "wav2lip.pth"
    
    # Output path for lip-synced video
    output_video = sinhala_video_path.parent / f"{sinhala_video_path.stem}_lipsynced.mp4"
    
    # Convert paths to absolute paths
    abs_video_path = sinhala_video_path.resolve()
    abs_audio_path = sinhala_m4a.resolve()
    abs_output_path = output_video.resolve()
    
    # Run inference.py in the wav2lip virtual environment with proper activation
    # We need to source the venv activate script before running the command
    wav2lip_command = (
        f"source {wav2lip_venv_path}/bin/activate && "
        f"python {inference_script} "
        f"--checkpoint_path {checkpoint_path} "
        f"--face {abs_video_path} "
        f"--audio {abs_audio_path} "
        f"--outfile {abs_output_path}"
    )
    
    result = subprocess.run(
        wav2lip_command,
        shell=True,
        cwd=str(wav2lip_dir),
        executable="/bin/bash"
    )
    
    if result.returncode != 0:
        print(f"Error during lip sync. Check output above.")
        sys.exit(1)
    print(f"Lip-synced Video Path: {output_video}")
    print("Process completed successfully!")


if __name__ == "__main__":
    main()