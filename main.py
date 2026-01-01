import sys
from pathlib import Path

from transcribe_video import convert_to_audio, transcribe_audio
from en_to_sin import translate_file
from sin_to_roman import romanize
from sinhala_tts import sinhala_audio
from final_video import join_video_audio
from join_audio_segments import join_segments
from convert_voice import convert_voice_folder

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

    # Convert voice of Sinhala audio segments
    voice_converted_audio_folder = convert_voice_folder(
        input_folder_path=sinhala_audio_folder,
        model_name="mahindasiri_thero_3.pth",
        output_folder_name="voice_converted_sinhala_audio_segments",
        f0_up_key=0,
        f0_method="rmvpe",
        index_rate=0.75,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.25,
        protect=0.33,
        output_format="wav",
    )

    # Join audio segments
    print("Joining audio segments...")
    sinhala_m4a = join_segments(voice_converted_audio_folder)

    # Join the original video with the new Sinhala audio
    print("Joining original video with Sinhala audio...")
    sinhala_video_path = join_video_audio(video_path, sinhala_m4a)
    print(f"Sinhala Video Path: {sinhala_video_path}")



if __name__ == "__main__":
    main()