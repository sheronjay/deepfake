# Deepfake Sinhala Dubbing Pipeline

## Introduction
This repository contains a multi-stage pipeline that takes an input video, transcribes the English audio, translates it to Sinhala, romanizes the Sinhala text, synthesizes Sinhala speech, converts the voice with an RVC model, and finally lip-syncs the output video with Wav2Lip. The main orchestration happens in `main.py`, which chains together the supporting scripts in this repo.

## Pipeline Overview
1. Extract audio from the source video (`transcribe_video.py`).
2. Transcribe English speech with WhisperX and store segment metadata (`transcribe_video.py`).
3. Translate the transcript to Sinhala with NLLB (`en_to_sin.py`).
4. Romanize Sinhala text (`sin_to_roman.py`).
5. Generate Sinhala TTS audio segments (`sinhala_tts.py`).
6. Convert the voice using an RVC model (`convert_voice.py`).
7. Stitch audio segments and mix them into a final track (`join_audio_segments.py`).
8. Replace the source video audio (`final_video.py`).
9. Lip-sync the video with Wav2Lip (`main.py`).

## Setup
This project relies on external projects and model assets that are intentionally ignored from Git. You will need to set up the following prerequisites before running the pipeline.

### System Requirements
- Python 3.10+ (CPU-only works; GPU optional for speed)
- `ffmpeg` available on your PATH

### Python Dependencies
Create a virtual environment for the top-level pipeline and install the key libraries used by the scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install whisperx transformers sentencepiece accelerate sacremoses aksharamukha TTS soundfile
```

> Note: `whisperx` and `transformers` may download large models at runtime. Refer to their docs for GPU support or custom cache locations.

### External Projects
The pipeline expects local clones/virtual environments for the following projects:

#### 1) RVC (Realtime Voice Conversion)
`main.py` calls `convert_voice.py` using a Python interpreter located at `./rvc/bin/python`. Create a virtual environment named `rvc` and install the RVC dependencies there. You also need the model and index assets under `./assets/`.

Expected layout:
```
./rvc/bin/python
./assets/weights/<your_model>.pth
./assets/indices/<your_index>.index
./assets/rmvpe/...
```

#### 2) Wav2Lip
`main.py` runs Wav2Lip’s `inference.py` using a Python interpreter at `./wav2lip/venv-wav2lip/bin/python`. Clone Wav2Lip under `./wav2lip/Wav2Lip`, set up its virtual environment, and download the checkpoint.

Expected layout:
```
./wav2lip/Wav2Lip/inference.py
./wav2lip/Wav2Lip/checkpoints/wav2lip.pth
./wav2lip/venv-wav2lip/bin/python
```

#### 3) Sinhala TTS Model
`./tts_model` is expected to contain the trained checkpoint and config file used by `sinhala_tts.py`.

Expected layout:
```
./tts_model/checkpoint_80000.pth
./tts_model/config.json
```

### Optional: Lip Enhancer
There is an optional post-processing script in `lip-enhancer/` that can be run on Wav2Lip output videos if you want additional smoothing. This is not called from `main.py` by default.

## Running the Full Pipeline
Once the dependencies and external projects are installed, run the pipeline from the repository root:

```bash
python main.py /path/to/input_video.mp4
```

The pipeline will generate intermediate artifacts in folders like `audios/`, `segment_metadata/`, `sinhala_audio_segments/`, `joined_sinhala_audio/`, and `sinhala_video/`.

## Troubleshooting
- Ensure `ffmpeg` is installed and available on your PATH.
- If you see missing model errors, verify that the files in `assets/`, `tts_model/`, and `wav2lip/Wav2Lip/checkpoints/` exist and match the expected filenames in the scripts.
- The RVC and Wav2Lip environments are independent from the main pipeline environment; verify each venv has the proper dependencies.
