#!/usr/bin/env python3
"""
RVC training data preparation script (constants only).

Pipeline:
1) Load WAV
2) Convert to mono
3) Resample to 44.1kHz
4) Peak-normalize
5) Split by silence
6) Enforce clip length limits
7) Trim leading/trailing silence
8) Save to ./train_data

Dependencies:
    pip install numpy soundfile scipy
"""

from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# =======================
# CONFIG (CONSTANTS)
# =======================
INPUT_WAV = ["life_is_suffering__buddhism_in_english.wav", "should_all_buddhists_meditate__buddhism_in_english.wav", "what_does_it_mean_to_be_happy_in_life__buddhism_in_english.wav", "why_do_bad_things_happen_to_good_people__buddhism_in_english.wav", "this_is_why_people_fail_to_achieve_the_goals__buddhism_in_english.wav", "learn_to_be_alone__buddhism_in_english.wav"]
OUTPUT_DIR = "train_data"
FILENAME_PREFIX = "mahindasiri_thero"

TARGET_SR = 44100
TARGET_PEAK = 0.95

SILENCE_THRESHOLD_DB = -35.0
MIN_SILENCE_MS = 600
PAD_MS = 80

MIN_CLIP_LEN_S = 2.5
MAX_CLIP_LEN_S = 15.0

EDGE_TRIM_DB = -40.0
MAX_EDGE_TRIM_MS = 300

FRAME_MS = 30
HOP_MS = 10
# =======================


def to_mono(x):
    return x if x.ndim == 1 else np.mean(x, axis=1)


def resample_audio(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def peak_normalize(x, peak):
    m = np.max(np.abs(x))
    return x if m < 1e-9 else (x * (peak / m)).astype(np.float32)


def frame_rms_db(x, frame_len, hop_len):
    n = 1 + max(0, len(x) - frame_len) // hop_len
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        f = x[i * hop_len : i * hop_len + frame_len]
        rms[i] = 20 * np.log10(np.sqrt(np.mean(f**2) + 1e-12))
    return rms


def split_by_silence(x, sr):
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    min_sil_frames = int(MIN_SILENCE_MS / HOP_MS)
    pad = int(sr * PAD_MS / 1000)

    rms_db = frame_rms_db(x, frame_len, hop_len)
    silent = rms_db < SILENCE_THRESHOLD_DB

    segments = []
    in_speech = False
    start = 0
    silence_run = 0

    for i, s in enumerate(silent):
        silence_run = silence_run + 1 if s else 0

        if not in_speech and not s:
            in_speech = True
            start = i * hop_len

        elif in_speech and silence_run >= min_sil_frames:
            end = (i - silence_run + 1) * hop_len + frame_len
            segments.append((max(0, start - pad), min(len(x), end + pad)))
            in_speech = False

    if in_speech:
        segments.append((max(0, start - pad), len(x)))

    return segments


def trim_edges(seg, sr):
    max_edge = int(sr * MAX_EDGE_TRIM_MS / 1000)
    frame_len = int(sr * 0.02)
    hop_len = int(sr * 0.01)

    def find_voice(arr):
        rms = frame_rms_db(arr, frame_len, hop_len)
        idx = np.where(rms >= EDGE_TRIM_DB)[0]
        return idx[0] * hop_len if len(idx) else len(arr)

    left = find_voice(seg[:max_edge])
    right = len(seg) - find_voice(seg[::-1][:max_edge])
    return seg[left:right] if right > left else seg


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    count = 0
    for aud in INPUT_WAV:
        x, sr = sf.read(Path("audio") / aud)
        x = to_mono(np.asarray(x, dtype=np.float32))
        x = resample_audio(x, sr, TARGET_SR)
        x = peak_normalize(x, TARGET_PEAK)

        segments = split_by_silence(x, TARGET_SR)

        
        for s, e in segments:
            dur = (e - s) / TARGET_SR
            if dur < MIN_CLIP_LEN_S or dur > MAX_CLIP_LEN_S:
                continue

            seg = trim_edges(x[s:e], TARGET_SR)
            dur2 = len(seg) / TARGET_SR
            if dur2 < MIN_CLIP_LEN_S or dur2 > MAX_CLIP_LEN_S:
                continue

            count += 1
            out_path = out_dir / f"{FILENAME_PREFIX}_{count:03d}.wav"
            sf.write(out_path, seg, TARGET_SR, subtype="PCM_16")

        print(f"Saved {count} clips to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
