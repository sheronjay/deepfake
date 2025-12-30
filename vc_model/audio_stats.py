import numpy as np
import soundfile as sf
from pathlib import Path

wav = Path("audio") /"why_do_bad_things_happen_to_good_people__buddhism_in_english.wav"

x, sr = sf.read(wav)
if x.ndim > 1:
    x = x.mean(axis=1)  # mono

# Basic stats
peak = np.max(np.abs(x))
rms = np.sqrt(np.mean(x**2))
rms_db = 20 * np.log10(rms + 1e-12)
peak_db = 20 * np.log10(peak + 1e-12)

# Silence estimation
silence_threshold_db = -40
frame_len = int(0.03 * sr)  # 30 ms
hop_len = int(0.01 * sr)    # 10 ms

silence_frames = 0
total_frames = 0

for i in range(0, len(x) - frame_len, hop_len):
    frame = x[i:i+frame_len]
    frms = np.sqrt(np.mean(frame**2))
    frms_db = 20 * np.log10(frms + 1e-12)
    if frms_db < silence_threshold_db:
        silence_frames += 1
    total_frames += 1

silence_ratio = silence_frames / max(total_frames, 1)

print(f"Sample rate: {sr} Hz")
print(f"Peak: {peak:.3f} ({peak_db:.1f} dBFS)")
print(f"RMS: {rms:.4f} ({rms_db:.1f} dBFS)")
print(f"Estimated silence ratio (< {silence_threshold_db} dB): {silence_ratio:.2%}")
