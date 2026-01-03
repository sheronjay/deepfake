from __future__ import annotations

import argparse
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp


# -----------------------------
# FaceMesh lip landmark indices
# -----------------------------
# These contours are commonly used and give a stable mouth mask.
# We use an OUTER polygon for lips, and an INNER polygon to exclude mouth opening.
LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61
]
LIPS_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191, 78
]


def _landmarks_to_points(landmarks, w: int, h: int, idxs: list[int]) -> np.ndarray:
    """Convert normalized MediaPipe landmarks -> pixel points (int32)."""
    pts = []
    for i in idxs:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)


def _soft_polygon_mask(h: int, w: int, outer_pts: np.ndarray, inner_pts: np.ndarray | None,
                       feather: int = 21) -> np.ndarray:
    """
    Create a soft mask (0..255 uint8):
      - fill outer polygon white
      - optionally cut out inner polygon black
      - gaussian allow smooth blending
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [outer_pts], 255)
    if inner_pts is not None:
        cv2.fillPoly(mask, [inner_pts], 0)

    k = max(3, feather | 1)  # force odd kernel size
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def _ema_smooth(prev: np.ndarray | None, cur: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """Exponential moving average for masks to reduce flicker."""
    if prev is None:
        return cur
    out = (alpha * prev.astype(np.float32) + (1.0 - alpha) * cur.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def _enhance_lip_ring(frame_bgr: np.ndarray, lip_ring_mask: np.ndarray,
                      unsharp_amount: float = 1.35, unsharp_blur_sigma: float = 1.2,
                      clahe_clip: float = 2.0) -> np.ndarray:
    """
    Enhance only the lip ring region:
      1) bilateral smooth to reduce blockiness but keep edges
      2) unsharp mask to sharpen lip edges
      3) mild CLAHE on L channel for local contrast
      4) blend with feathered lip_ring_mask
    """
    f = frame_bgr.astype(np.float32)

    # Edge-preserving smoothing (helps Wav2Lip blockiness around mouth)
    smooth = cv2.bilateralFilter(frame_bgr, d=7, sigmaColor=60, sigmaSpace=60).astype(np.float32)

    # Unsharp mask
    blur = cv2.GaussianBlur(smooth, (0, 0), unsharp_blur_sigma)
    sharp = cv2.addWeighted(smooth, unsharp_amount, blur, -(unsharp_amount - 1.0), 0)

    # CLAHE on L channel (local contrast)
    sharp_u8 = np.clip(sharp, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(sharp_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR).astype(np.float32)

    # Blend only inside lip mask
    m = (lip_ring_mask.astype(np.float32) / 255.0)[..., None]
    out = f * (1.0 - m) + enhanced * m
    return np.clip(out, 0, 255).astype(np.uint8)


def _fix_inner_mouth_magenta(frame_bgr: np.ndarray, inner_mouth_mask: np.ndarray,
                            strength: float = 0.55) -> np.ndarray:
    """
    Optional: suppress purple/magenta tint inside mouth (common Wav2Lip artifact).
    We reduce saturation for magenta-like hues inside inner mouth mask.
    """
    f = frame_bgr.astype(np.float32)
    m = (inner_mouth_mask.astype(np.float32) / 255.0)[..., None]

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # OpenCV Hue range is 0..179; magenta/purple often sits around ~135..175 depending on lighting
    magenta = ((h > 135) & (h < 175) & (s > 40)).astype(np.float32)

    s2 = s * (1.0 - strength * magenta)
    hsv2 = cv2.merge([h, np.clip(s2, 0, 255), v]).astype(np.uint8)
    corrected = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR).astype(np.float32)

    out = f * (1.0 - m) + corrected * m
    return np.clip(out, 0, 255).astype(np.uint8)


def _get_video_props(cap: cv2.VideoCapture) -> tuple[int, int, float]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or fps <= 0:
        fps = 25.0
    return w, h, fps


def _mux_audio_ffmpeg(video_no_audio: Path, audio_source: Path, out_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg or skip --audio.")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_no_audio),
        "-i", str(audio_source),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def enhance_lips_in_video(
    input_video: str | Path,
    output_video: str | Path,
    audio_source: str | Path | None = None,
    *,
    feather: int = 21,
    mask_smooth_alpha: float = 0.85,
    fix_inner_mouth_purple: bool = True,
    magenta_strength: float = 0.55,
    unsharp_amount: float = 1.35,
    unsharp_blur_sigma: float = 1.2,
    clahe_clip: float = 2.0,
) -> Path:
    """
    Main function you asked for.

    Args:
      input_video: path to video (e.g., wav2lip output).
      output_video: output path (mp4). If audio_source is provided, output_video is final muxed file.
      audio_source: optional path to audio (wav/m4a/mp4). If provided, ffmpeg will mux audio.

    Returns:
      Path to the final output video.
    """
    input_video = Path(input_video)
    output_video = Path(output_video)

    tmp_no_audio = output_video
    if audio_source is not None:
        # write a temporary no-audio file first
        tmp_no_audio = output_video.with_name(output_video.stem + "_noaudio.mp4")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    w, h, fps = _get_video_props(cap)

    # Use a common mp4-compatible codec. If you prefer H.264, use ffmpeg for encoding.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_no_audio), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create VideoWriter (mp4v). Try a different output path/codec.")

    mp_face_mesh = mp.solutions.face_mesh

    prev_lip_ring_mask = None
    prev_inner_mask = None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,      # video tracking mode (better + faster for videos)
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        pbar = tqdm(total=total, desc="Enhancing lips", unit="frame")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark

                outer = _landmarks_to_points(lms, w, h, LIPS_OUTER)
                inner = _landmarks_to_points(lms, w, h, LIPS_INNER)

                # Lip ring mask: outer minus inner
                lip_ring = _soft_polygon_mask(h, w, outer, inner, feather=feather)
                lip_ring = _ema_smooth(prev_lip_ring_mask, lip_ring, alpha=mask_smooth_alpha)
                prev_lip_ring_mask = lip_ring

                out = _enhance_lip_ring(
                    frame, lip_ring,
                    unsharp_amount=unsharp_amount,
                    unsharp_blur_sigma=unsharp_blur_sigma,
                    clahe_clip=clahe_clip
                )

                # Optional: inner mouth color correction (purple/magenta suppression)
                if fix_inner_mouth_purple:
                    inner_only = _soft_polygon_mask(h, w, inner, None, feather=max(11, feather - 6))
                    inner_only = _ema_smooth(prev_inner_mask, inner_only, alpha=mask_smooth_alpha)
                    prev_inner_mask = inner_only
                    out = _fix_inner_mouth_magenta(out, inner_only, strength=magenta_strength)

                frame = out

            writer.write(frame)
            pbar.update(1)

        pbar.close()

    cap.release()
    writer.release()

    # If user wants audio, mux it back in
    if audio_source is not None:
        audio_source = Path(audio_source)
        _mux_audio_ffmpeg(tmp_no_audio, audio_source, output_video)
        # you can keep or delete tmp_no_audio; keeping is useful for debugging
        return output_video

    return tmp_no_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input video (e.g., wav2lip output mp4)")
    ap.add_argument("output", help="Output video (mp4)")
    ap.add_argument("--audio", default=None, help="Optional audio source (wav/m4a/mp4) to mux into output")
    ap.add_argument("--no-purple-fix", action="store_true", help="Disable inner-mouth purple suppression")
    args = ap.parse_args()

    enhance_lips_in_video(
        args.input,
        args.output,
        audio_source=args.audio,
        fix_inner_mouth_purple=not args.no_purple_fix
    )
    print("Done:", args.output)


if __name__ == "__main__":
    main()
