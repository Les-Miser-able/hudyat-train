"""Numerical contract of HUDYAT/client's current recognition pipeline.

Hands are contiguous 63-value blocks. Presence flags are at 126 and 127,
not between the hands. No full face landmarks are stored.
"""
from __future__ import annotations

import math
import numpy as np

FEATURE_VERSION = "hudyat-browser-v1"
FRAMES = 32


def pack_hands(result, swap: bool = False) -> tuple[np.ndarray, int]:
    out, scores, collisions = np.zeros(128, np.float32), [-1.0, -1.0], 0
    for points, categories in zip(result.hand_landmarks, result.handedness):
        if len(points) != 21 or not categories:
            continue
        label, score = categories[0].category_name.lower(), float(categories[0].score)
        if label not in ("left", "right") or not math.isfinite(score):
            continue
        slot = int(label == "right") ^ int(swap)
        xyz = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32)
        if not np.isfinite(xyz).all():
            continue
        if scores[slot] >= 0:
            collisions += 1
        if score <= scores[slot]:
            continue
        out[slot * 63:(slot + 1) * 63] = xyz.ravel()
        out[126 + slot], scores[slot] = 1, score
    return out, collisions


def sign_window(present, fps: float) -> tuple[int, int]:
    present = np.asarray(present, dtype=bool)
    if not len(present) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("A sign window needs frames and a valid frame rate.")
    found = np.flatnonzero(present)
    if not len(found):
        return 0, len(present) - 1
    margin = math.floor(0.1 * fps + 0.5)
    start, end = max(0, int(found[0]) - margin), min(len(present) - 1, int(found[-1]) + margin)
    need = min(8, len(present))
    if end - start + 1 < need:
        start = math.floor((start + end) / 2 - (need - 1) / 2 + 0.5)
        start = max(0, min(start, len(present) - need))
        end = start + need - 1
    return start, end


def normalize_frame(frame, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions.")
    frame, out = np.asarray(frame, dtype=np.float64), np.zeros(128, np.float32)
    for slot in range(2):
        if not frame[126 + slot]:
            continue
        base = slot * 63
        points = frame[base:base + 63].reshape(21, 3) * [width, height, width]
        points -= points[0].copy()
        palm = np.linalg.norm(points[9, :2])  # the browser uses the 2D pixel palm length
        if not np.isfinite(palm) or palm < 1e-6:
            continue
        out[base:base + 63] = (points / palm).ravel()
        out[126 + slot] = 1
    return out


def normalize_sequence(frames, width: int, height: int) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    out = np.asarray([normalize_frame(f, width, height) for f in frames])
    for slot in range(2):
        base = slot * 63
        valid = np.flatnonzero(out[:, 126 + slot])
        if not len(valid):
            continue
        palms = np.hypot((frames[valid, base + 27] - frames[valid, base]) * width,
                         (frames[valid, base + 28] - frames[valid, base + 1]) * height)
        scale = float(np.median(palms))
        wrists = frames[valid, base:base + 3] * [width, height, width]
        out[valid, base:base + 3] = (wrists - wrists[0]) / scale
    return out


def location_features(raw, face, width: int, height: int) -> np.ndarray:
    out = np.zeros(13, np.float32)
    if face is None or len(face) <= 152:
        return out
    points = np.asarray([[p.x, p.y] for p in face], dtype=np.float64) * [width / height, 1]
    size = np.linalg.norm(points[10] - points[152])
    if not np.isfinite(points[[1, 10, 152]]).all() or not np.isfinite(size) or size < 1e-6:
        return out
    for slot in range(2):
        if not raw[126 + slot]:
            continue
        hand = np.asarray(raw[slot * 63:(slot + 1) * 63], dtype=np.float64).reshape(21, 3)
        out[slot * 6:slot * 6 + 6] = ((hand[[0, 4, 8], :2] * [width / height, 1] - points[1]) / size).ravel()
    out[12] = 1
    return out


def resample_legacy(frames, n: int = FRAMES) -> np.ndarray:
    """HUDYAT category-core.js resample, including its gap interpolation behavior."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 128 or not len(frames) or n < 1:
        raise ValueError("Resampling needs nonempty 128-feature frames.")
    out = np.zeros((n, 128), np.float32)
    for j in range(n):
        position = j * (len(frames) - 1) / (n - 1) if n > 1 else 0
        lo, hi = math.floor(position), min(math.floor(position) + 1, len(frames) - 1)
        ratio, nearest = position - lo, math.floor(position + 0.5)
        out[j, :126] = frames[lo, :126].astype(float) * (1 - ratio) + frames[hi, :126].astype(float) * ratio
        for slot in range(2):
            out[j, 126 + slot] = frames[nearest, 126 + slot]
            if not out[j, 126 + slot]:
                out[j, slot * 63:(slot + 1) * 63] = 0
    return out


def resample_sign(frames, n: int = FRAMES, extra=None) -> np.ndarray:
    """HUDYAT sign-capture.js resampleSign: interpolate only present pairs."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or not len(frames) or frames.shape[1] not in (128, 136, 141) or n < 1:
        raise ValueError("Invalid sign frames.")
    dim = frames.shape[1]
    slots = [(list(range(h * 63, (h + 1) * 63)) + (extra[h] if extra else []), [126 + h], 126 + h)
             for h in range(2)]
    if dim == 141:
        slots += [(list(range(128, 134)), [126, 140], None),
                  (list(range(134, 140)), [127, 140], None), ([], [], 140)]
    out = np.zeros((n, dim), np.float32)
    for j in range(n):
        t = j * (len(frames) - 1) / (n - 1) if n > 1 else 0
        lo, hi = math.floor(t), min(math.floor(t) + 1, len(frames) - 1)
        weight = t - lo
        nearest = lo if weight < 0.5 else hi
        for indices, flags, flag in slots:
            both = all(frames[lo, f] > 0.5 and frames[hi, f] > 0.5 for f in flags)
            out[j, indices] = ((1 - weight) * frames[lo, indices].astype(float) + weight * frames[hi, indices].astype(float)
                               if both else frames[nearest, indices])
            if flag is not None:
                out[j, flag] = frames[nearest, flag]
    return out


def mirror_frames(frames) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    mirrored = frames.copy()
    mirrored[:, 0:126:3] *= -1
    out = np.concatenate([mirrored[:, 63:126], mirrored[:, :63], mirrored[:, [127, 126]]], axis=1)
    if frames.shape[1] == 141:
        location = frames[:, 128:140].copy()
        location[:, 0::2] *= -1
        out = np.concatenate([out, location[:, 6:12], location[:, :6], frames[:, 140:141]], axis=1)
    return out


def auto_hand(raws, width: int, height: int) -> str:
    """The current browser's canonical-hand selection (presence, then wrist travel)."""
    raws = np.asarray(raws, dtype=np.float32)
    rich = np.zeros((len(raws), 136), np.float32)
    rich[:, :128] = raws
    for h in range(2):
        base, dest = h * 63, 128 + h * 4
        valid = raws[:, 126 + h] > 0.5
        rich[valid, dest:dest + 3] = raws[valid, base:base + 3] * [width / height, 1, width / height]
        delta = (raws[valid, base + 27:base + 30] - raws[valid, base:base + 3]) * [width / height, 1, width / height]
        rich[valid, dest + 3] = np.linalg.norm(delta, axis=1)
    win = resample_sign(rich, extra=[list(range(128, 132)), list(range(132, 136))])
    left, right = win[:, 126] > 0.5, win[:, 127] > 0.5
    any_count = int((left | right).sum())
    only_l, only_r = int((left & ~right).sum()), int((right & ~left).sum())
    if not any_count:
        return "right"
    if (only_l + only_r) * 2 > any_count:
        return "left" if only_l > only_r else "right"

    def distance(present, base):
        points = win[present, base:base + 4].astype(float)
        if len(points) < 2:
            return 0
        scale = float(np.median(points[:, 3]))
        return float(np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1).sum()) / (scale if scale > 1e-6 else 1)

    return "left" if distance(left, 128) > distance(right, 132) else "right"


def static_pose(frames, hands: int = 1, raw=None, aspect: float = 1.0) -> int:
    """One stable pose per source, so validation/test do not multiply a clip's votes."""
    frames = np.asarray(frames)
    valid = frames[:, 126:128].sum(axis=1) >= hands
    if frames.shape[1] == 141:
        valid &= frames[:, 140] > 0.5
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        raise ValueError("No usable held pose detected.")
    centre = (len(frames) - 1) / 2

    def cost(i):
        adjacent = [j for j in (i - 1, i + 1) if 0 <= j < len(frames) and np.array_equal(frames[j, 126:128], frames[i, 126:128])]
        coordinates = list(range(126)) + (list(range(128, 140)) if frames.shape[1] == 141 else [])
        change = float(np.mean([np.mean((frames[i, coordinates] - frames[j, coordinates]) ** 2) for j in adjacent])) if adjacent else 1e6
        if raw is not None:
            movements = []
            for slot in range(2):
                base = slot * 63
                if not frames[i, 126 + slot]:
                    continue
                palm = np.linalg.norm((raw[i, base + 27:base + 29] - raw[i, base:base + 2]) * [aspect, 1])
                if palm > 1e-6:
                    movements.extend(float(np.linalg.norm((raw[i, base:base + 3] - raw[j, base:base + 3]) * [aspect, 1, aspect]) / palm) ** 2 for j in adjacent)
            if movements:
                change += float(np.mean(movements))
        return change + abs(i - centre) * 1e-7

    return int(min(candidates, key=cost))


def prepare(raws, locations, width: int, height: int, fps: float, category: str, variant: str, hands: int = 1):
    raws, locations = np.asarray(raws), np.asarray(locations)
    present = np.any(raws[:, 126:128] > 0.5, axis=1)
    start, end = sign_window(present, fps)
    raw, location = raws[start:end + 1], locations[start:end + 1]
    stats = {"frames": len(raws), "start": start, "end": end, "window_frames": len(raw),
             "hand_rate": float(present[start:end + 1].mean()),
             "both_hands_rate": float(np.all(raw[:, 126:128] > 0.5, axis=1).mean()),
             "face_rate": float(location[:, 12].mean()), "fps": fps, "window_seconds": len(raw) / fps}
    if variant == "sequence" and category != "family":
        normalized = normalize_sequence(raw, width, height)
        sample = resample_legacy(normalized)
        normalization, hand = "hand-relative-sequence-v1", "unchanged"
    else:
        normalized = np.asarray([normalize_frame(f, width, height) for f in raw])
        if category == "family":
            location = location.copy()
            for slot in range(2):
                location[normalized[:, 126 + slot] == 0, slot * 6:slot * 6 + 6] = 0
            normalized = np.concatenate([normalized, location], axis=1)
        if variant == "static":
            pose = static_pose(normalized, hands, raw, width / height)
            stats["pose_frame"] = start + pose
            sample = np.repeat(normalized[pose:pose + 1], FRAMES, axis=0)
        else:
            sample = resample_sign(normalized)
        normalization = "hand-face-location-v1" if category == "family" else "hand-relative-v1"
        hand = auto_hand(raw, width, height) if category != "alphabet" else "unchanged"
        if hand == "left":
            sample = mirror_frames(sample)
    if not np.isfinite(sample).all() or not np.any(sample[:, 126:128] > 0.5):
        raise ValueError("All hand detections were degenerate or non-finite.")
    stats["canonical_hand"] = hand
    return sample.astype(np.float32), normalization, stats
