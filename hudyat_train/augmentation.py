from __future__ import annotations

import numpy as np

from .features import mirror_frames, resample_sign


def augment(sample, rng: np.random.Generator, mirror_ok: bool = False, static: bool = False):
    out = np.asarray(sample, dtype=np.float32).copy()
    angle = np.deg2rad(rng.uniform(-15, 15))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = rng.uniform(0.9, 1.1)
    for slot in range(2):
        present, base = out[:, 126 + slot] > 0.5, slot * 63
        points = out[present, base:base + 63].reshape(-1, 21, 3)
        points[:, :, :2] = points[:, :, :2] @ rotation.T
        points *= scale
        noise_shape = (1, 21, 3) if static else points.shape
        if len(points):
            points += rng.normal(0, 0.01, noise_shape)
        out[present, base:base + 63] = points.reshape(-1, 63)
    if not static:
        # Changes in timing and occasional missing camera frames are applied to train only.
        kept = np.unique(np.r_[0, np.flatnonzero(rng.random(len(out)) > 0.08), len(out) - 1])
        compact = out[kept]
        speed = rng.uniform(0.8, 1.2)
        positions = np.linspace(0, 1, len(compact)) ** speed
        uniform = np.linspace(0, 1, len(out))
        warped = np.empty_like(out)
        for j, t in enumerate(uniform):
            hi = min(max(1, int(np.searchsorted(positions, t))), len(compact) - 1)
            lo = hi - 1
            w = (t - positions[lo]) / max(positions[hi] - positions[lo], 1e-9)
            nearest = lo if w < 0.5 else hi
            warped[j] = compact[nearest]
            for slot in range(2):
                base, flag = slot * 63, 126 + slot
                if compact[lo, flag] and compact[hi, flag]:
                    warped[j, base:base + 63] = (1 - w) * compact[lo, base:base + 63] + w * compact[hi, base:base + 63]
                if out.shape[1] == 141 and compact[lo, flag] and compact[hi, flag] and compact[lo, 140] and compact[hi, 140]:
                    start = 128 + slot * 6
                    warped[j, start:start + 6] = (1 - w) * compact[lo, start:start + 6] + w * compact[hi, start:start + 6]
        out = warped
    if mirror_ok and rng.random() < 0.5:
        out = mirror_frames(out)
    return out.astype(np.float32)
