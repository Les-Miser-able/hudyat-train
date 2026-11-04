"""Offline media decoding, detection, resumable caching, and sample generation."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import io
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .config import Settings
from .dataset import Source, discover
from .features import FEATURE_VERSION, location_features, pack_hands, prepare
from .jobs import Job
from .storage import fingerprint, sha256, write_bytes, write_json


def detector_metadata(settings: Settings, face: bool) -> dict:
    hand = settings.path("hand_model")
    if not hand.is_file():
        raise ValueError(f"Missing hand detector: {hand}. See models/README.md.")
    metadata = {
        "backend": "MediaPipe HandLandmarker", "mediapipe_version": importlib.metadata.version("mediapipe"),
        "model_sha256": sha256(hand), "model_path": str(hand), "num_hands": 2,
        "detection_confidence": settings.get("detection_confidence"), "presence_confidence": settings.get("detection_confidence"),
        "tracking_confidence": 0.5, "swap_hands": bool(settings.get("swap_hands")),
        "image_orientation": "EXIF transpose for images; unmirrored pixels for all media",
    }
    if face:
        path = settings.path("face_model")
        if not path.is_file():
            raise ValueError(f"The Family category requires the face detector: {path}")
        metadata["face"] = {"backend": "MediaPipe FaceLandmarker", "model_path": str(path),
                            "model_sha256": sha256(path), "num_faces": 1,
                            "nose": 1, "forehead": 10, "chin": 152, "hand_points": [0, 4, 8]}
    return metadata


def detect_media(source: Source, settings: Settings, job: Job) -> dict:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    is_image, with_face = source.kind == "image", source.category == "family"
    mode = vision.RunningMode.IMAGE if is_image else vision.RunningMode.VIDEO
    raw, locations, collisions = [], [], 0
    with ExitStack() as stack:
        hands = stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(settings.path("hand_model"))), running_mode=mode,
            num_hands=2, min_hand_detection_confidence=settings.get("detection_confidence"),
            min_hand_presence_confidence=settings.get("detection_confidence"), min_tracking_confidence=0.5)))
        face = stack.enter_context(vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(settings.path("face_model"))), running_mode=mode,
            num_faces=1))) if with_face else None

        def frame(rgb, timestamp):
            nonlocal collisions
            job.check()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = hands.detect(image) if is_image else hands.detect_for_video(image, timestamp)
            packed, duplicate = pack_hands(result, bool(settings.get("swap_hands")))
            faces = face.detect(image) if face and is_image else face.detect_for_video(image, timestamp) if face else None
            points = faces.face_landmarks[0] if faces and faces.face_landmarks else None
            raw.append(packed)
            locations.append(location_features(packed, points, rgb.shape[1], rgb.shape[0]))
            collisions += duplicate

        if is_image:
            from PIL import Image, ImageOps
            with Image.open(source.path) as picture:
                rgb = np.asarray(ImageOps.exif_transpose(picture).convert("RGB"))
            height, width = rgb.shape[:2]
            fps = 30.0  # only used by the one-frame window; the image is repeated to 32 frames
            frame(rgb, 0)
        else:
            cap = cv2.VideoCapture(str(source.path))
            try:
                if not cap.isOpened():
                    raise ValueError("Video could not be opened.")
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError("Video has no valid frame rate; re-encode it before extraction.")
                previous, width, height = -1, 0, 0
                while True:
                    job.check()
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    h, w = bgr.shape[:2]
                    if width and (width, height) != (w, h):
                        raise ValueError("Video frame dimensions changed within the clip.")
                    width, height = w, h
                    timestamp = max(previous + 1, int(len(raw) * 1000 / fps + 0.5))
                    frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), timestamp)
                    previous = timestamp
                if not raw or (reported > len(raw) + 2):
                    raise ValueError(f"Incomplete video decode: {len(raw)} of {reported} reported frames.")
            finally:
                cap.release()
    return {"raw": np.asarray(raw, dtype=np.float32), "locations": np.asarray(locations, dtype=np.float32),
            "width": width, "height": height, "fps": fps, "collisions": collisions}


def source_signature(sources: list[Source]) -> str:
    return fingerprint([s.record() for s in sources])


def extract(settings: Settings, category: str, variant: str, job: Job | None = None) -> dict:
    job = job or Job()
    catalog = Catalog(settings.path("signs_file"))
    category = catalog.category(category)
    sources, issues, excluded = discover(settings, category, variant)
    if issues:
        raise ValueError("Fix the dataset scan errors first: " + "; ".join(f"{i['source']}: {i['reason']}" for i in issues[:5]))
    if not sources:
        raise ValueError(f"No usable raw files for {category}/{variant}. Add media under dataset/raw/<category>/<class>/<signer>/.")
    detectors = detector_metadata(settings, category == "family")
    extraction_key = fingerprint({"detectors": detectors, "features": FEATURE_VERSION})
    sample_key = fingerprint({"extraction": extraction_key, "variant": variant, "category": category})[:12]
    dest = settings.path("extracted_dir") / variant / category
    items, normalization = [], None
    for i, source in enumerate(sources, 1):
        job.check()
        job.emit(f"[{i}/{len(sources)}] {source.relative}")
        item = {**source.record(), "status": "rejected"}
        cache = settings.path("cache_dir") / extraction_key[:16] / f"{source.digest}.npz"
        try:
            data = None
            if cache.exists():
                try:
                    with np.load(cache, allow_pickle=False) as saved:
                        data = {name: saved[name] for name in saved.files}
                    job.emit("  Cached detections")
                except (OSError, ValueError, EOFError):
                    job.emit("  Rebuilding unreadable detection cache")
            if data is None:
                data = detect_media(source, settings, job)
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **data)
                write_bytes(cache, buffer.getvalue())
            sample, normalization, quality = prepare(data["raw"], data["locations"], int(data["width"]), int(data["height"]),
                                                     float(data["fps"]), category, variant, source.hands)
            quality["duplicate_hand_labels"] = int(data["collisions"])
            quality["width"], quality["height"] = int(data["width"]), int(data["height"])
            item["quality"] = quality
            if source.kind == "video" and quality["window_seconds"] > 3.0 + 1 / quality["fps"]:
                raise ValueError("The sign window exceeds the deployed app's 3-second limit. Supply one sign per clip.")
            if quality["hand_rate"] < settings.get("min_hand_rate"):
                raise ValueError(f"Hand detection {quality['hand_rate']:.1%} is below {settings.get('min_hand_rate'):.0%} in the sign window.")
            if source.hands == 2 and quality["both_hands_rate"] < settings.get("min_both_hands_rate"):
                raise ValueError(f"Both-hands detection {quality['both_hands_rate']:.1%} is below the required rate.")
            if category == "family" and not np.any(sample[:, 140] > 0.5):
                raise ValueError("No usable face location in the model input; Family needs the face in view.")
            relative = f"{source.sign_id}/{source.digest[:20]}-{sample_key}.npy"
            buffer = io.BytesIO()
            np.save(buffer, sample, allow_pickle=False)
            write_bytes(dest / relative, buffer.getvalue())
            item.update(status="accepted", sample=relative, sample_sha256=sha256(dest / relative), shape=list(sample.shape))
            job.emit(f"  Accepted: {sample.shape}, hands {quality['hand_rate']:.1%}")
        except (ValueError, OSError, RuntimeError) as error:
            item["reason"] = str(error)
            job.emit(f"  Rejected: {error}")
        items.append(item)
    job.check()
    manifest = {
        "schema": 1, "created_at": datetime.now(timezone.utc).isoformat(), "category": category, "variant": variant,
        "feature_version": FEATURE_VERSION, "catalog_sha256": sha256(settings.path("signs_file")),
        "source_signature": source_signature(sources), "detectors": detectors,
        "settings": settings.values, "normalization": normalization, "input_shape": [32, 141 if category == "family" else 128],
        "expected_signs": [s.sign_id for s in catalog.selected(category, variant)], "items": items, "excluded": excluded,
    }
    write_json(dest / "index.json", manifest)
    accepted = sum(it["status"] == "accepted" for it in items)
    job.emit(f"Extraction complete: {accepted} accepted, {len(items) - accepted} rejected. Index: {dest / 'index.json'}")
    return manifest
