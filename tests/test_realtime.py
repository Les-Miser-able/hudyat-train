from types import SimpleNamespace
import threading

import numpy as np
import pytest

from hudyat_train.features import prepare
from hudyat_train.realtime import Observation, SignCapture, prepare_observations


def observation(at, present=True):
    raw = np.zeros(128, np.float32)
    raw[127] = int(present)
    return Observation(at, raw, np.zeros(13, np.float32), 1280, 720)


def test_automatic_capture_ends_on_hands_down_and_keeps_prebuffer():
    capture = SignCapture()
    completed = []
    for i in range(50):
        output = capture.push(observation(i / 30, 5 <= i <= 33))
        if output:
            completed.append(output)
    assert len(completed) == 1
    items, reason = completed[0]
    assert reason == "hands_down"
    assert items[0].at < 5 / 30
    assert any(not o.present for o in items)


def test_automatic_capture_ignores_single_frame_blips():
    capture = SignCapture()
    output = [capture.push(observation(i / 30, i == 5)) for i in range(30)]
    assert not any(output)


def test_long_hold_emits_once_until_hands_rest():
    capture = SignCapture()
    completed = [capture.push(observation(i / 30)) for i in range(200)]
    assert sum(item is not None for item in completed) == 1
    for i in range(200, 210):
        assert capture.push(observation(i / 30, False)) is None
    next_sign = [capture.push(observation(i / 30, i < 240)) for i in range(210, 250)]
    assert sum(item is not None for item in next_sign) == 1


def test_manual_mode_needs_explicit_capture_and_finish():
    capture = SignCapture("manual")
    for i in range(15):
        assert capture.push(observation(i / 30)) is None
    assert capture.active is None
    capture.start()
    for i in range(15, 30):
        assert capture.push(observation(i / 30)) is None
    items, reason = capture.finish()
    assert len(items) == 15 and reason == "manual_finish"
    assert capture.active is None


def test_manual_capture_has_three_second_limit_and_reset_discards_pending():
    capture = SignCapture("manual")
    capture.start()
    output = None
    for i in range(100):
        output = capture.push(observation(i / 30))
        if output:
            break
    assert output and output[1] == "max_length"
    capture.start()
    capture.push(observation(5))
    capture.reset()
    assert capture.finish() is None


@pytest.mark.parametrize("category,variant,normalization,canonical", [
    ("alphabet", "sequence", "hand-relative-sequence-v1", False),
    ("alphabet", "static", "hand-relative-v1", False),
    ("family", "sequence", "hand-face-location-v1", True),
    ("family", "static", "hand-face-location-v1", True),
])
def test_live_input_matches_offline_training_preprocessing(detections, category, variant, normalization, canonical):
    info = SimpleNamespace(variant=variant, features=141 if category == "family" else 128,
                           preprocessing={"normalization": normalization, "handedness": {"canonical": canonical}})
    observations = [Observation(i / 30, raw, loc, 1280, 720) for i, (raw, loc) in enumerate(zip(detections["raw"], detections["locations"]))]
    live, stats = prepare_observations(info, observations)
    offline, _, quality = prepare(detections["raw"], detections["locations"], 1280, 720, 30, category, variant)
    assert np.allclose(live, offline)
    assert stats["hand_rate"] == quality["hand_rate"]


def test_no_hand_capture_is_invalid_without_running_the_model():
    info = SimpleNamespace(variant="static", features=128, preprocessing={"normalization": "hand-relative-v1"})
    with pytest.raises(ValueError, match="No hand detected"):
        prepare_observations(info, [observation(0, False)])


def test_live_worker_can_preview_a_30_fps_source_without_throttling(monkeypatch, tmp_path):
    import cv2
    import hudyat_train.realtime as live

    clock = SimpleNamespace(at=0.0)

    class Video:
        def isOpened(self):
            return True

        def get(self, key):
            return 30.0

        def read(self):
            clock.at += 1 / 30
            return (True, np.zeros((48, 64, 3), np.uint8)) if clock.at < 3 else (False, None)

        def release(self):
            pass

    class Detector:
        def __init__(self, *args):
            pass

        def detect(self, bgr, at):
            return Observation(at, np.zeros(128, np.float32), np.zeros(13, np.float32), 64, 48)

        def close(self):
            pass

    info = SimpleNamespace(classes=["A", "B"], category="alphabet", variant="sequence")
    session = SimpleNamespace(directory=tmp_path, threshold=0.85, rows=[], close=lambda *args: {})
    monkeypatch.setattr(live, "perf_counter", lambda: clock.at)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: Video())
    monkeypatch.setattr(live, "describe_model", lambda *args: info)
    monkeypatch.setattr(live, "Predictor", lambda *args: object())
    monkeypatch.setattr(live, "detector_paths", lambda *args: {})
    monkeypatch.setattr(live, "TestSession", lambda *args: session)
    monkeypatch.setattr(live, "FrameDetector", Detector)
    monkeypatch.setattr(live, "_preview", lambda *args: "preview")
    events = []
    live.run_live(None, tmp_path, video=tmp_path / "clip.mp4", emit=events.append)
    frames = [event for event in events if event["type"] == "frame"]
    assert len(frames) >= 75, f"Only {len(frames)} previews for a 3-second, 30 FPS source"


@pytest.mark.parametrize("busy_stage", ["detection", "prediction"])
def test_camera_preview_continues_while_model_work_is_busy(monkeypatch, tmp_path, busy_stage):
    import cv2
    import hudyat_train.realtime as live
    from hudyat_train.jobs import Cancelled

    detecting, preview_during_detection = threading.Event(), threading.Event()
    clock = SimpleNamespace(at=0.0)

    class Camera:
        reads = 0
        released = False

        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def read(self):
            self.reads += 1
            if self.reads == 2:
                assert detecting.wait(2), "The first camera frame never reached detection"
            if self.reads > 2:
                return False, None
            clock.at += 1 / 30
            return True, np.zeros((48, 64, 3), np.uint8)

        def release(self):
            self.released = True

    class Detector:
        def __init__(self, *args):
            pass

        def detect(self, bgr, at):
            if busy_stage == "detection":
                detecting.set()
                assert preview_during_detection.wait(2), "Camera preview froze during landmark detection"
            raw = np.zeros(128, np.float32)
            raw[127] = 1
            return Observation(at, raw, np.zeros(13, np.float32), 64, 48)

        def close(self):
            pass

    class Predictor:
        def predict(self, sample):
            if busy_stage == "prediction":
                detecting.set()
                assert preview_during_detection.wait(2), "Camera preview froze during model prediction"
            return {"predicted": "A", "confidence": 0.9, "top3": []}

    def check():
        if preview_during_detection.is_set():
            raise Cancelled()

    def emit(event):
        if event["type"] == "frame" and detecting.is_set():
            preview_during_detection.set()

    camera = Camera()
    info = SimpleNamespace(classes=["A", "B"], category="alphabet", variant="static")
    session = SimpleNamespace(directory=tmp_path, threshold=0.85, rows=[], close=lambda *args: {})
    monkeypatch.setattr(live, "perf_counter", lambda: clock.at)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    monkeypatch.setattr(live, "describe_model", lambda *args: info)
    monkeypatch.setattr(live, "Predictor", lambda *args: Predictor())
    monkeypatch.setattr(live, "detector_paths", lambda *args: {})
    monkeypatch.setattr(live, "TestSession", lambda *args: session)
    monkeypatch.setattr(live, "FrameDetector", Detector)
    monkeypatch.setattr(live, "prepare_observations", lambda *args: (np.zeros((32, 128), np.float32), {}))
    monkeypatch.setattr(live, "_preview", lambda *args: "preview")
    live.run_live(None, tmp_path, emit=emit, job=SimpleNamespace(check=check))
    assert preview_during_detection.is_set()
    assert camera.released


def test_camera_stream_keeps_the_latest_capture_and_propagates_disconnect(monkeypatch):
    import hudyat_train.realtime as live

    class Camera:
        count = 0
        released = False

        def read(self):
            self.count += 1
            if self.count > 100:
                return False, None
            return True, np.full((2, 2, 3), self.count, np.uint8)

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(live, "perf_counter", lambda: camera.count / 30)
    stream = live.CameraStream(camera, 0, lambda *args: None)
    stream.start()
    stream.thread.join(timeout=2)
    try:
        at, frame = stream.read()
        assert at == pytest.approx(100 / 30)
        assert np.all(frame == 100)
        with pytest.raises(ValueError, match="stopped delivering frames"):
            stream.read()
        assert camera.released
    finally:
        stream.close()


def test_camera_preview_without_landmarks_is_a_png_and_does_not_change_model_pixels():
    import base64
    import cv2
    from hudyat_train.realtime import _preview

    image = np.random.default_rng(42).integers(0, 256, (360, 640, 3), np.uint8)
    original = image.copy()
    encoded = _preview(image)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (236, 420, 3)
    assert np.array_equal(image, original)
