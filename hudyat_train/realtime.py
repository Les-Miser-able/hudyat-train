"""Live test worker and pure capture state, independent of desktop widgets."""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
import base64
import json
import math
from pathlib import Path
import queue
import sys
import threading
from time import perf_counter

from .inference import Predictor, describe_model, detector_paths
from .experiments import append_event
from .jobs import Cancelled, Job
from .testing import TestSession


@dataclass
class Observation:
    at: float
    raw: object
    location: object
    width: int
    height: int
    detector_ms: float = 0

    @property
    def present(self):
        return bool(self.raw[126] > 0.5 or self.raw[127] > 0.5)


class CameraStream:
    """Keep the camera moving and retain only the newest frame for detection."""
    def __init__(self, capture, origin, on_frame):
        self.capture, self.origin, self.on_frame = capture, origin, on_frame
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.latest, self.error, self.finished = None, None, False
        self.thread = threading.Thread(target=self._read, name="hudyat-camera", daemon=True)

    def start(self):
        self.thread.start()

    def _read(self):
        try:
            while not self.stopped.is_set():
                ok, bgr = self.capture.read()
                if self.stopped.is_set():
                    break
                if not ok:
                    raise ValueError("The camera stopped delivering frames.")
                at = perf_counter() - self.origin
                with self.condition:
                    self.latest = (at, bgr)
                    self.condition.notify()
                self.on_frame(bgr, at)
        except BaseException as error:
            self.error = error
        finally:
            self.capture.release()
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.latest is not None or self.finished, timeout=0.05)
            frame, self.latest = self.latest, None
            if frame is not None:
                return frame
            if self.error is not None:
                raise self.error
            return None

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)


class SignCapture:
    """0.1s prebuffer, 0.2s hands-down end, 3s maximum; manual capture is explicit."""
    def __init__(self, mode="automatic"):
        if mode not in ("automatic", "manual"):
            raise ValueError("Capture mode must be automatic or manual.")
        self.mode = mode
        self.reset()

    def reset(self):
        self.ring, self.active = deque(), None
        self.absent_since = None
        self.waiting_for_rest = False

    def start(self):
        if self.mode != "manual":
            raise ValueError("Manual capture is available in manual mode.")
        self.active = []

    def finish(self, reason="manual_finish"):
        active, self.active = self.active, None
        self.absent_since = None
        self.ring.clear()
        if self.mode == "automatic" and reason == "max_length":
            self.waiting_for_rest = True
        if active is None or not active:
            return None
        if self.mode == "automatic":
            hands = [o.at for o in active if o.present]
            if len(hands) < 2 or hands[-1] - hands[0] < 0.2 - 1e-9:
                return None
        return active, reason

    def push(self, observation):
        if self.active is None:
            if self.waiting_for_rest:
                self.absent_since = None if observation.present else self.absent_since if self.absent_since is not None else observation.at
                if self.absent_since is not None and observation.at - self.absent_since >= 0.2 - 1e-9:
                    self.waiting_for_rest, self.absent_since = False, None
            if self.mode == "automatic" and observation.present and not self.waiting_for_rest:
                self.active = [*self.ring, observation]
                self.ring.clear()
            else:
                self.ring.append(observation)
                while self.ring and observation.at - self.ring[0].at > 0.1 + 1e-9:
                    self.ring.popleft()
            return None
        self.active.append(observation)
        if self.mode == "automatic":
            self.absent_since = None if observation.present else self.absent_since if self.absent_since is not None else observation.at
            if self.absent_since is not None and observation.at - self.absent_since >= 0.2 - 1e-9:
                return self.finish("hands_down")
        if self.active[-1].at - self.active[0].at >= 3.0 - 1e-9:
            return self.finish("max_length")
        return None


class FrameDetector:
    def __init__(self, info, paths):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp, self.stack, self.last_timestamp = mp, ExitStack(), -1
        extraction = info.preprocessing["extraction"]
        self.swap = bool(extraction.get("swap_hands", False))
        try:
            self.hands = self.stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(paths["hand"])), running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=extraction.get("detection_confidence", 0.5),
                min_hand_presence_confidence=extraction.get("presence_confidence", extraction.get("detection_confidence", 0.5)),
                min_tracking_confidence=extraction.get("tracking_confidence", 0.5))))
            self.face = self.stack.enter_context(vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(paths["face"])), running_mode=vision.RunningMode.VIDEO, num_faces=1))) if "face" in paths else None
        except BaseException:
            self.stack.close()
            raise

    def detect(self, bgr, at):
        import cv2
        import numpy as np
        from .features import location_features, pack_hands
        start = perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        timestamp = max(self.last_timestamp + 1, int(at * 1000 + 0.5))
        self.last_timestamp = timestamp
        result = self.hands.detect_for_video(image, timestamp)
        raw, _ = pack_hands(result, self.swap)
        face = self.face.detect_for_video(image, timestamp) if self.face else None
        points = face.face_landmarks[0] if face and face.face_landmarks else None
        location = location_features(raw, points, rgb.shape[1], rgb.shape[0])
        return Observation(at, raw, location, rgb.shape[1], rgb.shape[0], (perf_counter() - start) * 1000)

    def close(self):
        self.stack.close()


def prepare_observations(info, observations):
    import numpy as np
    from .features import (auto_hand, mirror_frames, normalize_frame, normalize_sequence,
                           resample_legacy, resample_sign, sign_window, static_pose)
    if not observations:
        raise ValueError("No frames captured.")
    width, height = observations[0].width, observations[0].height
    if any((o.width, o.height) != (width, height) for o in observations):
        raise ValueError("Camera dimensions changed during capture. Start a new attempt.")
    span = observations[-1].at - observations[0].at
    fps = (len(observations) - 1) / span if len(observations) > 1 and span > 0 else 30.0
    raw = np.asarray([o.raw for o in observations], np.float32)
    locations = np.asarray([o.location for o in observations], np.float32)
    start, end = sign_window([o.present for o in observations], fps)
    raw, locations = raw[start:end + 1], locations[start:end + 1]
    present = np.any(raw[:, 126:128] > 0.5, axis=1)
    hand_rate = float(present.mean())
    minimum = info.preprocessing.get("sign_window", {}).get("minimum_hand_detection", 0.8)
    if not present.any():
        raise ValueError("No hand detected. Keep the signing hand visible.")
    if hand_rate < minimum:
        raise ValueError(f"Hand detection {hand_rate:.0%} is below {minimum:.0%} in this sign window.")
    normalization = info.preprocessing["normalization"]
    normalized = normalize_sequence(raw, width, height) if normalization == "hand-relative-sequence-v1" else np.asarray([normalize_frame(f, width, height) for f in raw])
    if normalization == "hand-face-location-v1":
        for slot in range(2):
            locations[normalized[:, 126 + slot] == 0, slot * 6:slot * 6 + 6] = 0
        normalized = np.concatenate([normalized, locations], axis=1)
    if info.variant == "static":
        pose = static_pose(normalized, 1, raw, width / height)
        sample = np.repeat(normalized[pose:pose + 1], 32, axis=0)
    elif info.features == 141 or "resampleSign" in info.preprocessing.get("interpolation", ""):
        sample = resample_sign(normalized)
    else:
        sample = resample_legacy(normalized)
    canonical = info.preprocessing.get("handedness", {}).get("canonical", False)
    if canonical and auto_hand(raw, width, height) == "left":
        sample = mirror_frames(sample)
    if not np.any(sample[:, 126:128] > 0.5):
        raise ValueError("No usable hand geometry detected.")
    if info.features == 141 and not np.any((sample[:, 140] > 0.5) & np.any(sample[:, 126:128] > 0.5, axis=1)):
        raise ValueError("Family needs the face and signing hand visible together.")
    return sample, {"frames": len(observations), "hand_rate": hand_rate, "fps": fps,
                    "capture_seconds": span, "mean_detector_ms": float(np.mean([o.detector_ms for o in observations]))}


def _preview(bgr, observation=None):
    import cv2
    height, width = bgr.shape[:2]
    view = cv2.resize(bgr, (420, max(1, int(420 * height / width))), interpolation=cv2.INTER_AREA)
    for slot, color in [(0, (80, 200, 240)), (1, (120, 230, 110))]:
        if observation is None or not observation.raw[126 + slot]:
            continue
        points = observation.raw[slot * 63:(slot + 1) * 63].reshape(21, 3)
        for x, y, _ in points:
            cv2.circle(view, (int(x * view.shape[1]), int(y * view.shape[0])), 2, color, -1)
    view = cv2.flip(view, 1)  # display only; detection and stored coordinates stay unmirrored
    ok, binary = cv2.imencode(".png", view, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return base64.b64encode(binary).decode("ascii") if ok else None


def run_live(settings, model: Path, camera=0, mode="automatic", target=None, video=None, control_stdin=False,
             max_attempts=None, job=None, emit=None, preview=True):
    import cv2
    job = job or Job()
    output = emit or (lambda event: print(json.dumps(event, allow_nan=False), flush=True))
    output_lock = threading.Lock()

    def emit(event):
        # Camera previews and scored results share one JSON-lines stream.
        with output_lock:
            output(event)

    emit({"type": "status", "message": "Loading the model and warming up inference..."})
    info = describe_model(model)
    if target is not None and target not in info.classes:
        raise ValueError("The expected sign is not included in this model.")
    predictor = Predictor(info)
    paths = detector_paths(settings, info)
    session = TestSession(settings, info, "live_test", {"camera": camera, "mode": mode, "video": str(video) if video else None,
                                                       "backend": "Python Keras + MediaPipe", "warm_up_excluded": True, "initial_target": target,
                                                       "preview_target_fps": 30, "camera_buffer": "latest frame"})
    commands = queue.Queue()
    if control_stdin:
        def read_commands():
            for line in sys.stdin:
                try:
                    commands.put(json.loads(line))
                except ValueError:
                    continue
            commands.put({"type": "stop"})
        threading.Thread(target=read_commands, daemon=True).start()
    capture, detector, camera_stream = None, None, None
    controller, recent = SignCapture(mode), deque()
    status, error, stop, frame_count = "complete", None, False, 0
    next_preview, last_prediction, origin = -1.0, -1.0, perf_counter()
    preview_times = deque(maxlen=60)
    state_lock = threading.Lock()
    preview_state = {"observation": None, "capturing": False, "fps": None}

    def show_preview(bgr, at):
        nonlocal next_preview
        now = perf_counter()
        if not preview or now < next_preview - 1e-9:
            return
        next_preview = now + 1 / 30 if next_preview < 0 else max(next_preview + 1 / 30, now)
        preview_times.append(now)
        preview_fps = (len(preview_times) - 1) / (preview_times[-1] - preview_times[0]) if len(preview_times) > 1 and preview_times[-1] > preview_times[0] else None
        with state_lock:
            state = dict(preview_state)
        observation = state["observation"]
        # Old landmarks must not float over a new pose during slow inference.
        overlay = observation if observation is not None and 0 <= at - observation.at <= 0.2 and (observation.height, observation.width) == bgr.shape[:2] else None
        emit({"type": "frame", "png": _preview(bgr, overlay),
              "left": bool(overlay.raw[126]) if overlay else False,
              "right": bool(overlay.raw[127]) if overlay else False,
              "face": bool(overlay.location[12]) if overlay else False,
              "detector_ms": observation.detector_ms if observation else 0,
              "capturing": state["capturing"], "fps": state["fps"], "preview_fps": preview_fps, "at": at})

    def score(items, reason):
        try:
            started = perf_counter()
            sample, quality = prepare_observations(info, items)
            result = predictor.predict(sample)
            result["pipeline_ms"] = (perf_counter() - started) * 1000
            row = session.add(result, target, mode=controller.mode, source="video_replay" if video else "webcam", reason=reason, **quality)
        except ValueError as failure:
            row = session.add(expected=target, mode=controller.mode, source="video_replay" if video else "webcam", error=str(failure), reason=reason)
        emit({"type": "attempt", "row": row, "path": str(session.directory)})

    try:
        detector = FrameDetector(info, paths)
        capture = cv2.VideoCapture(str(video)) if video else cv2.VideoCapture(int(camera))
        if not capture.isOpened():
            raise ValueError("Camera/video could not be opened. Check the camera index and close other camera applications.")
        if not video:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)
        file_fps = float(capture.get(cv2.CAP_PROP_FPS)) if video else None
        if video and (not math.isfinite(file_fps) or file_fps <= 0):
            raise ValueError("Replay video has no valid frame rate.")
        emit({"type": "ready", "classes": info.classes, "category": info.category, "variant": info.variant,
              "threshold": session.threshold, "path": str(session.directory), "mode": controller.mode})
        if not video:
            camera_stream = CameraStream(capture, origin, show_preview)
            camera_stream.start()
        while not stop:
            job.check()
            while not commands.empty():
                command = commands.get_nowait()
                action = command.get("type")
                if action in ("stop", "capture", "finish", "target", "mode", "reset"):
                    append_event(session.directory / "events.jsonl", {"type": "control", "action": action,
                                                                       "label": command.get("label"), "mode": command.get("mode")})
                if action == "stop":
                    stop = True
                elif action == "capture" and controller.mode == "manual":
                    controller.start()
                    with state_lock:
                        preview_state["capturing"] = True
                    emit({"type": "status", "message": "Capturing. Finish the attempt or wait for the 3-second limit."})
                elif action == "finish" and controller.mode == "manual":
                    finished = controller.finish()
                    with state_lock:
                        preview_state["capturing"] = False
                    if finished:
                        score(*finished)
                elif action in ("target", "mode", "reset"):
                    if action == "target":
                        new_target = command.get("label")
                        if new_target is not None and new_target not in info.classes:
                            emit({"type": "status", "message": "Expected sign is not in this model."})
                            continue
                        target = new_target
                    if action == "mode":
                        if command.get("mode") not in ("manual", "automatic"):
                            continue
                        controller.mode = command["mode"]
                    controller.reset()
                    recent.clear()
                    with state_lock:
                        preview_state.update(observation=None, capturing=False, fps=None)
                    emit({"type": "status", "message": "Ready for a new attempt. Capture reset."})
            if stop:
                break
            if video:
                ok, bgr = capture.read()
                if not ok:
                    finished = controller.finish("end_of_video")
                    if finished:
                        score(*finished)
                    break
                at = frame_count / file_fps
            else:
                frame = camera_stream.read()
                if frame is None:
                    continue
                at, bgr = frame
            frame_count += 1
            observation = detector.detect(bgr, at)
            recent.append(observation)
            while recent and at - recent[0].at > 3.0:
                recent.popleft()
            finished = controller.push(observation)
            with state_lock:
                preview_state.update(observation=observation, capturing=controller.active is not None,
                                     fps=(len(recent) - 1) / (recent[-1].at - recent[0].at) if len(recent) > 1 and recent[-1].at > recent[0].at else None)
            if finished:
                score(*finished)
            now = perf_counter()
            if video:
                show_preview(bgr, at)
            if now - last_prediction >= 0.6 and observation.present and not finished:
                try:
                    items = [observation] if info.variant == "static" else list(recent)
                    if info.variant == "static" or items[-1].at - items[0].at >= 0.4:
                        sample, _ = prepare_observations(info, items)
                        emit({"type": "estimate", **predictor.predict(sample)})
                except ValueError:
                    pass
                last_prediction = now
            if max_attempts and len(session.rows) >= max_attempts:
                break
    except Cancelled:
        emit({"type": "status", "message": "Stopping. Saving the session reports..."})
    except BaseException as failure:
        status, error = "failed", str(failure)
        raise
    finally:
        if camera_stream is not None:
            camera_stream.close()
        elif capture is not None:
            capture.release()
        if detector is not None:
            detector.close()
        summary = session.close(status, error)
        emit({"type": "session_complete", "path": str(session.directory), "summary": summary, "status": status, "error": error})
    return session.directory
