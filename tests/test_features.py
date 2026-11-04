from types import SimpleNamespace as NS

import numpy as np
import pytest

from hudyat_train.features import (auto_hand, mirror_frames, normalize_frame, normalize_sequence,
                                    pack_hands, prepare, resample_legacy, resample_sign, sign_window)


def test_sign_window_keeps_static_hands_and_uses_half_up_margin():
    assert sign_window([False] * 4 + [True] * 8 + [False] * 8, 25) == (1, 14)
    assert sign_window([False] * 12, 30) == (0, 11)
    assert sign_window([False, True, False], 30) == (0, 2)
    assert sign_window([False] * 10 + [True] + [False] * 10, 1) == (7, 14)


def test_hand_slots_are_contiguous_and_duplicate_label_uses_best_score():
    points = lambda x: [NS(x=x, y=0.2, z=0.1) for _ in range(21)]
    result = NS(hand_landmarks=[points(0.1), points(0.8), points(0.4)],
                handedness=[[NS(category_name="Right", score=0.7)], [NS(category_name="Left", score=0.9)], [NS(category_name="Right", score=0.95)]])
    frame, collisions = pack_hands(result)
    assert collisions == 1
    assert frame[0] == pytest.approx(0.8) and frame[63] == pytest.approx(0.4)
    assert frame[126:].tolist() == [1, 1]


def test_degenerate_hand_is_absent():
    frame = np.zeros(128, np.float32)
    frame[127] = 1
    assert not normalize_frame(frame, 1280, 720).any()


def test_sequence_preserves_wrist_motion_while_frame_removes_it(detections):
    raw = detections["raw"][3:29]
    sequence = normalize_sequence(raw, 1280, 720)
    frames = np.asarray([normalize_frame(f, 1280, 720) for f in raw])
    assert np.allclose(frames[:, 63:66], 0)
    assert sequence[-1, 63] > sequence[0, 63]
    assert np.allclose(sequence[:, 66:], frames[:, 66:])


def test_interpolation_matches_both_existing_browser_paths():
    frames = np.zeros((2, 128), np.float32)
    frames[0, :63], frames[0, 126] = 2, 1
    legacy, masked = resample_legacy(frames, 5), resample_sign(frames, 5)
    assert legacy[1, 0] == 1.5
    assert masked[1, 0] == 2
    assert not legacy[2:, :63].any() and not masked[2:, :63].any()
    assert set(legacy[:, 126]) <= {0, 1}


def test_location_does_not_interpolate_through_missing_face():
    frames = np.zeros((2, 141), np.float32)
    frames[:, 127] = 1
    frames[0, 140], frames[0, 134:140] = 1, 2
    output = resample_sign(frames, 5)
    assert output[1, 134] == 2 and output[2, 134] == 0
    assert output[:, 140].tolist() == [1, 1, 0, 0, 0]


def test_mirroring_twice_restores_hands_locations_and_presence(detections):
    sample, _, _ = prepare(detections["raw"], detections["locations"], 1280, 720, 30, "family", "sequence")
    assert np.array_equal(mirror_frames(mirror_frames(sample)), sample)


def test_static_has_one_pose_sequence_retains_time_and_family_has_location(detections):
    args = (detections["raw"], detections["locations"], 1280, 720, 30)
    static, static_norm, stats = prepare(*args, "alphabet", "static")
    sequence, seq_norm, _ = prepare(*args, "alphabet", "sequence")
    family, family_norm, _ = prepare(*args, "family", "sequence")
    assert static.shape == (32, 128) and np.all(static == static[0])
    assert sequence.shape == (32, 128) and not np.all(sequence == sequence[0])
    assert family.shape == (32, 141) and family[:, 140].any()
    assert static_norm == "hand-relative-v1" and seq_norm == "hand-relative-sequence-v1"
    assert family_norm == "hand-face-location-v1"
    assert stats["hand_rate"] == 26 / 32
    assert auto_hand(detections["raw"], 1280, 720) == "right"
