# Detector assets

Place the MediaPipe Tasks model files here. The existing collector copies used for this checkout are:

| File | SHA-256 |
|---|---|
| `hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `face_landmarker.task` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

The face file is required only for Family. Every extraction index records the exact hashes; export checks that the supplied detectors still match and packages them with the model. No download happens during extraction.

Use the same model files across recording, extraction, and browser inference. The upstream model locations are the official [hand model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and [face model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task). Check downloaded file hashes before substituting them for the existing copies.
