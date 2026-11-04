# HUDYAT feature and export contract

The reference implementation is `HUDYAT/client/src/features/fsl-recognition/{category-core.js,normalize.ts,sign-capture.js}`. The Python functions are in `hudyat_train/features.py`. `python -m hudyat_train verify` compares synthetic inputs using the actual HUDYAT files.

| Indices | Meaning |
|---|---|
| 0-62 | Left hand, 21 landmark-major XYZ coordinates |
| 63-125 | Right hand, 21 landmark-major XYZ coordinates |
| 126, 127 | Left/right binary presence |
| 128-133 | Family: left wrist, thumb tip, index tip XY relative to face |
| 134-139 | Family: right hand location in the same format |
| 140 | Family: face presence |

MediaPipe Tasks handedness labels are used as returned, without pixel mirroring or swapping by default. Higher handedness confidence wins if two detections claim the same slot. A missing or degenerate hand has zero coordinates and a zero presence flag.

Per-frame normalization multiplies XYZ by image width, height, width; subtracts the wrist; and divides by the 2D wrist-to-middle-MCP pixel distance. A palm distance below 1e-6 makes the hand absent. This matches `normalizeFrame`, including its distinction from the collector's 3D-distance prose.

`hand-relative-sequence-v1` keeps normalized landmarks 1-20 and replaces landmark 0 with wrist displacement from the first valid frame divided by the clip's median 2D palm length. It is used for non-Family sequence models. A still image produces a constant sequence with zero wrist displacement.

`hand-face-location-v1` appends hand locations measured from face landmark 1 (nose), with aspect-corrected X and a 2D forehead(10)-to-chin(152) scale. It is used for Family in both versions. No face yields zero location coordinates and a zero face flag. No full face landmark arrays are written to disk. Static Family requires a selected pose with usable face information.

Trimming uses the first and last raw hand detections, with a 0.1-second margin at each end and a minimum window of eight frames. Rounding is half-up, matching JavaScript Math.round for these nonnegative values. No movement threshold is used to find the sign window. Hand and both-hands detection rates are measured within that window. A video sign window over three seconds plus one frame of rounding tolerance is rejected.

Resampling always produces 32 float32 frames:

- Non-Family sequence models use HUDYAT's current `category-core.resample`: linear coordinates, nearest binary flags, and zero coordinates for a nearest absent hand. This existing function blends a present hand toward zero at detection gaps; reproducing it is necessary for current-runtime parity.
- Family uses `resampleSign`: hand coordinates are interpolated only if the hand is present on both sides; locations only if that hand and the face are present on both sides. Otherwise the nearest frame is copied.
- Static samples repeat one stable valid pose to 32 frames. Pose selection considers hand-shape changes, raw wrist movement, available face location, and proximity to the window's center.

Family, and non-Alphabet static models, use the current browser's canonical-hand rule: presence majority, then wrist travel in hand-size units. Left-dominant inputs are mirrored in normalized space, swapping slots and negating X, including face-location X. Non-Family sequence models and static Alphabet keep the detected slots; their mirrored augmentation is controlled by explicit per-sign approvals.

The exact network matches `category-core.buildModel`, including BatchNorm momentum 0.99/epsilon 0.0001, sigmoid recurrent activation, concatenated bidirectional outputs, and Dense/Softmax dimensions. Keras weights are written per layer as little-endian float32. Each specification records shape, byte offset, and byte count. The release identity includes weights, class order, preprocessing, category, version type, and development status.

Only complete, evaluated random-split runs can be exported. Checkpoint, class order, preprocessing, configuration, source index, splits, and metric checksums are checked before export. Matching detector assets are packaged by SHA-256. Existing versioned files with conflicting content are rejected. The catalog is atomically updated after the files are ready. No model is registered with HUDYAT's database automatically.

Synthetic parity checks use all three supported normalization functions, both resampling functions, hand mirroring, canonical-hand selection, sign-window rounding, and 128/141-feature inference. They cover landscape and portrait aspect ratios and detection gaps. These checks verify numerical behavior; physical camera behavior and inference latency still need deployment-device tests.
