# Validation performed

Validated locally on Windows with Python 3.12.10 and Node.js 24.21.0. These checks verify the application and pipeline mechanics. They are not the manuscript's recognition results.

**Automated checks: 64 passed**

`hudyat-track/.venv/Scripts/python.exe -m pytest -q -rs` passed all 64 tests. Coverage includes:

- Fixed vocabulary, class-folder aliases, nested media, signer coverage, metadata exclusions, and duplicate-content errors.
- Hand-slot ordering, duplicate handedness labels, degenerate hands, sign-window rounding, wrist-motion preservation, missing-hand/face interpolation, static poses, and mirroring.
- Cached extraction, quality rejection, cancellation, stale raw-data detection, source-disjoint random splits, signer-independent LOSO splits, training-only augmentation, and classification metrics.
- Development export restrictions, rejection of LOSO exports, and protection against changed class ordering after evaluation.
- Desktop construction with seven tabs, an actual background CLI scan, job completion, category/sign coverage, result tables, and live-test target controls.
- Concurrent experiment-history writes, recovery of older runs, immediate test-attempt persistence, failure status, unlabelled/partial test scoring, and rejection accounting.
- Automatic/manual capture boundaries, maximum-length rest requirements, blip rejection, live/offline preprocessing equality for 128/141-feature models, saved-test-only evaluation, checksum guards, and train/test-overlap rejection.
- Actual CSV/PNG/HTML report generation, including summary, per-class, and attempt tables.
- A 30 FPS preview source, fast desktop refresh, preview progress during blocked landmark detection and prediction, latest-frame buffering, disconnect propagation, and unchanged model pixels during PNG preview generation.
- Augmentation counts of zero, two, and five through the CLI and training call, unmodified validation inputs, train-only source records, and desktop augmentation controls.

The extraction test environment has NumPy 1.26.4, MediaPipe 0.10.21, OpenCV-contrib 4.11.0.86, and pytest 9.1.1. The native desktop startup check also caught and fixed Tcl executable discovery for redirected Windows launches. Visual interaction on the user's display was not tested; the desktop checks instantiate its widgets and run its background job workflow.

**Real media extraction**

`tools/smoke_extraction.py --work tmp/real-extraction-final` copied one kept Alphabet A video and one kept Family Father video from the collector into an isolated temporary workspace. It did not populate the study dataset or change the source files.

- Both videos were accepted in sequence and static versions.
- Alphabet produced 32x128 samples, with 87.5% hand detection in the sign window.
- Family produced 32x141 samples, with approximately 87.8% hand detection in the sign window.
- A still image from the Alphabet clip passed the IMAGE detector path at 100% hand detection.
- A still image from the Father clip was rejected because IMAGE-mode hand detection did not yield a usable held pose. This is a recorded quality rejection, not an assertion that every video frame is usable as an independent still image.

The detailed source/quality results are in `tmp/real-extraction-final/smoke-results.json`. No sign-recognition model was trained on these clips during validation.

**Synthetic training and export**

`tools/smoke_pipeline.py --work tmp/synthetic-final` exercised actual Keras training, best-checkpoint saving/loading, evaluation reports, plot generation, artifact checksums, and export for all four representative combinations:

| Category | Version | Input |
|---|---|---|
| Alphabet | sequence | 32x128 |
| Alphabet | static | 32x128 |
| Family | sequence | 32x141 |
| Family | static | 32x141 |

Every run used two synthetic classes and one epoch, with explicit development flags. The synthetic detectors were test fixtures, not real media extraction. Their scores measure no FSL recognition capability. Artifacts remain under `tmp/synthetic-final/`, separate from normal exports and the real raw-data directory.

A separate three-fold development LOSO run completed using the same synthetic Alphabet data. Each of the three signers was held out once for testing, another was used for validation, and the remaining signer trained the model. All fold reports and the aggregate mean/sample-standard-deviation summary were written successfully.

The original training environment used TensorFlow 2.21.0, Keras 3.15.1, and NumPy 2.3.5. The manuscript-pinned TensorFlow 2.19.1 environment could not be installed during that initial restricted network session. The later validation below uses the now-installed pinned environment. Non-development study runs still reject other TensorFlow versions. No manuscript or final-study claim was changed to match a test environment.

**HUDYAT numerical compatibility**

The verifier imported HUDYAT's actual `buildModel`, `loadWeights`, normalization, resampling, hand-selection, mirroring, and trimming functions. It used the existing TensorFlow.js installation in `fsl_trainer/pwa-sample/node_modules/@tensorflow/tfjs` because HUDYAT's own client node_modules was not installed in this checkout.

| Exported synthetic release | Largest Keras vs HUDYAT probability difference |
|---|---|
| Alphabet sequence | 0 |
| Alphabet static | 2.98e-8 |
| Family sequence | 5.96e-8 |
| Family static | 5.96e-8 |

Preprocessing across 1280x720, 640x480, and 720x1280 synthetic camera frames had a maximum difference of 5.77e-8. This includes detection gaps and missing faces. The tolerance is 2e-4. Model, weight, and detector checksums passed. Each bundle's `verification.json` records the compared release hashes and the actual HUDYAT runtime file hashes.

**Integration audit**

`python -m hudyat_train audit` read the current main-project source and generated `reports/integration-audit.json`. It found these lesson prompts outside the fixed study vocabulary:

- Colors: Black, White, Pink, Brown. Violet/Purple is already handled by HUDYAT's existing alias.
- Shapes: Pentagon, Hexagon.
- WH Questions: Which, How Many, How Much, Whose.

Calendar's month/day merge matches the 22 study signs. The current recognition implementation uses a hard-coded 0.8 score gate, while the manuscript/export metadata specify 0.85. The current database registration script does not rewrite faceUrl; Family currently uses the file-catalog route, which does. The audit does not inspect the deployed database or change HUDYAT.

**Remaining validation**

Seven-signer training/evaluation on the intended study dataset, recognition accuracy, approved left-handed coverage, full reference validation, actual browser camera behavior, and response time on deployment hardware remain unmeasured. Numerical export parity establishes format and preprocessing compatibility only. No release was installed into HUDYAT or registered in its database.

**Automated recording and model-testing extension**

The new training hooks were exercised with a fresh one-epoch synthetic random run and a three-fold synthetic LOSO run. Both produced the experiment-history rows, event logs, epoch tables, summary/per-class tables, HTML reports, and graphs. LOSO additionally produced fold and mean/SD CSV tables and `loso-metrics.png`. This remains development-only data.

`tools/smoke_testing.py` checked all four synthetic combinations (Alphabet/Family, sequence/static) through both native Keras checkpoints and exported weight loaders: eight load/inference paths passed against the stored parity probes. It then ran automated testing on two saved random-split samples and all 18 held-out samples across three LOSO folds. Each test session saved its predictions, confidence outcomes, model timing, graphs, and tables.

The live-worker pipeline was exercised with a copied Alphabet video, not a webcam. Automatic capture ended when the hands went down and produced a single logged attempt. A separate process-level manual test received Start capture and Finish commands over the same control pipe used by the desktop, delivered PNG preview events, recorded one manual attempt, and saved its report. Preview pixels were not included in the result logs. Physical webcam access, focus, lighting, and camera-driver behavior still need a device check.

The local compiled TensorFlow inference function matched the saved probabilities; warm-up is excluded from reported timings. Timing is scoped to Python model calls or Python input preparation plus model calls, rather than camera-to-screen or HUDYAT browser response time. The available extension-validation environment remains TensorFlow 2.21.0 / Keras 3.15.1 / NumPy 2.3.5 / MediaPipe 1.0.1; the project setup still pins the manuscript stack.

Detailed extension proof is saved under `tmp/synthetic-final/reports/testing-smoke-validation.json`. Test reports are under that temporary project's `reports/testing/`, and experiment tables are under its `reports/experiments.csv`. None of these synthetic models or diagnostic sessions were copied into the real project's runs or exports.

**Smooth preview and augmentation controls (2026-11-05)**

The preview previously emitted only 18 images for a three-second, 30 FPS source, while the desktop refreshed every 100 ms. Regression tests now exercise the same live-worker and desktop paths. The worker targets 30 preview FPS, the desktop polls every 16 ms while live, and an independent camera thread continues capture/preview during both detection and prediction. Detection receives only the newest available camera frame. Capture timestamps and the model's existing numerical preprocessing are preserved; video-file replay still processes every frame in order.

The current project's `.venv` reports TensorFlow 2.19.1, NumPy 1.26.4, MediaPipe 0.10.21, and OpenCV-contrib 4.11.0.86. A fresh `tools/smoke_pipeline.py --work tmp/preview-augmentation-check-20261105` run completed real one-epoch synthetic training and export for Alphabet/Family, sequence/static. Each random run used 14 training originals plus 70 generated copies, for 84 training samples, with two unmodified validation and two unmodified test sources. Each run saved `augmentation.csv` and augmentation events/counts.

`tools/smoke_testing.py` then passed all eight native/exported inference paths, with a maximum probability difference of 5.96e-8. It replayed the saved random test split and all 18 test samples across a fresh three-fold LOSO run. The process-level manual capture check received PNG previews, accepted Start/Finish commands, and saved exactly one attempt. Zero/two/five-copy training controls and source isolation are separately covered by the automated tests.

A paced replay of the copied Alphabet recording exercised the new webcam code path with real MediaPipe detection and Keras prediction, without opening a physical webcam. It emitted 89 preview frames at 29.87 FPS over about three seconds; the largest measured preview gap was 48.9 ms and mean detector time was 27.8 ms. It saved one attempt and released the capture source. Proof is in `tmp/preview-augmentation-check-20261105/reports/preview-thread-validation.json`; loader/manual-test proof is in the same project's `reports/testing-smoke-validation.json`.

These checks use synthetic models and an explicitly selected recording. Physical webcam drivers, display smoothness, lighting, and actual device performance still need a user-device check. No new study data, final recognition results, or deployed HUDYAT release were produced by these checks.
