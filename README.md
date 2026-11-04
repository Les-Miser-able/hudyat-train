# HUDYAT Train

A native desktop application and CLI for the middle of the HUDYAT pipeline:

`hudyat-track recordings -> hudyat-train landmarks and models -> HUDYAT browser inference`

The agreed manuscript and implementation decisions are recorded in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md). This project does not modify the manuscript, collector, or HUDYAT application.

**Start the application**

Use Python 3.12, 64-bit, including its Tcl/Tk component. From PowerShell in this folder, run:

```powershell
.\setup.ps1
.\launch.cmd
```

After setup, double-click `launch.cmd` to open the desktop application. All processing is local once Python packages and the detector assets are installed. Installing dependencies needs a network connection. `setup.ps1` uses this project's `.venv` and pins TensorFlow 2.19.1, MediaPipe 0.10.21, and NumPy 1.26.4.

The desktop shell and dataset scanner use the Python standard library. Extraction needs MediaPipe/OpenCV/NumPy/Pillow; training and export need TensorFlow/NumPy/Matplotlib. The Settings tab can select separate Python executables for these operations. Blank interpreter fields use the Python that launched the application. This is useful for testing with existing environments, but final study runs require TensorFlow 2.19.1.

The hand and face `.task` files belong in `models/`; see [models/README.md](models/README.md). Matching copies from the collector are already present in this checkout.

**Add your raw data**

The category, class, and S1-S7 folders have been created. Drag files into them using File Explorer:

```text
dataset/raw/
  alphabet/
    letter_a/                 # "A" is also accepted
      S1/
        take_01.mp4
        image/reference.jpg
      S2/
        video/take_01.mov
  family/
    father/                  # "Father" is also accepted
      S1/
        take_01.mp4
```

Folders below the signer can be nested freely. Supported media: MP4, AVI, MOV, MKV, WEBM, M4V, JPG, JPEG, PNG, BMP, and WEBP. Each video must contain one sign, with a sign window no longer than three seconds. Use original, unmirrored media and independent takes. Do not add augmented copies, extracted frames from the same video, or recompressed duplicates as independent observations.

The nine category IDs are `alphabet`, `numbers`, `greetings`, `calendar`, `shapes`, `colors`, `family`, `wh_questions`, and `word_concepts`. Display names such as `WH Questions` also work as raw folder names. Unknown labels and invalid signer folders are reported instead of silently creating new classes.

No metadata file is required. If you copy the collector's `dataset/metadata.csv` into this project's `dataset/`, its paths must match your copied raw folder names. Rows marked `needs_rerecord` are excluded; only `kept` rows are used. Unlisted media can still be used. Temporary `.tmp` recordings and hidden files are ignored. Identical file bytes in different class or signer folders are errors; identical copies within one class/signer are counted once. A content hash cannot identify a recompressed duplicate, so keep only the original take.

Raw media is read in place. Extraction does not move or delete it. `dataset/cache/` contains detected hand coordinates and derived face-location values; full face landmark arrays are not retained. `dataset/extracted/` contains the model inputs and an index recording accepted/rejected sources and their reasons.

**Use the desktop workflow**

1. In **Dataset**, scan the selected version. Select a category to see its sign-by-signer counts. These are raw counts before extraction quality checks.
2. In **Extract**, extract a category, or choose `all`. Hand detection must reach 80% inside the trimmed sign window. Two-hand signs also require 80% both-hands detection. Family requires usable face-location information.
3. In **Train**, choose `random` or `loso`. A study run requires every expected class and at least 15 accepted sources for each of S1-S7. Use the explicit **Development run** option for incomplete data or fewer epochs.
4. In **Test models**, select a model/run and use automated held-out testing or live webcam testing. Both automatically save their results.
5. In **Results**, browse the comparison, summary, per-class, and attempt tables, and view graphs or open the complete HTML report.
6. In **Export**, select a completed random-split run. Export a bundle, then verify it against the installed HUDYAT client. The job log is also saved under `reports/jobs/`.

The Stop button requests cancellation at the next frame or training batch. Closing the application during a job first requests cancellation. Interrupted or failed runs cannot be exported.

**The two versions**

| Version | Input sources | Model input | Intended use |
|---|---|---|---|
| `sequence` | Videos; images for classes marked static | Time-ordered, trimmed, uniformly resampled sequences | Static and moving signs, including J and Z |
| `static` | Images or a stable pose chosen from a video | One pose repeated to 32 frames | Held-pose comparison version; excludes classes marked dynamic |

Both use the same Conv1D-BiLSTM architecture and separate sample/run/export directories. Alphabet has 26 classes in the sequence version and 24 in the static version. The full vocabulary has 108 signs; the static subset has 106. An image cannot teach a moving sign, so still images of J/Z are excluded from sequence extraction.

Family uses 141 features, including 13 hand-to-face location values. Other categories use 128. Hand slots, normalization, resampling, and canonical-hand handling match HUDYAT's current runtime; [docs/FEATURE_CONTRACT.md](docs/FEATURE_CONTRACT.md) gives the exact mapping.

`signs.csv` keeps the manuscript vocabulary fixed. `mode` currently marks only J and Z dynamic, following Appendix A. `mirror_ok` controls expert-approved mirrored training augmentation. Its initial values were copied from the collector and are all `0`; change these only after approval. Other augmentation still generates five modified training copies. No automatic linguistic approval is inferred from a recording's presence.

**Training and reports**

Study defaults: Adam at 0.001; batch size 32; at most 150 epochs; early stopping on validation loss with patience 20 and best weights restored; ReduceLROnPlateau; class weighting; five augmented copies per original training source. Augmentation includes noise, scale, rotation, timing changes, frame removal, and approved mirroring. Static augmentation preserves a held pose.

In **Train > Data augmentation**, leave **Enable training augmentation** checked and **Augmented copies per original** at **5** for the study configuration. Each training source contributes one original plus five generated landmark samples. Rotation is up to 15 degrees, scale is 0.9-1.1, and landmark noise has a standard deviation of 0.01; sequence inputs also use timing changes (0.8-1.2) and frame removal (8% probability). Mirroring remains restricted to signs explicitly approved in `signs.csv`. Augmented samples are generated in memory after splitting; raw files and held-out samples are never augmented. Each run/fold saves `augmentation.csv` with the source, signer, label, and generated count, as well as an augmentation event and counts in `run.json`.

For experiments, enable **Development run** to change the copy count or turn augmentation off. The CLI equivalent is `train ... --development --augmentation-copies 3`; `--augmentation-copies 0` disables it. Omitting the option uses five copies. Changing augmentation affects newly trained models; existing checkpoints retain their recorded training settings.

Random splitting is stratified by class, approximately 70/15/15 with integer rounding and at least one independent source per split. LOSO holds each signer out for testing once, reserves the next signer for validation, and trains on the remaining signers. With seven signers, each fold uses five for training, one for validation, and one for testing. No held-out test signer is used for model selection. At least three signers are needed even for a development LOSO run.

Each run stores the `.keras` checkpoint, exact class ordering, preprocessing, source checksums, split assignments, configuration, history JSON/CSV, learning curves, predictions, per-class metrics, and a confusion matrix. LOSO additionally reports fold scores, pooled predictions, and mean/sample-standard-deviation metrics. LOSO models are evaluation artifacts; use a completed random-split run for export.

The classification benchmark is accuracy >=85%, macro F1 >=85%, and recall >=70% for each sign. Confidence rejection at 85% is reported separately from classification accuracy. Deployment response time must be measured in HUDYAT's browser on the target device; Python training timing and the parity check do not establish the manuscript's under-two-second target.

**Automatically recorded graphs and tables**

Training logs every completed epoch immediately to `events.jsonl`, `history.csv`, and `history.json`, so earlier epochs remain available if a later epoch fails or is interrupted. Run status, configuration, dataset identity, runtime versions, and completed metrics are recorded in a local SQLite history and exported to `reports/experiments.csv`. Completed test sessions use the same history. **Results** can import existing runs with **Refresh tables** and generate reports for older runs with **Build older reports**.

| Table | Contents | Saved file |
|---|---|---|
| Experiment comparison | Run/session, category, version, protocol, status, development flag, accuracy, precision/recall/F1, latency, dataset ID | `reports/experiments.csv` |
| Epoch history | Training/validation loss and accuracy, epoch, learning rate when available | `RUN/history.csv` |
| Data augmentation | Training source, signer, label, generated copies, total training samples, mirroring approval | `RUN/augmentation.csv` |
| Summary | Scores, sample counts, rejection/invalid counts, timing scope | `RUN_OR_SESSION/summary.csv` |
| Per class | Label, support, precision, recall, F1 | `RUN_OR_SESSION/per-class.csv` |
| Test attempts | Expected/predicted sign, confidence, correctness/rejection, model/pipeline timing, source, signer, fold | `SESSION/attempts.csv` |
| LOSO folds | Test/validation signer and scores per fold | `RUN/loso-folds.csv` |
| LOSO mean and SD | Fold means and sample standard deviations | `RUN/loso-summary.csv` |
| Parameters | Recorded configuration for reproducibility | `RUN_OR_SESSION/parameters.csv` |

PNG graphs include training/validation curves, a confusion matrix, per-class precision/recall/F1, LOSO fold/mean-SD comparisons, and test-session confidence/inference timing. The **Graphs** pane previews these; **Open full-size graph** and **Open HTML report** expose shareable local artifacts. The HTML report includes viewable tables and CSV links. Long attempt tables show the first 500 rows in the desktop; the CSV and HTML retain all attempts.

**Automated and manual testing**

**Automated held-out testing** replays the exact saved test split, checking the checkpoint, class order, split metadata, and sample checksums. It does not retrain or choose new test data. Selecting a LOSO parent tests all its saved fold checkpoints and their respective held-out samples. An exported model alone has no test-split provenance, so select the source training run for automated testing.

**Live webcam testing** accepts either a completed run/fold or an exported version folder containing `model.json` and `weights.bin`. Select the model and camera index, then press **Start camera**. The Python environment selected by **Train & test Python** must contain both TensorFlow and MediaPipe; the pinned setup provides them together.

The camera preview targets 30 FPS and refreshes independently of landmark detection and model prediction. Detection always takes the latest available camera frame, avoiding a growing queue of delayed video. The desktop displays separate preview and detection FPS; actual rates depend on the camera and computer. Hand points older than 0.2 seconds are hidden from the preview. Display mirroring and preview timing do not alter the model's landmark preprocessing. Video-file replay still processes every frame in order.

- Choose an **Expected sign** to score attempts, or leave **(unscored)** for exploration. The expected sign is never inferred from the model's own answer.
- In **Automatic sign window**, a hand appearing begins capture; hands disappearing for 0.2 seconds ends it. Windows have a 0.1-second prebuffer and a three-second limit. After a maximum-length attempt, lower the hands before the next attempt.
- In **Manual capture**, click **Start capture** (Space), perform a sign, then **Finish & evaluate** (Enter). A capture also ends after three seconds. Changing the expected sign or capture mode resets the pending capture and is logged.
- The camera view shows hand points, live estimates, top predictions, and detection timing. Estimates during a gesture are exploratory and do not add repeated scored attempts. Completed captures produce one logged row.
- Press **Stop & save reports** to close the camera and generate the session's graphs/tables. Closing the app also requests a stop. Camera pixels are previewed only; no video or preview image is saved in the result logs.

Live testing logs invalid captures, below-threshold predictions, correct/incorrect recognized signs, and unscored predictions separately. Raw top-1 accuracy uses valid labelled predictions. Successful-attempt rate also counts labelled rejections and invalid captures as failures. Unlabelled exploration has no accuracy score. Macro metrics in partial live sessions average only classes with ground-truth support, with untested classes explicitly listed; this is not complete study coverage.

Test sessions are stored in `reports/testing/<category>/<variant>/<timestamp-kind>/` with `session.json`, `events.jsonl`, `attempts.json`, `attempts.csv`, `summary.json`, report tables, PNG graphs, and `report.html`. Model timing excludes warm-up, camera capture, detection, preprocessing, GUI rendering, and browser work; separately reported pipeline timing adds Python input preparation. These diagnostic timings do not replace the manuscript's browser/device response-time tests.

**CLI examples**

Run from this folder with the configured environment:

```powershell
.\.venv\Scripts\python.exe -m hudyat_train init
.\.venv\Scripts\python.exe -m hudyat_train scan --variant sequence
.\.venv\Scripts\python.exe -m hudyat_train extract --category alphabet --variant both
.\.venv\Scripts\python.exe -m hudyat_train train --category alphabet --variant sequence --protocol random
.\.venv\Scripts\python.exe -m hudyat_train train --category alphabet --variant sequence --protocol loso
# Small development run; never a final study result:
.\.venv\Scripts\python.exe -m hudyat_train train --category alphabet --variant static --development --epochs 5
.\.venv\Scripts\python.exe -m hudyat_train export --run "runs/alphabet/sequence/YOUR_RUN"
.\.venv\Scripts\python.exe -m hudyat_train verify --bundle exports/sequence --hudyat "..\HUDYAT"
.\.venv\Scripts\python.exe -m hudyat_train audit --hudyat "..\HUDYAT"
.\.venv\Scripts\python.exe -m hudyat_train test --run "runs/alphabet/sequence/YOUR_RUN"
.\.venv\Scripts\python.exe -m hudyat_train results --rebuild
.\.venv\Scripts\python.exe -m hudyat_train live --run "runs/alphabet/sequence/YOUR_RUN" --camera 0 --mode manual
```

`doctor` checks installed packages and asset paths. `--root PATH` before the subcommand selects a project directory containing `project.json`. `export --allow-development` is required to export a development checkpoint for testing. `verify --tfjs PATH` can point at a separately installed `@tensorflow/tfjs` package if HUDYAT's client dependencies are not installed locally.

**Install a model into HUDYAT**

Exports default to `exports/sequence/` or `exports/static/`. They contain `catalog.json`, `models/<category>/<version>/{model.json,weights.bin,parity.json}`, and content-addressed shared `.task` files. Export preserves other categories already in the target catalog and updates the catalog only after writing the new release. Keep static and sequence releases separate; HUDYAT currently selects one release per category.

After reviewing the local bundle, run the exporter yourself with the target model-assets directory to merge the release into HUDYAT:

```powershell
.\.venv\Scripts\python.exe -m hudyat_train export --run "runs/alphabet/sequence/YOUR_RUN" --output "..\HUDYAT\server\model-assets"
```

For categories using the file catalog, HUDYAT reads that catalog directly. Other categories use database registrations; run `npm run models:register` from `HUDYAT/server` when appropriate. The current registration script does not rewrite `faceUrl`; Family currently uses the file-catalog route, which does handle it. Do not replace the complete existing catalog with a one-category staged catalog by copying it blindly.

The read-only `audit` report distinguishes file-format compatibility from lesson coverage. Current checked-in HUDYAT seeds contain four extra Colors prompts, two extra Shapes prompts, and four extra WH Questions prompts beyond the manuscript vocabulary. Calendar is correctly treated as months plus `days.json`; Violet/Purple is an existing runtime alias. HUDYAT also currently hard-codes an 80% acceptance gate instead of reading the exported 85% threshold. These main-project differences remain for a separate, agreed integration change.

**Validation**

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/smoke_pipeline.py --work tmp/synthetic-check
python tools/smoke_extraction.py --track "..\hudyat-track" --work tmp/real-check
# Once the synthetic smoke project and a video fixture exist:
python tools/smoke_testing.py --project tmp/synthetic-check --video "PATH_TO_ONE_SIGN_CLIP.mp4"
```

The smoke scripts write only isolated `tmp/` workspaces. Synthetic model scores are not FSL performance results. See [VALIDATION.md](VALIDATION.md) for the checks actually performed and their limits.
