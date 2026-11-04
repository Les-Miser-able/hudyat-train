# Agreed project context

The user confirmed this flow:

1. `hudyat-track` collects original media.
2. `hudyat-train` owns local raw data, extraction, training, evaluation, and export.
3. `HUDYAT` runs the exported models in its browser client.

The supplied manuscript remains fixed. Changes to its methodology or wording require discussion with the user's group. No manuscript changes have been made.

The user's explicit implementation decisions are:

- Measure the 80% detection criterion inside the sign window.
- Organize user-supplied raw data by category/class/signer, accepting videos and images in nested subfolders. The user will place this data in hudyat-train; there is no automatic import of the collector's full dataset.
- Provide a static-only version and a combined static/movement sequence version.
- Preserve HUDYAT's current model.json/weights.bin loading format.
- Build a desktop application, with CLI tools for the same workflow.

Raw-file retention and the export route are explicit implementation choices that differ from some manuscript wording. Source files stay in place; neither extraction nor training deletes them. The manuscript's SavedModel/tensorflowjs_converter/quantization wording is not implemented because the user explicitly selected the existing HUDYAT loader. These differences are documented for later group discussion rather than silently changing the document.

The study vocabulary is 108 signs in nine categories: Alphabet 26, Numbers 10, Greetings 10, Calendar 22, Shapes 8, Colors 6, Family 10, WH Questions 6, Word Concepts 10. Recordings are expected from S1-S7, with 15 repetitions per sign per signer. The complete target is 11,340 independent recordings before quality rejection. Public imports are outside this manuscript dataset. Reference signs, rather than every recorded take, are expert-validated.

The main model uses time-ordered 32-frame sequences. Family adds face-relative hand location for a 32x141 input; other categories use 32x128. The architecture is Conv1D(64), BatchNorm, MaxPool, Dropout, Conv1D(128), BatchNorm, Dropout, BiLSTM(128, return sequences), Dropout, BiLSTM(64), Dropout, Dense(64), Softmax. Dropout is 0.3. Study defaults and evaluation are recorded in README.md and each run's configuration.

LOSO is the principal estimate for unseen signers. The validation signer is separate from training and testing in every fold. Random 70/15/15 evaluation is also available. Augmentation never precedes splitting. Test data is not used for early stopping or learning-rate selection.

The separate static-only version is a comparison configuration requested by the user. It repeats a stable pose and excludes dynamic classes. It does not establish motion recognition for J/Z or replace the full sequence study model. Current Appendix A labels only J/Z dynamic; other labels are preserved until the group decides otherwise.

The trainer's numerical preprocessing follows the current HUDYAT source, which differs from the collector's README in two details: flags are at 126/127 after both hand blocks, and palm scaling uses 2D pixel distance. The 128-feature sequence path also preserves wrist displacement and follows the browser's existing gap interpolation. These details are tested against HUDYAT's actual functions.

Known integration differences are reported by `python -m hudyat_train audit`: additional HUDYAT lesson labels, an 80% hard-coded runtime confidence gate, and the existing registration script's face-URL handling. An export passing numerical parity proves loader compatibility, not recognition accuracy, complete lesson coverage, or deployment-device response time.

No real study training has been performed in this new project's raw dataset. The user will supply the data. Development smoke artifacts are isolated under tmp/ and explicitly marked.

The user additionally requested automatically recorded results, graphs, and tables, plus real-time model testing with automated and manual testing options. Automated testing replays saved held-out data. Live testing supports both automatic sign windows and explicit manual captures, with user-supplied expected labels for scored attempts. All test sessions preserve the loaded model's preprocessing and keep exploratory estimates separate from recorded attempts. Reports include CSV tables, PNG graphs, offline HTML, runtime/configuration provenance, and an experiment history. No webcam recording or raw-pixel retention was requested; preview pixels are not included in saved logs.

The user also requested smoother live testing frames and data augmentation controls. The camera now has an independent capture/preview thread targeting 30 FPS, a latest-frame buffer for detection, and a 16 ms desktop refresh while live. Preview and detection FPS are shown separately; this does not promise device-independent performance or change landmark preprocessing. Training augmentation was already enabled at five copies per training original. It is now explicit in the Train page and CLI (`--augmentation-copies`), with source-level counts saved to `augmentation.csv`. Non-default counts or disabling augmentation require a development run, preserving the agreed study configuration and train-only splitting rules.

The user additionally requested mirrored augmentation to support left-hand use. New training runs default to `--mirroring all`, exposed as **Train > Data augmentation > Left/right mirroring > All signs**. The first augmented copy is an exact opposite-hand reflection, with remaining copies alternating unmirrored and mirrored perturbations. At five copies, each source yields three mirrored and three original-orientation samples including the original. The `approved` policy retains the previous `mirror_ok=1` restriction; `off` disables reflection. Catalog approval flags remain unchanged. Mirroring swaps hand/presence/location slots and negates X, including wrist travel and Family face locations, without reversing time. Augmentation stays after splitting and is recorded per source and per run/fold. Extraction and browser canonical-hand preprocessing remain unchanged; actual left-hand recognition needs retraining/export and held-out left-hand evaluation. The manuscript remains unchanged.

The user requested an option to simulate all current HUDYAT camera/inference settings. **Test models > Simulate HUDYAT** and the `simulate` CLI now build a local browser shell around the actual main-project `useCamera`, `CategoryRecognition`, Recognition lab, TensorFlow.js/MediaPipe runtime, and their supporting modules. This uses the original baseline/enhanced/custom settings and logic rather than a separate Python port. It includes detector IMAGE/VIDEO selection, camera resolution, CPU/GPU and fallback, background/preview processing, preloading, sign/fixed/sliding windows, early confirmation, static Alphabet hold gates, and optional Alphabet candidate releases. The source checkout is read only; settings and source hashes are recorded. Simulation reports use a distinct protocol and separate settings configurations, preserving browser trials without treating them as held-out accuracy. Native Python live testing remains available. Completed random runs can be privately exported for simulation; no model is deployed or registered. Node.js 24 and `setup-simulator.cmd` provide the matched browser packages. Physical performance depends on the selected browser and device; the simulator omits lesson navigation/authentication/grading, and the manuscript remains unchanged.
