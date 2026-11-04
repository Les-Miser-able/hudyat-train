"""Reproducible study and development runs, with source-level split records."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import importlib.metadata
from pathlib import Path
import uuid

import numpy as np

from .augmentation import augment
from .catalog import Catalog
from .config import Settings
from .dataset import discover
from .evaluation import classification_report, plot_reports
from .extraction import source_signature
from .experiments import Registry, append_event, runtime_details, write_csv
from .features import FEATURE_VERSION
from .jobs import Job
from .network import build_model, tensorflow
from .splitting import loso_splits, random_split
from .storage import fingerprint, read_json, sha256, within, write_json


@dataclass(frozen=True)
class TrainOptions:
    category: str
    variant: str = "sequence"
    protocol: str = "random"
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 20
    augmentation_copies: int = 5
    seed: int = 42
    development: bool = False


def load_dataset(settings: Settings, options: TrainOptions):
    dest = settings.path("extracted_dir") / options.variant / options.category
    index = dest / "index.json"
    if not index.is_file():
        raise ValueError(f"Extract {options.category}/{options.variant} first.")
    manifest = read_json(index)
    if manifest["category"] != options.category or manifest["variant"] != options.variant:
        raise ValueError("Extraction index belongs to a different category/version.")
    if manifest["feature_version"] != FEATURE_VERSION or manifest["catalog_sha256"] != sha256(settings.path("signs_file")):
        raise ValueError("Vocabulary or preprocessing changed. Run extraction again.")
    for key in ("detection_confidence", "min_hand_rate", "min_both_hands_rate", "swap_hands"):
        if manifest["settings"][key] != settings.get(key):
            raise ValueError(f"Extraction setting '{key}' changed. Run extraction again.")
    sources, issues, _ = discover(settings, options.category, options.variant)
    if issues or source_signature(sources) != manifest["source_signature"]:
        raise ValueError("Raw files, metadata, or labels changed since extraction. Scan and extract again.")
    accepted = [it for it in manifest["items"] if it["status"] == "accepted"]
    by_sign = {s.sign_id: s for s in Catalog(settings.path("signs_file")).selected(options.category, options.variant)}
    missing = set(by_sign) - {it["sign_id"] for it in accepted}
    expected_signers = set(settings.get("expected_signers"))
    actual_signers = {it["signer"] for it in accepted}
    changed = options.epochs != 150 or options.batch_size != 32 or options.learning_rate != 0.001 or options.patience != 20 or options.augmentation_copies != 5
    changed |= expected_signers != {f"S{i}" for i in range(1, 8)} or settings.get("takes_per_signer") != 15 or settings.get("min_hand_rate") != 0.8
    coverage = Counter((it["sign_id"], it["signer"]) for it in accepted)
    shortages = [(sign, signer) for sign in by_sign for signer in expected_signers if coverage[sign, signer] < settings.get("takes_per_signer")]
    if not options.development and (missing or expected_signers != actual_signers or shortages or changed):
        detail = f"missing classes: {', '.join(sorted(missing)) or 'none'}; signers: {', '.join(sorted(actual_signers)) or 'none'}; incomplete sign/signer cells: {len(shortages)}"
        raise ValueError("The study dataset/settings are incomplete (" + detail + "). Finish data collection or explicitly choose a development run.")
    signs = [s for s in by_sign.values() if any(it["sign_id"] == s.sign_id for it in accepted)]
    if len(signs) < 2:
        raise ValueError("Training needs at least 2 classes with accepted sources.")
    classes = [s.label for s in signs]
    label_index = {s.sign_id: i for i, s in enumerate(signs)}
    samples = []
    for item in accepted:
        path = within(dest, item["sample"])
        if sha256(path) != item["sample_sha256"]:
            raise ValueError(f"Extracted sample changed: {item['sample']}. Extract again.")
        sample = np.load(path, allow_pickle=False)
        if list(sample.shape) != manifest["input_shape"] or sample.dtype != np.float32 or not np.isfinite(sample).all():
            raise ValueError(f"Invalid landmark sample: {item['sample']}")
        samples.append(sample)
    return manifest, accepted, np.asarray(samples), np.asarray([label_index[it["sign_id"]] for it in accepted]), signs, classes


def preprocessing(manifest, options):
    detectors = dict(manifest["detectors"])
    face = detectors.pop("face", None)
    config = {
        "frames": 32, "features": manifest["input_shape"][1], "normalization": manifest["normalization"],
        "feature_version": FEATURE_VERSION,
        "layout": ["left_xyz_landmarks_0_to_20", "right_xyz_landmarks_0_to_20", "left_present", "right_present"],
        "extraction": detectors, "variant": options.variant, "confidence_threshold": 0.85,
        "sign_window": {"trim_margin_s": 0.1, "trim_min_frames": 8, "minimum_hand_detection": manifest["settings"]["min_hand_rate"]},
        "interpolation": "HUDYAT category-core.resample" if manifest["input_shape"][1] == 128 and options.variant == "sequence" else "HUDYAT sign-capture.resampleSign",
        "handedness": {"canonical": options.category == "family" or (options.variant == "static" and options.category != "alphabet"),
                       "mirrored_augmentation": "only signs with mirror_ok=1 in signs.csv"},
        "static_method": "one stable pose per source, repeated to 32 frames" if options.variant == "static" else None,
        "augmentation": {"split": "train", "enabled": options.augmentation_copies > 0,
                         "input": "normalized landmarks", "copies_per_original": options.augmentation_copies, "noise_std": 0.01,
                         "scale_range": [0.9, 1.1], "rotation_degrees": 15, "speed_range": [0.8, 1.2], "frame_removal_probability": 0.08},
    }
    if face:
        config["face"] = face
        config["layout"] += ["left_location_wrist_thumb_tip_index_tip_xy", "right_location_wrist_thumb_tip_index_tip_xy", "face_present"]
    return config


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit(run, options, manifest, items, samples, labels, classes, split, job, extra=None, registry=None):
    tf = tensorflow()
    if not options.development and tf.__version__ != "2.19.1":
        raise ValueError(f"Study runs require the manuscript's TensorFlow 2.19.1; this interpreter has {tf.__version__}. Run setup.ps1 or use an explicitly marked development run.")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(options.seed)
    run.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "category": options.category, "variant": options.variant, "protocol": options.protocol,
             "development": options.development, "started_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
    kind = "training_fold" if "test_signer" in state else "training"
    state["dataset_id"] = fingerprint({name: [items[i]["digest"] for i in indices] for name, indices in split.items()})[:16]
    state["parameters"] = asdict(options)
    state["runtime"] = runtime_details()
    write_json(run / "run.json", state)
    write_json(run / "classes.json", classes)
    write_json(run / "class-map.json", [{"index": i, "sign_id": next(it["sign_id"] for it in items if it["label"] == label), "label": label}
                                       for i, label in enumerate(classes)])
    write_json(run / "training_config.json", asdict(options))
    write_json(run / "dataset-index.json", manifest)
    write_json(run / "preprocessing.json", preprocessing(manifest, options))
    write_json(run / "splits.json", {name: [items[i] for i in indices] for name, indices in split.items()})
    append_event(run / "events.jsonl", {"type": "started", **state, "parameters": asdict(options)})
    if registry:
        registry.save(run, kind, state, parameters=asdict(options))
    try:
        rng = np.random.default_rng(options.seed)
        x_train, y_train, augmentation_rows = [], [], []
        for i in split["train"]:
            job.check()
            x_train.append(samples[i])
            y_train.append(labels[i])
            for _ in range(options.augmentation_copies):
                job.check()
                x_train.append(augment(samples[i], rng, items[i]["mirror_ok"], options.variant == "static"))
                y_train.append(labels[i])
            augmentation_rows.append({"source": items[i]["relative"], "signer": items[i]["signer"], "label": items[i]["label"],
                                      "original_samples": 1, "augmented_copies": options.augmentation_copies,
                                      "total_training_samples": 1 + options.augmentation_copies, "mirror_allowed": bool(items[i]["mirror_ok"])})
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        state.update(training_sources=len(split["train"]), augmented_training_samples=len(x_train),
                     generated_training_samples=len(x_train) - len(split["train"]),
                     validation_sources=len(split["validation"]), test_sources=len(split["test"]))
        write_csv(run / "augmentation.csv", augmentation_rows)
        append_event(run / "events.jsonl", {"type": "augmentation", "copies_per_original": options.augmentation_copies,
                                           "original_sources": len(split["train"]), "generated_samples": state["generated_training_samples"],
                                           "total_training_samples": len(x_train), "split": "train"})
        job.emit(f"Data augmentation: {len(split['train'])} originals + {state['generated_training_samples']} generated = {len(x_train)} training samples ({options.augmentation_copies} copies per original). Validation/test are unmodified. Details: {run / 'augmentation.csv'}")
        write_json(run / "run.json", state)
        original_counts = Counter(int(labels[i]) for i in split["train"])
        weights = {i: len(split["train"]) / (len(classes) * original_counts[i]) for i in range(len(classes))}
        write_json(run / "class-weights.json", weights)

        def dataset(x, y, shuffle=False):
            data = tf.data.Dataset.from_tensor_slices((x, y))
            if shuffle:
                data = data.shuffle(len(x), seed=options.seed)
            data = data.batch(options.batch_size)
            config = tf.data.Options()
            config.threading.private_threadpool_size = 1
            config.threading.max_intra_op_parallelism = 1
            return data.with_options(config)

        train_data = dataset(x_train, y_train, True)
        val_data = dataset(samples[split["validation"]], labels[split["validation"]])
        model = build_model(len(classes), samples.shape[2])
        model.compile(optimizer=tf.keras.optimizers.Adam(options.learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        epoch_rows = []

        class Progress(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                job.check()

            def on_epoch_end(self, epoch, logs=None):
                job.check()
                row = {"epoch": epoch + 1, **{k: float(v) for k, v in (logs or {}).items()}}
                epoch_rows.append(row)
                append_event(run / "events.jsonl", {"type": "epoch", **row})
                write_csv(run / "history.csv", epoch_rows)
                write_json(run / "history.json", {k: [r[k] for r in epoch_rows] for k in row if k != "epoch"})
                if registry:
                    registry.save(run, kind, {**state, "last_epoch": epoch + 1}, parameters=asdict(options))
                job.emit(f"Epoch {epoch + 1}/{options.epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in (logs or {}).items()))

        job.emit(f"Training {len(classes)} classes: {len(split['train'])} original training clips -> {len(x_train)} samples; validation {len(split['validation'])}; test {len(split['test'])}.")
        history = model.fit(train_data, validation_data=val_data, epochs=options.epochs, verbose=0, class_weight=weights, shuffle=False,
                            callbacks=[Progress(), tf.keras.callbacks.ModelCheckpoint(str(run / "best.keras"), monitor="val_loss", save_best_only=True),
                                       tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=options.patience, restore_best_weights=True),
                                       tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)])
        job.check()
        # Always evaluate the saved best validation checkpoint, including when max epochs was reached.
        model = tf.keras.models.load_model(run / "best.keras", compile=False)
        test_indices = split["test"]
        probabilities = np.concatenate([model(samples[test_indices[j:j + options.batch_size]], training=False).numpy()
                                        for j in range(0, len(test_indices), options.batch_size)])
        guesses, truth = probabilities.argmax(axis=1), labels[test_indices]
        report = classification_report(truth, guesses, classes)
        confidence = probabilities.max(axis=1)
        accepted = confidence >= 0.85
        report["at_85_confidence"] = {"acceptance_rate": float(accepted.mean()),
                                      "accepted_accuracy": float((guesses[accepted] == truth[accepted]).mean()) if accepted.any() else None}
        write_json(run / "metrics.json", report)
        write_json(run / "history.json", {k: [float(v) for v in values] for k, values in history.history.items()})
        _csv(run / "history.csv", [{"epoch": i + 1, **{k: float(v[i]) for k, v in history.history.items()}} for i in range(len(history.epoch))])
        _csv(run / "per-class.csv", report["per_class"])
        _csv(run / "predictions.csv", [{"source": items[i]["relative"], "signer": items[i]["signer"],
                                      "actual": classes[int(truth[j])], "predicted": classes[int(guesses[j])],
                                      "confidence": float(confidence[j])} for j, i in enumerate(test_indices)])
        plot_reports(run, history.history, report, classes)
        from .reporting import result_report
        result_report(run, {**state, "status": "complete", "kind": kind, "epochs_completed": len(history.epoch)}, report)
        job.check()
        state.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                     best_model_sha256=sha256(run / "best.keras"), tensorflow=tf.__version__, numpy=np.__version__,
                     epochs_completed=len(history.epoch), test_samples=len(test_indices))
        state["artifact_sha256"] = {name: sha256(run / name) for name in
                                    ("classes.json", "class-map.json", "preprocessing.json", "training_config.json", "dataset-index.json", "splits.json", "metrics.json")}
        write_json(run / "run.json", state)
        append_event(run / "events.jsonl", {"type": "completed", "metrics": report})
        if registry:
            registry.save(run, kind, state, report, asdict(options))
        job.emit(f"Complete: accuracy {report['accuracy']:.1%}, macro F1 {report['macro_f1']:.1%}. Run: {run}")
        return report, truth.tolist(), guesses.tolist()
    except BaseException as error:
        state.update(status="interrupted" if error.__class__.__name__ in ("Cancelled", "KeyboardInterrupt") else "failed", error=str(error))
        write_json(run / "run.json", state)
        append_event(run / "events.jsonl", {"type": state["status"], "error": str(error)})
        if registry:
            registry.save(run, kind, state, parameters=asdict(options))
        raise


def train(settings: Settings, options: TrainOptions, job: Job | None = None) -> Path:
    job = job or Job()
    category = Catalog(settings.path("signs_file")).category(options.category)
    if category != options.category or options.variant not in ("static", "sequence") or options.protocol not in ("random", "loso"):
        raise ValueError("Use a canonical category ID, a static/sequence version, and random/loso protocol.")
    if not 1 <= options.epochs <= 150 or options.batch_size < 1 or options.learning_rate <= 0 or options.augmentation_copies < 0:
        raise ValueError("Invalid training settings; epochs must be 1-150.")
    job.emit("Checking source files, sample checksums, and study coverage...")
    manifest, items, samples, labels, signs, classes = load_dataset(settings, options)
    # Validate all folds before creating a run or importing TensorFlow.
    folds = [{"split": random_split(items, options.seed)}] if options.protocol == "random" else loso_splits(items)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run = settings.path("runs_dir") / category / options.variant / f"{stamp}-{options.protocol}"
    registry = Registry(settings.root)
    if options.protocol == "random":
        _fit(run, options, manifest, items, samples, labels, classes, folds[0]["split"], job, registry=registry)
    else:
        run.mkdir(parents=True, exist_ok=False)
        state = {"status": "running", "category": category, "variant": options.variant, "protocol": "loso", "development": options.development,
                 "started_at": datetime.now(timezone.utc).isoformat(), "parameters": asdict(options),
                 "dataset_id": fingerprint([item["digest"] for item in items])[:16]}
        write_json(run / "run.json", state)
        registry.save(run, "training", state, parameters=asdict(options))
        append_event(run / "events.jsonl", {"type": "started", **state})
        reports, truths, predictions = [], [], []
        try:
            for fold in folds:
                job.check()
                job.emit(f"LOSO: test {fold['test_signer']}; validation {fold['validation_signer']}; remaining signers train.")
                report, truth, guess = _fit(run / f"test-{fold['test_signer']}", options, manifest, items, samples, labels, classes,
                                            fold["split"], job, {k: v for k, v in fold.items() if k != "split"}, registry)
                reports.append({"test_signer": fold["test_signer"], "validation_signer": fold["validation_signer"], **report})
                truths.extend(truth)
                predictions.extend(guess)
            pooled = classification_report(truths, predictions, classes)
            summary = {"folds": reports, "pooled": pooled, "statistics": {
                metric: {"mean": float(np.mean([r[metric] for r in reports])), "std": float(np.std([r[metric] for r in reports], ddof=1))}
                for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}, "std_definition": "sample standard deviation across held-out signers"}
            summary["classification_benchmark_pass"] = summary["statistics"]["accuracy"]["mean"] >= 0.85 and summary["statistics"]["macro_f1"]["mean"] >= 0.85 and not pooled["benchmark"]["signs_below_70_recall"]
            write_json(run / "loso-summary.json", summary)
            from .reporting import loso_report
            loso_report(run, summary, {**state, "status": "complete", "kind": "training"})
            state["status"] = "complete"
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            write_json(run / "run.json", state)
            registry.save(run, "training", state, {**pooled, **{k: v["mean"] for k, v in summary["statistics"].items()}}, asdict(options))
            append_event(run / "events.jsonl", {"type": "completed", "statistics": summary["statistics"]})
            job.emit(f"LOSO complete. Summary: {run / 'loso-summary.json'}")
        except BaseException as error:
            state.update(status="interrupted" if error.__class__.__name__ in ("Cancelled", "KeyboardInterrupt") else "failed", error=str(error))
            write_json(run / "run.json", state)
            registry.save(run, "training", state, parameters=asdict(options))
            append_event(run / "events.jsonl", {"type": state["status"], "error": str(error)})
            raise
    return run
