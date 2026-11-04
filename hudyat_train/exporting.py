"""Export immutable model releases in HUDYAT's existing binary weight format."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .catalog import CATEGORY_NAMES
from .config import Settings
from .network import architecture, build_model, tensorflow
from .storage import fingerprint, read_json, sha256, write_bytes, write_json


def export_run(settings: Settings, run: Path, output: Path | None = None, allow_development: bool = False, emit=print) -> Path:
    run = run.resolve()
    state = read_json(run / "run.json")
    if state.get("status") != "complete" or state.get("protocol") != "random":
        raise ValueError("Export a completed random-split training run. LOSO runs are evaluation artifacts.")
    if state.get("development") and not allow_development:
        raise ValueError("This is a development run. Use --allow-development to export it for testing.")
    if sha256(run / "best.keras") != state["best_model_sha256"]:
        raise ValueError("The checkpoint changed after evaluation. Train/evaluate it again before exporting.")
    required = {"classes.json", "class-map.json", "preprocessing.json", "training_config.json", "dataset-index.json", "splits.json", "metrics.json"}
    if not required <= state.get("artifact_sha256", {}).keys():
        raise ValueError("Run is missing evaluated artifact checksums. Train it again with the current trainer.")
    for name in required:
        if sha256(run / name) != state["artifact_sha256"][name]:
            raise ValueError(f"{name} changed after evaluation. Train/evaluate again before exporting.")
    classes = read_json(run / "classes.json")
    prep = read_json(run / "preprocessing.json")
    prep["extraction"].pop("model_path", None)
    if "face" in prep:
        prep["face"].pop("model_path", None)
    category, variant, features = state["category"], state["variant"], prep["features"]
    if category not in CATEGORY_NAMES or variant not in ("static", "sequence"):
        raise ValueError("Unknown category/version in this run.")
    output = (output or settings.path("exports_dir") / variant).resolve()
    tf = tensorflow()
    model = tf.keras.models.load_model(run / "best.keras", compile=False)
    expected = build_model(len(classes), features)
    if architecture(model.get_config()) != architecture(expected.get_config()):
        raise ValueError("The checkpoint architecture does not match HUDYAT's Conv1D-BiLSTM loader.")
    chunks, specs, offset = [], [], 0
    # Export per layer, the exact order consumed by category-core.js loadWeights.
    for layer in model.layers:
        for weight in layer.get_weights():
            raw = np.asarray(weight, dtype="<f4").tobytes(order="C")
            specs.append({"shape": list(weight.shape), "dtype": "float32", "offset": offset, "bytes": len(raw)})
            chunks.append(raw)
            offset += len(raw)
    binary = b"".join(chunks)
    weight_hash = hashlib.sha256(binary).hexdigest()
    version = fingerprint({"weights": weight_hash, "preprocessing": prep, "classes": classes,
                           "category": category, "variant": variant, "development": state["development"]})[:16]
    assets = [("hand", prep["extraction"])]
    if features == 141:
        assets.append(("face", prep["face"]))
    detector_data = {}
    for kind, spec in assets:
        configured = settings.path(kind + "_model")
        if not configured.is_file() or sha256(configured) != spec["model_sha256"]:
            raise ValueError(f"The {kind} detector differs from extraction. Restore the matching .task file before exporting.")
        detector_data[kind] = (configured.read_bytes(), spec["model_sha256"])
    meta = {"format": "fsl-conv1d-bilstm-v1", "category": category, "variant": variant, "version": version,
            "inputShape": [32, features], "classes": classes, "preprocessing": prep, "weights": specs,
            "weightBytes": len(binary), "weightSha256": weight_hash, "development": state["development"]}
    rng = np.random.default_rng(42)
    probes = rng.normal(0, 0.25, (3, 32, features)).astype(np.float32)
    probes[:, :, 126:128] = 1
    probes[0, :, :63], probes[0, :, 126] = 0, 0
    probes[2, 12:20, 63:126], probes[2, 12:20, 127] = 0, 0
    if features == 141:
        probes[:, :, 140] = 1
        probes[0, :, 128:134], probes[2, 12:20, 134:140] = 0, 0
    parity = {"inputs": probes.tolist(), "expected": model(probes, training=False).numpy().tolist(),
              "purpose": "synthetic numerical parity only; contains no source recordings"}
    base = f"./models/{category}/{version}"
    dest = output / "models" / category / version
    catalog_path = output / "catalog.json"
    catalog = read_json(catalog_path) if catalog_path.exists() else {"categories": []}
    entries = {entry["id"]: entry for entry in catalog["categories"]}
    previous = entries.get(category, {})
    if previous.get("variant", variant) != variant:
        raise ValueError("Keep static and sequence releases in separate export directories.")
    dest.mkdir(parents=True, exist_ok=True)
    # A versioned path is immutable; a conflicting file is never overwritten.
    import json
    payloads = {"model.json": (json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(),
                "weights.bin": binary, "parity.json": (json.dumps(parity, allow_nan=False) + "\n").encode()}
    for name, payload in payloads.items():
        path = dest / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Existing release has different content: {path}")
    for name, payload in payloads.items():
        write_bytes(dest / name, payload)
    release = {"id": category, "name": CATEGORY_NAMES[category], "available": True, "variant": variant,
               "development": state["development"], "version": version, "modelUrl": base + "/model.json",
               "weightsUrl": base + "/weights.bin", "modelSha256": sha256(dest / "model.json"),
               "weightsSha256": weight_hash, "bytes": len(binary) + (dest / "model.json").stat().st_size, "classes": len(classes)}
    for kind, (data, digest) in detector_data.items():
        target = output / "shared" / (digest + ".task")
        if target.exists() and sha256(target) != digest:
            raise ValueError(f"Existing shared detector has invalid content: {target}")
        write_bytes(target, data)
        release.update({kind + "Url": f"./shared/{digest}.task", kind + "Sha256": digest, kind + "Bytes": len(data)})
    entries[category] = release
    catalog["categories"] = sorted(entries.values(), key=lambda entry: entry["name"])
    write_json(catalog_path, catalog)  # publish the catalog only after every new artifact is ready
    emit(f"Exported {category}/{variant}: {len(classes)} classes, version {version}. Bundle: {output}")
    return output
