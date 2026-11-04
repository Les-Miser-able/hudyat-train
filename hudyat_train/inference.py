"""Load an evaluated checkpoint or exported HUDYAT release for local tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .catalog import CATEGORY_NAMES
from .storage import read_json, sha256


@dataclass
class ModelInfo:
    directory: Path
    classes: list[str]
    preprocessing: dict
    category: str
    variant: str
    development: bool
    checkpoint_sha256: str
    state: dict
    exported: bool = False

    @property
    def features(self):
        return int(self.preprocessing["features"])


def describe_model(directory: Path) -> ModelInfo:
    directory = directory.resolve()
    if directory.is_file():
        directory = directory.parent
    if (directory / "run.json").is_file():
        state = read_json(directory / "run.json")
        if state.get("status") != "complete":
            raise ValueError("Choose a completed, evaluated training run.")
        if not (directory / "best.keras").is_file():
            raise ValueError("This LOSO parent has several models. Choose a test-S* fold for live inference, or use automated testing on the parent.")
        if sha256(directory / "best.keras") != state["best_model_sha256"]:
            raise ValueError("The checkpoint changed after evaluation.")
        for name, digest in state.get("artifact_sha256", {}).items():
            if Path(name).name != name or sha256(directory / name) != digest:
                raise ValueError(f"Evaluated artifact changed: {name}")
        classes = read_json(directory / "classes.json")
        prep = read_json(directory / "preprocessing.json")
        info = ModelInfo(directory, classes, prep, state["category"], state["variant"], bool(state.get("development")), state["best_model_sha256"], state)
    elif (directory / "model.json").is_file():
        metadata = read_json(directory / "model.json")
        if metadata.get("format") != "fsl-conv1d-bilstm-v1":
            raise ValueError("This export does not use HUDYAT's supported model format.")
        if sha256(directory / "weights.bin") != metadata["weightSha256"]:
            raise ValueError("Exported weight checksum failed.")
        prep = metadata["preprocessing"]
        prep.setdefault("features", metadata["inputShape"][1])
        variant = metadata.get("variant", prep.get("variant", "sequence"))
        info = ModelInfo(directory, metadata["classes"], prep, metadata["category"], variant,
                         bool(metadata.get("development", True)), metadata["weightSha256"], metadata, True)
    else:
        raise ValueError("Select a run folder containing run.json and best.keras, or an export version containing model.json and weights.bin.")
    if info.category not in CATEGORY_NAMES or info.variant not in ("static", "sequence"):
        raise ValueError("Unsupported model category or version.")
    normalization = info.preprocessing.get("normalization")
    if normalization not in ("hand-relative-v1", "hand-relative-sequence-v1", "hand-face-location-v1"):
        raise ValueError("Unsupported landmark normalization.")
    if info.features != (141 if normalization == "hand-face-location-v1" else 128):
        raise ValueError("Feature count does not match model preprocessing.")
    if len(info.classes) < 2 or len(info.classes) != len(set(info.classes)):
        raise ValueError("Invalid class ordering.")
    return info


class Predictor:
    def __init__(self, info: ModelInfo):
        import numpy as np
        from .network import build_model, tensorflow
        self.info = info
        tf = tensorflow()
        if not info.exported:
            self.model = tf.keras.models.load_model(info.directory / "best.keras", compile=False)
        else:
            self.model = build_model(len(info.classes), info.features)
            metadata = info.state
            binary = (info.directory / "weights.bin").read_bytes()
            weights = []
            for spec in metadata["weights"]:
                count = int(np.prod(spec["shape"]))
                if spec["dtype"] != "float32" or spec["offset"] % 4 or spec["bytes"] != count * 4 or spec["offset"] < 0 or spec["offset"] + spec["bytes"] > len(binary):
                    raise ValueError("Invalid exported weight layout.")
                weights.append(np.frombuffer(binary, dtype="<f4", count=count, offset=spec["offset"]).copy().reshape(spec["shape"]))
            offset = 0
            for layer in self.model.layers:
                count = len(layer.weights)
                layer.set_weights(weights[offset:offset + count])
                offset += count
            if offset != len(weights):
                raise ValueError("Unexpected exported weight count.")
        if list(self.model.input_shape[1:]) != [32, info.features] or self.model.output_shape[-1] != len(info.classes):
            raise ValueError("Model shape and class metadata disagree.")
        # Warm-up is deliberately excluded from attempt timing and accuracy.
        self._infer = tf.function(lambda value: self.model(value, training=False),
                                  input_signature=[tf.TensorSpec((1, 32, info.features), tf.float32)], autograph=False)
        self._infer(np.zeros((1, 32, info.features), np.float32)).numpy()

    def predict(self, sample):
        import numpy as np
        sample = np.asarray(sample, dtype=np.float32)
        if sample.shape != (32, self.info.features) or not np.isfinite(sample).all():
            raise ValueError("Invalid model input shape or coordinates.")
        started = perf_counter()
        probabilities = self._infer(sample[None]).numpy()[0]
        elapsed = (perf_counter() - started) * 1000
        if probabilities.shape != (len(self.info.classes),) or not np.isfinite(probabilities).all():
            raise ValueError("The model returned invalid probabilities.")
        order = np.argsort(-probabilities)
        return {"predicted": self.info.classes[int(order[0])], "confidence": float(probabilities[order[0]]),
                "margin": float(probabilities[order[0]] - probabilities[order[1]]), "model_ms": elapsed,
                "top3": [{"label": self.info.classes[int(i)], "probability": float(probabilities[i])} for i in order[:3]]}


def detector_paths(settings, info: ModelInfo):
    paths = {}
    for kind, spec in [("hand", info.preprocessing["extraction"]), ("face", info.preprocessing.get("face"))]:
        if spec is None:
            continue
        digest = spec["model_sha256"]
        candidates = [settings.path(kind + "_model")]
        candidates += [base / "shared" / (digest + ".task") for base in [info.directory, *info.directory.parents]]
        found = next((p for p in candidates if p.is_file() and sha256(p) == digest), None)
        if found is None:
            raise ValueError(f"The {kind} detector used for this model is missing. Restore the matching .task asset.")
        paths[kind] = found
    return paths
