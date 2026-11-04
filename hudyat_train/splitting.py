"""Split independent source recordings before generating any augmented samples."""
from __future__ import annotations

from collections import defaultdict
import random


def random_split(items: list[dict], seed: int = 42) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for i, item in enumerate(items):
        groups[item["sign_id"]].append(i)
    result = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for label, indices in sorted(groups.items()):
        if len(indices) < 3:
            raise ValueError(f"{label} needs at least 3 independent sources to populate train, validation, and test.")
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * 0.15 + 0.5))
        n_val = max(1, int(len(indices) * 0.15 + 0.5))
        result["test"].extend(indices[:n_test])
        result["validation"].extend(indices[n_test:n_test + n_val])
        result["train"].extend(indices[n_test + n_val:])
    validate_split(items, result)
    return result


def loso_splits(items: list[dict]) -> list[dict]:
    signers = sorted({item["signer"] for item in items})
    if len(signers) < 3:
        raise ValueError("LOSO needs at least 3 signers: separate training, validation, and test signers.")
    folds = []
    for i, signer in enumerate(signers):
        val_signer = signers[(i + 1) % len(signers)]
        split = {"train": [], "validation": [], "test": []}
        for j, item in enumerate(items):
            name = "test" if item["signer"] == signer else "validation" if item["signer"] == val_signer else "train"
            split[name].append(j)
        validate_split(items, split)
        folds.append({"test_signer": signer, "validation_signer": val_signer, "split": split})
    return folds


def validate_split(items, split):
    if any(not split[name] for name in ("train", "validation", "test")):
        raise ValueError("Every split must contain independent source recordings.")
    digests = [{items[i]["digest"] for i in split[name]} for name in ("train", "validation", "test")]
    if any(digests[i] & digests[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("The same raw recording appears in more than one split.")
    labels = {item["sign_id"] for item in items}
    for name in split:
        missing = labels - {items[i]["sign_id"] for i in split[name]}
        if missing:
            raise ValueError(f"{name} split has no recordings for: {', '.join(sorted(missing))}")
