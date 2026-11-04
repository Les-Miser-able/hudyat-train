"""Read-only comparison with HUDYAT's checked-in lesson seeds and score gates."""
from __future__ import annotations

from pathlib import Path
import re

from .catalog import Catalog, CATEGORY_NAMES, key
from .config import Settings
from .storage import read_json


def audit(settings: Settings, hudyat: Path, emit=print):
    catalog = Catalog(settings.path("signs_file"))
    learning = hudyat / "server" / "prisma" / "data" / "learning"
    rows = []

    def target_key(label):
        normalized = key(label)
        return "purple" if normalized == "violet" else normalized

    for category, name in CATEGORY_NAMES.items():
        files = [learning / (category.replace("_", "-") + ".json")]
        if category == "calendar":
            files.append(learning / "days.json")  # fslGesturesSeeder.ts merges Days into Calendar
        labels = [row["label"] for file in files for row in read_json(file)]
        expected = [sign.label for sign in catalog.signs if sign.category == category]
        trainer_keys = {target_key(label) for label in expected}
        main_keys = {target_key(label) for label in labels}
        row = {"category": category, "name": name, "trainer_classes": len(expected), "hudyat_seed_classes": len(labels),
               "unsupported_hudyat_prompts": [label for label in labels if target_key(label) not in trainer_keys],
               "trainer_signs_without_hudyat_lesson": [label for label in expected if target_key(label) not in main_keys]}
        rows.append(row)
        emit(f"{name}: trainer {len(expected)}, HUDYAT seeds {len(labels)}; unsupported prompts: {', '.join(row['unsupported_hudyat_prompts']) or 'none'}")
    runtime = hudyat / "client" / "src" / "features" / "fsl-recognition"
    source = (runtime / "category-recognition.tsx").read_text(encoding="utf-8")
    gates = sorted({float(value) for value in re.findall(r"score\s*\?\?\s*0\)\s*>=\s*(\.\d+|\d+\.\d+)", source)})
    emit(f"Manuscript confidence: 0.85; hard-coded recognition score gates found: {gates or 'none detected; review manually'}")
    return {"source": "Checked-in seed files and runtime source; this does not inspect or modify the database.",
            "categories": rows, "manuscript_confidence": 0.85, "runtime_score_gates_found": gates,
            "notes": ["Violet/Purple is already an alias in HUDYAT's matchesTarget.",
                      "Static Alphabet excludes J and Z; use the sequence release for the complete alphabet.",
                      "Model metadata carries the 0.85 threshold, but the current runtime has its own score gates.",
                      "The current registerModels.ts does not rewrite faceUrl; Family currently uses the file catalog route."]}
