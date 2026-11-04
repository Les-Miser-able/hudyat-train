"""Discover user-managed raw files without moving, rewriting, or deleting them."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import re

from .catalog import Catalog, CATEGORY_NAMES
from .config import Settings
from .storage import sha256

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class Source:
    path: Path
    relative: str
    category: str
    sign_id: str
    label: str
    signer: str
    kind: str
    digest: str
    hands: int
    mode: str
    mirror_ok: bool

    def record(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "path"}


def _metadata(raw: Path) -> dict[str, dict]:
    path = raw.parent / "metadata.csv"
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            relative = row.get("file_path", "").replace("\\", "/")
            if "raw/" in relative:
                relative = relative.split("raw/", 1)[1]
            if relative in rows:
                raise ValueError(f"Duplicate metadata row: {relative}")
            rows[relative] = row
    return rows


def discover(settings: Settings, category: str | None = None, variant: str = "sequence", emit=print):
    catalog, raw = Catalog(settings.path("signs_file")), settings.path("raw_dir")
    if category:
        category = catalog.category(category)
    metadata = _metadata(raw)
    sources, issues, excluded = [], [], []
    content = {}
    for path in sorted(raw.rglob("*")) if raw.exists() else []:
        if not path.is_file() or path.suffix.lower() not in IMAGES | VIDEOS:
            continue
        relative = path.relative_to(raw)
        parts = relative.parts
        if any(part.startswith(".") for part in parts) or ".tmp." in path.name.lower():
            continue
        name = relative.as_posix()
        try:
            if not path.resolve().is_relative_to(raw.resolve()):
                raise ValueError("Symlink points outside the raw dataset")
            if len(parts) < 4:
                raise ValueError("Expected <category>/<class>/<signer>/<file>")
            cat = catalog.category(parts[0])
            if category and cat != category:
                continue
            sign = catalog.resolve(cat, parts[1])
            signer = parts[2].upper()
            if not re.fullmatch(r"S[1-9][0-9]*", signer):
                raise ValueError("Signer folder must be S1, S2, S3, ...; public data is outside the study dataset")
            row = metadata.get(name, {})
            if row and row.get("status", "kept").strip() != "kept":
                excluded.append({"source": name, "reason": "not kept in metadata.csv"})
                continue
            if row.get("orientation", "camera_raw_unmirrored").strip() != "camera_raw_unmirrored":
                raise ValueError("Only unmirrored source media is supported")
            if variant == "static" and sign.mode == "dynamic":
                excluded.append({"source": name, "reason": "dynamic class excluded from static version"})
                continue
            if variant == "sequence" and sign.mode == "dynamic" and path.suffix.lower() in IMAGES:
                excluded.append({"source": name, "reason": "a moving sign needs a video, not a still image"})
                continue
            digest = sha256(path)
            identity = (cat, sign.sign_id, signer)
            if digest in content:
                old_identity, old_path = content[digest]
                if identity != old_identity:
                    raise ValueError(f"Identical media has different classes/signers: {old_path}")
                excluded.append({"source": name, "reason": f"duplicate of {old_path}"})
                continue
            content[digest] = (identity, name)
            sources.append(Source(path, name, cat, sign.sign_id, sign.label, signer,
                                  "image" if path.suffix.lower() in IMAGES else "video", digest,
                                  sign.hands, sign.mode, sign.mirror_ok))
        except (ValueError, OSError) as error:
            issues.append({"source": name, "reason": str(error)})
    return sources, issues, excluded


def scan(settings: Settings, category=None, variant="sequence", emit=print) -> dict:
    sources, issues, excluded = discover(settings, category, variant, emit)
    catalog = Catalog(settings.path("signs_file"))
    counts = Counter((s.category, s.sign_id, s.signer) for s in sources)
    categories = [catalog.category(category)] if category else list(CATEGORY_NAMES)
    coverage = []
    for cat in categories:
        signs = catalog.selected(cat, variant)
        coverage.append({
            "category": cat, "name": CATEGORY_NAMES[cat], "classes_expected": len(signs),
            "classes_present": len({s.sign_id for s in sources if s.category == cat}),
            "sources": sum(s.category == cat for s in sources),
            "signers": sorted({s.signer for s in sources if s.category == cat}),
            "signs": [{"sign_id": sign.sign_id, "label": sign.label, "mode": sign.mode,
                       "counts": {signer: counts[cat, sign.sign_id, signer] for signer in settings.get("expected_signers")}}
                      for sign in signs],
        })
    report = {"variant": variant, "raw_dir": str(settings.path("raw_dir")), "sources": len(sources),
              "issues": issues, "excluded": excluded, "coverage": coverage}
    for row in coverage:
        emit(f"{row['name']}: {row['sources']} files, {row['classes_present']}/{row['classes_expected']} classes, signers {', '.join(row['signers']) or 'none'}")
    for issue in issues:
        emit(f"ERROR {issue['source']}: {issue['reason']}")
    return report
