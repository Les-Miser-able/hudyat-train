"""The fixed study vocabulary, with explicit static/moving and mirror approvals."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

CATEGORY_NAMES = {
    "alphabet": "Alphabet", "numbers": "Numbers", "greetings": "Greetings",
    "calendar": "Calendar", "shapes": "Shapes", "colors": "Colors", "family": "Family",
    "wh_questions": "WH Questions", "word_concepts": "Word Concepts",
}


def key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


@dataclass(frozen=True)
class Sign:
    category: str
    sign_id: str
    label: str
    hands: int
    mode: str
    mirror_ok: bool


class Catalog:
    def __init__(self, path: Path):
        self.signs: list[Sign] = []
        self.by_folder: dict[tuple[str, str], Sign] = {}
        self.categories = {key(c): c for c in CATEGORY_NAMES}
        self.categories.update({key(n): c for c, n in CATEGORY_NAMES.items()})
        seen = set()
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                sign = Sign(row["category"].strip(), row["sign_id"].strip(), row["sign_label"].strip(),
                            int(row["hands"]), row["mode"].strip(), row.get("mirror_ok", "0").strip() == "1")
                if sign.category not in CATEGORY_NAMES or sign.sign_id in seen:
                    raise ValueError(f"Invalid or duplicate sign: {sign.sign_id}")
                if not re.fullmatch(r"[a-z][a-z0-9_]*", sign.sign_id) or sign.hands not in (1, 2) or sign.mode not in ("static", "dynamic"):
                    raise ValueError(f"Invalid sign configuration: {sign.sign_id}")
                seen.add(sign.sign_id)
                self.signs.append(sign)
                for folder in (sign.sign_id, sign.label):
                    alias = (sign.category, key(folder))
                    if alias in self.by_folder and self.by_folder[alias] != sign:
                        raise ValueError(f"Ambiguous class folder: {folder}")
                    self.by_folder[alias] = sign

    def category(self, folder: str) -> str:
        try:
            return self.categories[key(folder)]
        except KeyError:
            raise ValueError(f"Unknown category folder: {folder}") from None

    def resolve(self, category: str, folder: str) -> Sign:
        category = self.category(category)
        try:
            return self.by_folder[(category, key(folder))]
        except KeyError:
            raise ValueError(f"Unknown class '{folder}' in {category}") from None

    def selected(self, category: str, variant: str) -> list[Sign]:
        return [s for s in self.signs if s.category == category and (variant != "static" or s.mode == "static")]
