from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class Cancelled(Exception):
    pass


@dataclass
class Job:
    cancel_file: Path | None = None

    def check(self):
        if self.cancel_file and self.cancel_file.exists():
            raise Cancelled("Stopped by the user.")

    def emit(self, message: str):
        print(message, flush=True)
