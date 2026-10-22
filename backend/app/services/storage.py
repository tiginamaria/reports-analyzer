"""Where uploaded PDFs live on disk: under their SHA-256, so the same bytes map to one file."""

import hashlib
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from app.config import get_settings

PDF_MAGIC = b"%PDF-"


class ReportStorage:
    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        d = self._root or get_settings().reports_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path_for(self, digest: str) -> Path:
        return self.root / f"{digest}.pdf"

    def save_temporary(self, source: BinaryIO) -> Path:
        """Uploads land under a random name until their hash is known."""
        tmp = self.root / f"{uuid.uuid4()}.upload"
        with tmp.open("wb") as out:
            shutil.copyfileobj(source, out)
        return tmp

    def store(self, tmp: Path, digest: str) -> Path:
        """Move a temporary upload to its final place."""
        stored = self.path_for(digest)
        tmp.replace(stored)
        return stored

    def remove(self, digest: str) -> None:
        self.path_for(digest).unlink(missing_ok=True)

    @staticmethod
    def is_pdf(path: Path) -> bool:
        with path.open("rb") as f:
            return f.read(len(PDF_MAGIC)) == PDF_MAGIC

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
