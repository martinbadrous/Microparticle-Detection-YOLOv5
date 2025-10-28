"""General project utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTENSIONS: Sequence[str] = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def list_images(folder: Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[str]:
    """Return a sorted list of images inside ``folder``.

    Parameters
    ----------
    folder:
        Directory to search.
    extensions:
        Glob patterns to include.  Defaults to standard image extensions.
    """

    folder = Path(folder)
    paths = []
    for ext in extensions:
        paths.extend(folder.glob(ext))
    return sorted(str(p) for p in paths)


__all__ = ["IMAGE_EXTENSIONS", "list_images"]

