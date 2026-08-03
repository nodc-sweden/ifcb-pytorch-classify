"""Thin wrapper over ifcbkit for reading ROIs out of raw IFCB bins.

A raw IFCB bin is a three-file set (``.adc``/``.roi``/``.hdr``) sharing one base
name (the *LID*, which encodes the instrument and timestamp). These helpers
isolate the rest of the pipeline from the ``ifcbkit`` package: they yield decoded
RGB images per ROI, iterate the bins in a directory, and extract a bin's LID
from any of its file paths. The ``ifcbkit`` import is deferred into the functions
so importing this module is cheap and doesn't hard-require ifcbkit.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class BinFiles:
    """The resolved ``.adc`` and ``.roi`` paths of a single bin, plus its LID."""

    lid: str
    adc_path: Path
    roi_path: Path


def iter_bin_images(bin_source: str | Path | BinFiles) -> Iterator[tuple[int, Image.Image]]:
    """Yield (target_number, RGB PIL Image) for each ROI in an IFCB bin.

    bin_source can be a file path (str/Path) to any of the three fileset files
    (.adc, .roi, .hdr), or a :class:`BinFiles` from :func:`iter_directory_bins`.
    """
    if isinstance(bin_source, (str, Path)):
        bin_source = _resolve_bin_files(bin_source)
    yield from _iter_images_from_bin(bin_source)


def _resolve_bin_files(bin_path: str | Path) -> BinFiles:
    """Locate a bin's ``.adc``/``.roi`` siblings from any one of its file paths."""
    path = Path(bin_path)
    lid = get_bin_lid(path)
    files = BinFiles(lid=lid, adc_path=path.with_name(f"{lid}.adc"), roi_path=path.with_name(f"{lid}.roi"))

    missing = [p for p in (files.adc_path, files.roi_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete fileset for bin {lid}: missing {', '.join(str(p) for p in missing)}")

    return files


def _iter_images_from_bin(files: BinFiles) -> Iterator[tuple[int, Image.Image]]:
    """Yield ``(target_number, RGB image)`` in ascending target order."""
    from ifcbkit import bin_images

    images = bin_images(files.lid, files.adc_path.read_bytes(), files.roi_path.read_bytes())
    # ifcbkit hands back mode-"L" images; the models expect three channels.
    for target_num in sorted(images):
        img = images[target_num]
        yield target_num, img if img.mode == "RGB" else img.convert("RGB")


def iter_directory_bins(dir_path: str | Path) -> Iterator[tuple[str, BinFiles]]:
    """Yield (bin_lid, BinFiles) for each complete fileset under a directory.

    This is the single authority on which bins a directory run processes. The
    discovery it delegates to only returns filesets that have a ``.hdr``, and it
    skips ``skip``/``beads`` paths — the same filtering pyifcb applied, so the
    exclusions are deliberate. Use :func:`find_headerless_bins` to report the one
    exclusion that is not — an otherwise-complete fileset with no header.
    """
    from ifcbkit import SyncIfcbDataDirectory

    dd = SyncIfcbDataDirectory(str(dir_path))
    for entry in dd.list():
        lid = entry["pid"]
        yield lid, BinFiles(lid=lid, adc_path=Path(entry["adc"]), roi_path=Path(entry["roi"]))


def find_headerless_bins(dir_path: str | Path) -> list[Path]:
    """Return ``.roi`` paths under ``dir_path`` whose fileset has no ``.hdr``.

    Directory discovery needs the header, so these bins are readable —
    :func:`iter_bin_images` opens them from a path fine — but invisible to a
    directory run. Callers surface them so the omission is reported, not silent.

    Deliberately narrow. Discovery also skips ``skip``/``beads`` paths and any
    layout other than flat or ``Dyyyy/Dyyyymmdd/``, matching what pyifcb did;
    reporting those would fire on every bin of a differently-organised archive.
    Only the missing-header case is flagged, since that fileset is otherwise
    complete and its exclusion is an accident of the file layout rather than a
    choice.
    """
    roi_files = Path(dir_path).rglob("*.roi")
    return sorted(p for p in roi_files if p.with_suffix(".adc").is_file() and not p.with_suffix(".hdr").is_file())


def get_bin_lid(bin_path: str | Path) -> str:
    """Extract the LID (sample name) from a bin file path."""
    return Path(bin_path).stem.split(".")[0]
