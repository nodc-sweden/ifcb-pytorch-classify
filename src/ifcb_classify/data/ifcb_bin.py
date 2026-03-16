from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image


def iter_bin_images(bin_source) -> Iterator[tuple[int, Image.Image]]:
    """Yield (target_number, RGB PIL Image) for each ROI in an IFCB bin.

    bin_source can be a file path (str/Path) to any of the three fileset files
    (.adc, .roi, .hdr), or an already-opened bin object.
    """
    if isinstance(bin_source, (str, Path)):
        from ifcb import open_raw

        fbin = open_raw(str(bin_source))
        with fbin:
            yield from _iter_images_from_bin(fbin)
    else:
        yield from _iter_images_from_bin(bin_source)


def _iter_images_from_bin(fbin) -> Iterator[tuple[int, Image.Image]]:
    for target_num in fbin.images.index:
        arr = fbin.images[target_num]
        img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
        if img.mode != "RGB":
            img = img.convert("RGB")
        yield target_num, img


def iter_directory_bins(dir_path: str | Path) -> Iterator[tuple[str, object]]:
    """Yield (bin_lid, bin_object) for each bin in a DataDirectory."""
    from ifcb import DataDirectory

    dd = DataDirectory(str(dir_path))
    for fbin in dd:
        yield fbin.lid, fbin


def get_bin_lid(bin_path: str | Path) -> str:
    """Extract the LID (sample name) from a bin file path."""
    return Path(bin_path).stem.split(".")[0]
