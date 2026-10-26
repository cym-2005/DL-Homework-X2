"""Read the four local MNIST IDX files used in Homework X2.

By default, the files are read from ``mnist_data/`` next to this module.
The classifier, loss, gradients, and training stay in the notebook.
"""

from pathlib import Path
import gzip
import struct

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "mnist_data"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_images(path):
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        pixels = np.frombuffer(stream.read(), dtype=np.uint8)
    if magic != 2051 or pixels.size != count * rows * columns:
        raise ValueError(f"Invalid MNIST image file: {path}")
    return pixels.reshape(count, rows, columns)


def _read_labels(path):
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        labels = np.frombuffer(stream.read(), dtype=np.uint8)
    if magic != 2049 or labels.size != count:
        raise ValueError(f"Invalid MNIST label file: {path}")
    return labels.astype(np.int64)


def load_mnist(data_dir=None):
    """Return training images/labels and test images/labels in that order."""
    data_dir = DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)
    paths = {name: data_dir / filename for name, filename in FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local MNIST file(s): " + ", ".join(missing))
    return (
        _read_images(paths["train_images"]),
        _read_labels(paths["train_labels"]),
        _read_images(paths["test_images"]),
        _read_labels(paths["test_labels"]),
    )
