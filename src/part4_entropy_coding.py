"""
Part 4 - Entropy Coding
- Serialise encoded frames into bytes (pickle + numpy)
- Apply lzma lossless compression
- Write to .bin output file
- Decoder reads and decompresses the file
"""

import io
import lzma
import pickle
import struct
import numpy as np
from pathlib import Path


# ── Magic bytes to identify our file format ──────────────────────────────────
MAGIC = b"MPEG4ENC"
VERSION = 1


def serialise_frame(encoded_frame: dict) -> bytes:
    """
    Serialise a single encoded frame (I or P) to bytes using pickle.
    numpy arrays are stored efficiently with their native serialisation.
    """
    buf = io.BytesIO()
    pickle.dump(encoded_frame, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


def deserialise_frame(data: bytes) -> dict:
    """Deserialise a single encoded frame from bytes."""
    buf = io.BytesIO(data)
    return pickle.load(buf)


def write_bin(encoded_frames: list[dict],
              output_path: str | Path,
              metadata: dict | None = None) -> int:
    """
    Compress and write all encoded frames to a .bin file.

    File layout:
      [8B magic][1B version][4B n_frames][8B metadata_len][metadata_bytes]
      For each frame: [4B frame_len][frame_bytes]
    All data is lzma-compressed after the header.

    Returns the total file size in bytes.
    """
    output_path = Path(output_path)
    meta = metadata or {}

    # Serialise all frames
    frame_bufs = [serialise_frame(f) for f in encoded_frames]

    # Build raw payload: n_frames × (len + data)
    raw = io.BytesIO()
    raw.write(struct.pack(">I", len(frame_bufs)))
    for fb in frame_bufs:
        raw.write(struct.pack(">I", len(fb)))
        raw.write(fb)

    # Compress the payload
    compressed = lzma.compress(raw.getvalue(),
                                format=lzma.FORMAT_XZ,
                                preset=6)

    # Serialise metadata
    meta_bytes = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)

    # Write file
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">B", VERSION))
        f.write(struct.pack(">Q", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(compressed)

    return output_path.stat().st_size


def read_bin(input_path: str | Path) -> tuple[list[dict], dict]:
    """
    Read and decompress a .bin file.
    Returns (list_of_encoded_frames, metadata_dict).
    """
    input_path = Path(input_path)
    with open(input_path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Not a valid encoder file: magic={magic!r}")

        version = struct.unpack(">B", f.read(1))[0]
        if version != VERSION:
            raise ValueError(f"Unsupported file version: {version}")

        meta_len  = struct.unpack(">Q", f.read(8))[0]
        meta      = pickle.loads(f.read(meta_len))
        compressed = f.read()

    raw_data = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    raw      = io.BytesIO(raw_data)

    n_frames = struct.unpack(">I", raw.read(4))[0]
    frames   = []
    for _ in range(n_frames):
        frame_len = struct.unpack(">I", raw.read(4))[0]
        frame_data = raw.read(frame_len)
        frames.append(deserialise_frame(frame_data))

    return frames, meta
