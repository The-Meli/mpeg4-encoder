"""
Part 2 - Intra-frame Coding (I-frames)
- Divide each channel into 8×8 blocks
- Apply 2D DCT
- Quantise with a quality-factor-scaled quantisation matrix
- Encode coefficients in zig-zag order
- Decoder: dequantise then IDCT
"""

import numpy as np
from scipy.fftpack import dct, idct


# Standard JPEG luminance quantisation matrix
LUMA_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float32)

# Standard JPEG chrominance quantisation matrix
CHROMA_QUANT_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)

# Pre-computed zig-zag index order for an 8×8 block
_ZIGZAG_ORDER = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]


def get_quant_matrix(quality_factor: int, chroma: bool = False) -> np.ndarray:
    """
    Scale the base quantisation matrix by a quality factor (1-100, higher = better quality).
    """
    base = CHROMA_QUANT_TABLE if chroma else LUMA_QUANT_TABLE
    if quality_factor <= 0:
        quality_factor = 1
    if quality_factor > 100:
        quality_factor = 100

    if quality_factor < 50:
        scale = 5000 / quality_factor
    else:
        scale = 200 - 2 * quality_factor

    q = np.floor((base * scale + 50) / 100).astype(np.float32)
    q = np.clip(q, 1, 255)
    return q


def dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II (orthonormal) on an 8×8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block: np.ndarray) -> np.ndarray:
    """2D IDCT-II (orthonormal) on an 8×8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def quantise(dct_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Quantise DCT coefficients."""
    return np.round(dct_block / q_matrix).astype(np.int16)


def dequantise(q_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Dequantise coefficients."""
    return (q_block.astype(np.float32) * q_matrix)


def zigzag_scan(block: np.ndarray) -> np.ndarray:
    """Flatten an 8×8 block in zig-zag order."""
    return np.array([block[r, c] for r, c in _ZIGZAG_ORDER], dtype=np.int16)


def zigzag_inverse(coeffs: np.ndarray) -> np.ndarray:
    """Reconstruct an 8×8 block from zig-zag coefficients."""
    block = np.zeros((8, 8), dtype=np.int16)
    for idx, (r, c) in enumerate(_ZIGZAG_ORDER):
        block[r, c] = coeffs[idx]
    return block


def _pad_channel(channel: np.ndarray) -> np.ndarray:
    """Pad channel to multiple of 8."""
    h, w = channel.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    return np.pad(channel, ((0, ph), (0, pw)), mode='edge')


def encode_channel_intra(channel: np.ndarray, q_matrix: np.ndarray) -> dict:
    """
    Encode a full channel (Y, Cb, or Cr) using DCT + quantisation.
    Returns encoded coefficients and metadata.
    """
    h, w = channel.shape
    padded = _pad_channel(channel)
    ph, pw = padded.shape

    # Centre around 0 before DCT
    centred = padded.astype(np.float32) - 128.0

    blocks_h = ph // 8
    blocks_w = pw // 8
    coeffs_all = []

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            block = centred[bi*8:(bi+1)*8, bj*8:(bj+1)*8]
            dct_block  = dct2(block)
            q_block    = quantise(dct_block, q_matrix)
            zz         = zigzag_scan(q_block)
            coeffs_all.append(zz)

    return {
        "coeffs":   np.array(coeffs_all, dtype=np.int16),
        "orig_h":   h,
        "orig_w":   w,
        "padded_h": ph,
        "padded_w": pw,
    }


def decode_channel_intra(encoded: dict, q_matrix: np.ndarray) -> np.ndarray:
    """
    Decode a channel from its DCT coefficients.
    """
    h        = encoded["orig_h"]
    w        = encoded["orig_w"]
    ph       = encoded["padded_h"]
    pw       = encoded["padded_w"]
    coeffs_all = encoded["coeffs"]

    blocks_h = ph // 8
    blocks_w = pw // 8
    recon = np.zeros((ph, pw), dtype=np.float32)

    idx = 0
    for bi in range(blocks_h):
        for bj in range(blocks_w):
            zz         = coeffs_all[idx]; idx += 1
            q_block    = zigzag_inverse(zz)
            dq_block   = dequantise(q_block, q_matrix)
            block_recon = idct2(dq_block)
            recon[bi*8:(bi+1)*8, bj*8:(bj+1)*8] = block_recon

    # Undo centering and crop padding
    recon = np.clip(recon + 128.0, 0, 255)
    return recon[:h, :w]


def encode_iframe(preprocessed: dict, quality_factor: int = 50) -> dict:
    """
    Encode a pre-processed frame as an I-frame.
    """
    qY  = get_quant_matrix(quality_factor, chroma=False)
    qC  = get_quant_matrix(quality_factor, chroma=True)

    enc_Y  = encode_channel_intra(preprocessed["Y"],      qY)
    enc_Cb = encode_channel_intra(preprocessed["Cb_sub"], qC)
    enc_Cr = encode_channel_intra(preprocessed["Cr_sub"], qC)

    return {
        "type":           "I",
        "quality_factor": quality_factor,
        "Y":              enc_Y,
        "Cb":             enc_Cb,
        "Cr":             enc_Cr,
        "orig_h":         preprocessed["orig_h"],
        "orig_w":         preprocessed["orig_w"],
    }


def decode_iframe(encoded_frame: dict) -> dict:
    """
    Decode an I-frame, returning a preprocessed-like dict for reconstruction.
    """
    qf  = encoded_frame["quality_factor"]
    qY  = get_quant_matrix(qf, chroma=False)
    qC  = get_quant_matrix(qf, chroma=True)

    Y      = decode_channel_intra(encoded_frame["Y"],  qY)
    Cb_sub = decode_channel_intra(encoded_frame["Cb"], qC)
    Cr_sub = decode_channel_intra(encoded_frame["Cr"], qC)

    return {
        "Y":      Y,
        "Cb_sub": Cb_sub,
        "Cr_sub": Cr_sub,
        "orig_h": encoded_frame["orig_h"],
        "orig_w": encoded_frame["orig_w"],
    }
