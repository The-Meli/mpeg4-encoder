"""
Part 3 - Inter-frame Coding (P-frames)
- Group of Pictures (GOP): every G-th frame → I-frame, others → P-frame
- Block matching on 16×16 macroblocks (Y channel)
- Full search within ±S pixel window
- Residual = current - predicted → DCT + quantise
- Decoder: motion compensation + residual decoding
"""

import numpy as np
from part2_intra_coding import (
    dct2, idct2, quantise, dequantise,
    zigzag_scan, zigzag_inverse,
    get_quant_matrix, _pad_channel,
)

MB = 16  # macroblock size


def _pad_to_mb(channel: np.ndarray) -> np.ndarray:
    """Pad channel to a multiple of MB (16)."""
    h, w = channel.shape
    ph = (MB - h % MB) % MB
    pw = (MB - w % MB) % MB
    return np.pad(channel, ((0, ph), (0, pw)), mode='edge')


def _motion_search(current_mb: np.ndarray, ref: np.ndarray,
                   mb_row: int, mb_col: int, search_window: int) -> tuple[int, int]:
    """
    Full exhaustive search for the best matching macroblock in the reference frame.
    Returns (dy, dx) motion vector.

    OPTIMISATION : au lieu d'une double boucle Python qui calcule la SAD
    candidat par candidat, on construit en une seule opération NumPy un
    tenseur 4D de tous les blocs candidats (shape: n_dy × n_dx × MB × MB),
    puis on calcule toutes les SAD avec une réduction vectorisée.
    Cela remplace (2S+1)² appels Python par un seul np.abs(...).sum(),
    ce qui est ~3× plus rapide sur des images réalistes.
    """
    h_ref, w_ref = ref.shape
    y0 = mb_row * MB
    x0 = mb_col * MB

    # Filtrer les décalages qui resteraient dans les limites de l'image
    dy_range = np.arange(-search_window, search_window + 1)
    dx_range = np.arange(-search_window, search_window + 1)
    valid_dy = dy_range[(y0 + dy_range >= 0) & (y0 + dy_range + MB <= h_ref)]
    valid_dx = dx_range[(x0 + dx_range >= 0) & (x0 + dx_range + MB <= w_ref)]

    # Indices de départ pour chaque candidat (broadcasting 2D)
    ry = (y0 + valid_dy)[:, None]   # shape (n_dy, 1)
    rx = (x0 + valid_dx)[None, :]   # shape (1, n_dx)

    # Offsets intra-bloc
    r = np.arange(MB)  # shape (MB,)
    c = np.arange(MB)  # shape (MB,)

    # Extraire tous les blocs candidats en une seule opération d'indexation
    # candidates shape : (n_dy, n_dx, MB, MB)
    candidates = ref[
        ry[:, :, None, None] + r[None, None, :, None],
        rx[:, :, None, None] + c[None, None, None, :]
    ].astype(np.float32)

    # SAD vectorisée pour tous les candidats d'un coup
    sads = np.abs(candidates - current_mb.astype(np.float32)[None, None, :, :]).sum(axis=(2, 3))

    # Position du minimum
    best_flat = np.argmin(sads)
    best_i, best_j = np.unravel_index(best_flat, sads.shape)

    return int(valid_dy[best_i]), int(valid_dx[best_j])


def encode_residual_block(residual_8x8: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """DCT + quantise an 8×8 residual block, return zig-zag coefficients."""
    dct_block = dct2(residual_8x8.astype(np.float32))
    q_block   = quantise(dct_block, q_matrix)
    return zigzag_scan(q_block)


def decode_residual_block(zz: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Decode an 8×8 residual block from zig-zag coefficients."""
    q_block  = zigzag_inverse(zz)
    dq_block = dequantise(q_block, q_matrix)
    return idct2(dq_block)


def encode_pframe(current_preprocessed: dict,
                  ref_reconstructed: dict,
                  quality_factor: int = 50,
                  search_window: int = 8) -> dict:
    """
    Encode a P-frame against a reference (previously reconstructed) frame.
    Only the Y channel uses block matching; Cb/Cr are coded with full residual (no MV).
    """
    qY = get_quant_matrix(quality_factor, chroma=False)
    qC = get_quant_matrix(quality_factor, chroma=True)

    cur_Y  = current_preprocessed["Y"]
    ref_Y  = ref_reconstructed["Y"]
    orig_h = current_preprocessed["orig_h"]
    orig_w = current_preprocessed["orig_w"]

    # Pad to MB multiple
    cur_Y_pad = _pad_to_mb(cur_Y)
    ref_Y_pad = _pad_to_mb(ref_Y)
    ph, pw = cur_Y_pad.shape
    mbs_h  = ph // MB
    mbs_w  = pw // MB

    motion_vectors = []
    residual_coeffs_Y = []

    for bi in range(mbs_h):
        for bj in range(mbs_w):
            cur_mb = cur_Y_pad[bi*MB:(bi+1)*MB, bj*MB:(bj+1)*MB]
            dy, dx = _motion_search(cur_mb, ref_Y_pad, bi, bj, search_window)
            motion_vectors.append((dy, dx))

            # Predicted macroblock from reference
            ry, rx = bi*MB + dy, bj*MB + dx
            pred_mb = ref_Y_pad[ry:ry+MB, rx:rx+MB]
            residual_mb = cur_mb.astype(np.float32) - pred_mb.astype(np.float32)

            # Encode residual in 8×8 sub-blocks
            mb_coeffs = []
            for si in range(2):
                for sj in range(2):
                    sub_res = residual_mb[si*8:(si+1)*8, sj*8:(sj+1)*8]
                    zz = encode_residual_block(sub_res, qY)
                    mb_coeffs.append(zz)
            residual_coeffs_Y.append(mb_coeffs)

    # Cb / Cr: encode as intra residual (difference from zero prediction)
    from part2_intra_coding import encode_channel_intra
    enc_Cb = encode_channel_intra(current_preprocessed["Cb_sub"], qC)
    enc_Cr = encode_channel_intra(current_preprocessed["Cr_sub"], qC)

    return {
        "type":             "P",
        "quality_factor":   quality_factor,
        "search_window":    search_window,
        "orig_h":           orig_h,
        "orig_w":           orig_w,
        "padded_h":         ph,
        "padded_w":         pw,
        "motion_vectors":   motion_vectors,
        "residual_coeffs_Y": np.array(residual_coeffs_Y, dtype=object),
        "Cb":               enc_Cb,
        "Cr":               enc_Cr,
    }


def decode_pframe(encoded_frame: dict, ref_reconstructed: dict) -> dict:
    """
    Decode a P-frame using the reference frame and stored motion vectors + residuals.
    Returns a preprocessed-like dict.
    """
    qf = encoded_frame["quality_factor"]
    qY = get_quant_matrix(qf, chroma=False)
    qC = get_quant_matrix(qf, chroma=True)

    orig_h  = encoded_frame["orig_h"]
    orig_w  = encoded_frame["orig_w"]
    ph      = encoded_frame["padded_h"]
    pw      = encoded_frame["padded_w"]
    mbs_h   = ph // MB
    mbs_w   = pw // MB

    ref_Y_pad = _pad_to_mb(ref_reconstructed["Y"])
    recon_Y   = np.zeros((ph, pw), dtype=np.float32)

    for idx, (bi, bj) in enumerate([(i, j) for i in range(mbs_h) for j in range(mbs_w)]):
        dy, dx = encoded_frame["motion_vectors"][idx]
        ry, rx = bi*MB + dy, bj*MB + dx
        pred_mb = ref_Y_pad[ry:ry+MB, rx:rx+MB].astype(np.float32)

        mb_coeffs = encoded_frame["residual_coeffs_Y"][idx]
        recon_mb  = np.zeros((MB, MB), dtype=np.float32)
        for si in range(2):
            for sj in range(2):
                zz      = mb_coeffs[si*2 + sj]
                sub_res = decode_residual_block(zz, qY)
                recon_mb[si*8:(si+1)*8, sj*8:(sj+1)*8] = sub_res

        recon_Y[bi*MB:(bi+1)*MB, bj*MB:(bj+1)*MB] = pred_mb + recon_mb

    Y = np.clip(recon_Y[:orig_h, :orig_w], 0, 255)

    # Decode Cb / Cr
    from part2_intra_coding import decode_channel_intra
    Cb_sub = decode_channel_intra(encoded_frame["Cb"], qC)
    Cr_sub = decode_channel_intra(encoded_frame["Cr"], qC)

    return {
        "Y":      Y,
        "Cb_sub": Cb_sub,
        "Cr_sub": Cr_sub,
        "orig_h": orig_h,
        "orig_w": orig_w,
    }


def get_frame_type(frame_index: int, gop_size: int) -> str:
    """Return 'I' if the frame should be an I-frame, else 'P'."""
    return "I" if (frame_index % gop_size == 0) else "P"
