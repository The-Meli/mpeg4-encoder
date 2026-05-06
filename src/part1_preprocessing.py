
import numpy as np
import cv2


def bgr_to_ycbcr(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a BGR frame to YCbCr color space (BT.601).
    Returns Y, Cb, Cr channels as float32 arrays.
    """
    frame_float = frame_bgr.astype(np.float32)
    B = frame_float[:, :, 0]
    G = frame_float[:, :, 1]
    R = frame_float[:, :, 2]

    Y  =  0.299   * R + 0.587   * G + 0.114   * B
    Cb = -0.16874 * R - 0.33126 * G + 0.5     * B + 128.0
    Cr =  0.5     * R - 0.41869 * G - 0.08131 * B + 128.0

    Y  = np.clip(Y,  0, 255)
    Cb = np.clip(Cb, 0, 255)
    Cr = np.clip(Cr, 0, 255)

    return Y, Cb, Cr


def ycbcr_to_bgr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr channels back to BGR (BT.601).
    """
    Y  = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0

    R = Y + 1.402   * Cr
    G = Y - 0.34414 * Cb - 0.71414 * Cr
    B = Y + 1.772   * Cb

    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)

    bgr = np.stack([B, G, R], axis=2).astype(np.uint8)
    return bgr


def subsample_420(Cb: np.ndarray, Cr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply 4:2:0 chroma subsampling: downsample Cb and Cr by factor 2 in both dimensions.
    """
    Cb_sub = Cb[::2, ::2]
    Cr_sub = Cr[::2, ::2]
    return Cb_sub, Cr_sub


def upsample_420(Cb_sub: np.ndarray, Cr_sub: np.ndarray,
                 orig_h: int, orig_w: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Upsample Cb and Cr back to original resolution using nearest-neighbour.
    """
    Cb_up = cv2.resize(Cb_sub, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    Cr_up = cv2.resize(Cr_sub, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return Cb_up, Cr_up


def preprocess_frame(frame_bgr: np.ndarray) -> dict:
    """
    Full pre-processing pipeline for a single frame.
    Returns a dict with all channels needed for encoding.
    """
    h, w = frame_bgr.shape[:2]
    Y, Cb, Cr = bgr_to_ycbcr(frame_bgr)
    Cb_sub, Cr_sub = subsample_420(Cb, Cr)

    return {
        "Y":      Y,
        "Cb_sub": Cb_sub,
        "Cr_sub": Cr_sub,
        "orig_h": h,
        "orig_w": w,
    }


def reconstruct_frame(preprocessed: dict) -> np.ndarray:
    """
    Reconstruct a BGR frame from a preprocessed dict (for visualisation / testing).
    """
    Y      = preprocessed["Y"]
    Cb_sub = preprocessed["Cb_sub"]
    Cr_sub = preprocessed["Cr_sub"]
    h      = preprocessed["orig_h"]
    w      = preprocessed["orig_w"]

    Cb, Cr = upsample_420(Cb_sub, Cr_sub, h, w)
    return ycbcr_to_bgr(Y, Cb, Cr)
