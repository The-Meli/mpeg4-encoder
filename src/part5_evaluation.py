"""
Part 5 - Evaluation & Visualisation
5a: Compression ratio, PSNR, frame-type breakdown
5b: Full pipeline matplotlib figure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2


# ── 5a Quality Metrics ────────────────────────────────────────────────────────

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two uint8 frames."""
    orig  = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    mse   = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_compression_ratio(original_frames: list[np.ndarray],
                               compressed_size_bytes: int) -> float:
    """
    ratio = original_size / compressed_size
    Original size = sum of raw BGR bytes for all frames.
    """
    total_orig = sum(f.nbytes for f in original_frames)
    return total_orig / compressed_size_bytes


def frame_type_breakdown(encoded_frames: list[dict]) -> dict:
    i_count = sum(1 for f in encoded_frames if f["type"] == "I")
    p_count = sum(1 for f in encoded_frames if f["type"] == "P")
    return {"I": i_count, "P": p_count, "total": i_count + p_count}


# ── 5b Pipeline Visualisation ─────────────────────────────────────────────────

def visualise_pipeline(
    original_frames:      list[np.ndarray],
    reconstructed_frames: list[np.ndarray],
    encoded_frames:       list[dict],
    preprocessed_frames:  list[dict],
    sample_iframe_enc:    dict,
    sample_pframe_enc:    dict | None = None,
    save_path:            str | Path | None = None,
) -> plt.Figure:
    """
    Build a comprehensive pipeline figure with 5 sections:
      1. Original frames (up to 4)
      2. Color channels (Y, Cb, Cr) of frame 0
      3. DCT & Quantisation walkthrough (one 8×8 block)
      4. Motion vectors overlaid on a P-frame
      5. Residuals & Reconstruction comparison
    Uses GridSpec for a clean, conflict-free layout.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor("#0f1117")
    plt.rcParams.update({"text.color": "white"})

    def _style(ax, title=""):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        if title:
            ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # Overall GridSpec: 5 rows, each section gets its own row
    gs = GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.35)

    # ── Section 1: Original Frames (row 0, up to 4 cols) ─────────────────────
    n_show = min(4, len(original_frames))
    fig.text(0.01, 0.97, "① Original frames", color="#7eb8f7", fontsize=11, weight="bold")
    for i in range(n_show):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2RGB))
        ftype = encoded_frames[i]["type"] if i < len(encoded_frames) else "?"
        _style(ax, f"Frame {i}  [{ftype}]")

    # ── Section 2: Color channels (row 1, cols 0-2) ───────────────────────────
    fig.text(0.01, 0.79, "② Color channels (frame 0)", color="#7eb8f7", fontsize=11, weight="bold")
    pp     = preprocessed_frames[0]
    ch_data = [pp["Y"], pp["Cb_sub"], pp["Cr_sub"]]
    ch_name = ["Y — luma", "Cb — chroma-B", "Cr — chroma-R"]
    ch_cmap = ["gray", "cool", "hot"]
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(ch_data[k], cmap=ch_cmap[k], vmin=0, vmax=255)
        _style(ax, ch_name[k])

    # ── Section 3: DCT & Quantisation (row 2, all 4 cols) ────────────────────
    fig.text(0.01, 0.61, "③ DCT & Quantisation (8×8 block)", color="#7eb8f7", fontsize=11, weight="bold")
    from part2_intra_coding import (dequantise, idct2, get_quant_matrix, dct2, quantise)
    qf        = sample_iframe_enc["quality_factor"]
    qY        = get_quant_matrix(qf, chroma=False)
    Y_orig    = preprocessed_frames[0]["Y"]
    raw_blk   = Y_orig[0:8, 0:8].astype(np.float32) - 128.0
    dct_blk   = dct2(raw_blk)
    q_blk     = quantise(dct_blk, qY)
    recon_blk = np.clip(idct2(dequantise(q_blk, qY)) + 128.0, 0, 255)

    stages = [
        (raw_blk + 128, "gray",    "Raw 8×8 pixels"),
        (np.abs(dct_blk),  "plasma",  "DCT coefficients"),
        (np.abs(q_blk),    "viridis", "Quantised coeffs"),
        (recon_blk,        "gray",    "Reconstructed block"),
    ]
    for k, (blk, cmap, title) in enumerate(stages):
        ax = fig.add_subplot(gs[2, k])
        ax.imshow(blk, cmap=cmap, aspect="equal")
        _style(ax, title)

    # ── Section 4: Motion vectors (row 3, cols 0-1) ───────────────────────────
    fig.text(0.01, 0.43, "④ Motion vectors", color="#7eb8f7", fontsize=11, weight="bold")
    ax_mv = fig.add_subplot(gs[3, 0:2])
    p_frame_idx = next((i for i, f in enumerate(encoded_frames) if f["type"] == "P"), None)

    if p_frame_idx is not None and sample_pframe_enc is not None:
        bg = cv2.cvtColor(original_frames[p_frame_idx], cv2.COLOR_BGR2RGB)
        ax_mv.imshow(bg, alpha=0.7)
        mvs   = sample_pframe_enc["motion_vectors"]
        ph    = sample_pframe_enc["padded_h"]
        pw    = sample_pframe_enc["padded_w"]
        mbs_w = pw // 16
        for idx, (dy, dx) in enumerate(mvs):
            bi = idx // mbs_w
            bj = idx  % mbs_w
            cx, cy = bj * 16 + 8, bi * 16 + 8
            if abs(dx) > 0 or abs(dy) > 0:
                ax_mv.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                               arrowprops=dict(arrowstyle="->", color="cyan",
                                               lw=0.8, mutation_scale=8))
        _style(ax_mv, f"Motion vectors — frame {p_frame_idx} [P]")
    else:
        ax_mv.text(0.5, 0.5, "No P-frame available", ha="center", va="center",
                   color="gray", fontsize=10)
        _style(ax_mv, "Motion vectors")

    # ── Section 5: Residuals & Reconstruction (row 4, cols 0-2) ──────────────
    fig.text(0.01, 0.23, "⑤ Residuals & Reconstruction", color="#7eb8f7", fontsize=11, weight="bold")
    if p_frame_idx is not None:
        orig_bgr   = original_frames[p_frame_idx]
        recon_bgr  = reconstructed_frames[p_frame_idx]
        residual   = np.abs(orig_bgr.astype(np.float32) - recon_bgr.astype(np.float32)).mean(axis=2)

        for k, (img, cmap, title) in enumerate([
            (residual,                              "hot",  "Residual map"),
            (cv2.cvtColor(orig_bgr,  cv2.COLOR_BGR2RGB), None,  "Original frame"),
            (cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB), None,  "Reconstructed frame"),
        ]):
            ax = fig.add_subplot(gs[4, k])
            ax.imshow(img, cmap=cmap, vmin=(0 if cmap else None), vmax=(50 if cmap else None))
            if title == "Reconstructed frame":
                psnr_val = compute_psnr(orig_bgr, recon_bgr)
                title = f"Reconstructed  PSNR={psnr_val:.1f} dB"
            _style(ax, title)

    fig.suptitle("MPEG-4 Simplified Encoder — Pipeline Visualisation",
                 color="white", fontsize=15, weight="bold", y=0.995)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[✓] Pipeline figure saved → {save_path}")

    return fig


def plot_compression_vs_qf(frames: list[np.ndarray],
                            qf_range: range | list = range(10, 100, 10),
                            gop_size: int = 10,
                            save_path: str | Path | None = None):
    """
    Sweep quality factors and plot compression ratio.
    Quick demo using the first few frames.
    """
    from encoder import encode_video
    from part4_entropy_coding import write_bin
    import tempfile, os

    ratios = []
    for qf in qf_range:
        enc_frames, _ = encode_video(frames, quality_factor=qf, gop_size=gop_size)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tf:
            tmp = tf.name
        compressed_bytes = write_bin(enc_frames, tmp)
        os.unlink(tmp)
        ratio = compute_compression_ratio(frames, compressed_bytes)
        ratios.append(ratio)
        print(f"  QF={qf:3d} → ratio={ratio:.2f}x")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(qf_range), ratios, "o-", color="#7eb8f7", lw=2)
    ax.set_xlabel("Quality Factor", color="white")
    ax.set_ylabel("Compression Ratio", color="white")
    ax.set_title("Compression Ratio vs Quality Factor", color="white")
    ax.set_facecolor("#1a1d27")
    fig.patch.set_facecolor("#0f1117")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_gop_vs_compression(frames: list[np.ndarray],
                             gop_range: range | list = range(1, 15),
                             quality_factor: int = 50,
                             save_path: str | Path | None = None):
    """
    Sweep GOP sizes and plot effect on compression ratio.
    """
    from encoder import encode_video
    from part4_entropy_coding import write_bin
    import tempfile, os

    ratios = []
    for gop in gop_range:
        enc_frames, _ = encode_video(frames, quality_factor=quality_factor, gop_size=gop)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tf:
            tmp = tf.name
        compressed_bytes = write_bin(enc_frames, tmp)
        os.unlink(tmp)
        ratio = compute_compression_ratio(frames, compressed_bytes)
        ratios.append(ratio)
        print(f"  GOP={gop:3d} → ratio={ratio:.2f}x")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(gop_range), ratios, "s-", color="#f7b87e", lw=2)
    ax.set_xlabel("GOP Size (G)", color="white")
    ax.set_ylabel("Compression Ratio", color="white")
    ax.set_title("Compression Ratio vs GOP Size", color="white")
    ax.set_facecolor("#1a1d27")
    fig.patch.set_facecolor("#0f1117")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig
