"""
encoder.py — Main encoder/decoder pipeline
Combines Parts 1-4 into full encode_video() and decode_video() functions.
"""

import numpy as np
import cv2
from pathlib import Path

from part1_preprocessing import preprocess_frame, reconstruct_frame
from part2_intra_coding   import encode_iframe, decode_iframe
from part3_inter_coding   import encode_pframe, decode_pframe, get_frame_type
from part4_entropy_coding import write_bin, read_bin


def load_frames(frames_dir: str | Path) -> list[np.ndarray]:
    """
    Load all .png/.jpg frames from a directory, sorted by filename.
    Returns a list of BGR numpy arrays.
    """
    frames_dir = Path(frames_dir)
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No image files found in {frames_dir}")
    frames = [cv2.imread(str(p)) for p in paths]
    print(f"[✓] Loaded {len(frames)} frames from {frames_dir}")
    return frames


def encode_video(
    frames:         list[np.ndarray],
    quality_factor: int = 50,
    gop_size:       int = 10,
    search_window:  int = 8,
    verbose:        bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Encode a list of BGR frames.

    Returns:
      encoded_frames  — list of encoded dicts (I or P frames)
      preprocessed    — list of preprocessed dicts (for visualisation)
    """
    encoded_frames  = []
    preprocessed    = []
    last_recon      = None   # last reconstructed (decoded) frame for P-frame reference

    for i, frame_bgr in enumerate(frames):
        pp = preprocess_frame(frame_bgr)
        preprocessed.append(pp)

        ftype = get_frame_type(i, gop_size)

        if ftype == "I" or last_recon is None:
            enc = encode_iframe(pp, quality_factor=quality_factor)
            # Decode immediately to use as reference for next P-frame
            last_recon = decode_iframe(enc)
        else:
            enc = encode_pframe(pp, last_recon,
                                 quality_factor=quality_factor,
                                 search_window=search_window)
            last_recon = decode_pframe(enc, last_recon)

        encoded_frames.append(enc)
        if verbose:
            print(f"  Frame {i:4d} [{enc['type']}] encoded")

    if verbose:
        from part5_evaluation import frame_type_breakdown
        breakdown = frame_type_breakdown(encoded_frames)
        print(f"[✓] Encoding done — {breakdown['I']} I-frames, {breakdown['P']} P-frames")

    return encoded_frames, preprocessed


def decode_video(
    encoded_frames:  list[dict],
    original_frames: list[np.ndarray] | None = None,
    verbose:         bool = True,
) -> list[np.ndarray]:
    """
    Decode a list of encoded frames back to BGR images.
    If original_frames is provided, PSNR is computed and displayed for each frame.
    """
    from part5_evaluation import compute_psnr

    reconstructed = []
    last_recon    = None
    psnr_values   = []

    for i, enc in enumerate(encoded_frames):
        if enc["type"] == "I":
            dec = decode_iframe(enc)
        else:
            dec = decode_pframe(enc, last_recon)

        last_recon = dec
        bgr = reconstruct_frame(dec)
        reconstructed.append(bgr)

        # ── PSNR ──────────────────────────────────────────────────────────────
        if original_frames is not None and i < len(original_frames):
            psnr = compute_psnr(original_frames[i], bgr)
            psnr_values.append(psnr)
            if verbose:
                psnr_str = f"{psnr:.2f} dB" if psnr != float("inf") else "inf (lossless)"
                print(f"  Frame {i:4d} [{enc['type']}] decoded  |  PSNR: {psnr_str}")
        else:
            if verbose:
                print(f"  Frame {i:4d} [{enc['type']}] decoded")

    # ── Summary ───────────────────────────────────────────────────────────────
    if verbose:
        print(f"[✓] Decoding done — {len(reconstructed)} frames")
        if psnr_values:
            finite_psnr = [v for v in psnr_values if v != float("inf")]
            if finite_psnr:
                avg  = np.mean(finite_psnr)
                best = np.max(finite_psnr)
                worst= np.min(finite_psnr)
                print(f"    PSNR moyen  : {avg:.2f} dB")
                print(f"    PSNR max    : {best:.2f} dB")
                print(f"    PSNR min    : {worst:.2f} dB")

    return reconstructed


def save_frames(frames: list[np.ndarray], output_dir: str | Path):
    """Save reconstructed BGR frames as PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        path = output_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(path), frame)
    print(f"[✓] Saved {len(frames)} frames to {output_dir}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simplified MPEG-4 Encoder/Decoder")

    sub = parser.add_subparsers(dest="command")

    # Encode
    enc_p = sub.add_parser("encode", help="Encode frames to .bin")
    enc_p.add_argument("input_dir",  type=str, help="Directory of input frames")
    enc_p.add_argument("output_bin", type=str, help="Output .bin file")
    enc_p.add_argument("--qf",  type=int, default=50,  help="Quality factor (1-100)")
    enc_p.add_argument("--gop", type=int, default=10,  help="GOP size")
    enc_p.add_argument("--sw",  type=int, default=8,   help="Motion search window")

    # Decode
    dec_p = sub.add_parser("decode", help="Decode .bin to frames")
    dec_p.add_argument("input_bin",   type=str, help="Input .bin file")
    dec_p.add_argument("output_dir",  type=str, help="Directory for output frames")
    dec_p.add_argument("--original_dir", type=str, default=None,
                       help="(Optional) Original frames directory — enables PSNR computation")

    # Visualise
    vis_p = sub.add_parser("visualise", help="Run full pipeline + save figures")
    vis_p.add_argument("input_dir",   type=str, help="Directory of input frames")
    vis_p.add_argument("output_dir",  type=str, help="Output directory for figures")
    vis_p.add_argument("--qf",  type=int, default=50)
    vis_p.add_argument("--gop", type=int, default=10)

    args = parser.parse_args()

    if args.command == "encode":
        frames = load_frames(args.input_dir)
        enc_frames, _ = encode_video(frames,
                                      quality_factor=args.qf,
                                      gop_size=args.gop,
                                      search_window=args.sw)
        meta = {"quality_factor": args.qf, "gop_size": args.gop,
                "n_frames": len(frames)}
        size = write_bin(enc_frames, args.output_bin, metadata=meta)
        orig_size = sum(f.nbytes for f in frames)
        print(f"[✓] Written {args.output_bin} ({size:,} bytes)")
        print(f"    Compression ratio: {orig_size/size:.2f}x")

    elif args.command == "decode":
        enc_frames, meta = read_bin(args.input_bin)
        print(f"[✓] Read {len(enc_frames)} frames from {args.input_bin}")

        # Load original frames for PSNR if provided
        original_frames = None
        if args.original_dir:
            original_frames = load_frames(args.original_dir)

        recon = decode_video(enc_frames, original_frames=original_frames)
        save_frames(recon, args.output_dir)

    elif args.command == "visualise":
        import matplotlib
        matplotlib.use("Agg")
        from part5_evaluation import (
            visualise_pipeline,
            plot_compression_vs_qf,
            plot_gop_vs_compression,
        )

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        frames = load_frames(args.input_dir)
        enc_frames, pp_frames = encode_video(frames,
                                              quality_factor=args.qf,
                                              gop_size=args.gop)
        # Pass original frames so PSNR is shown during visualise too
        recon_frames = decode_video(enc_frames, original_frames=frames)

        p_enc = next((f for f in enc_frames if f["type"] == "P"), None)
        i_enc = next((f for f in enc_frames if f["type"] == "I"), None)

        fig = visualise_pipeline(
            original_frames=frames,
            reconstructed_frames=recon_frames,
            encoded_frames=enc_frames,
            preprocessed_frames=pp_frames,
            sample_iframe_enc=i_enc,
            sample_pframe_enc=p_enc,
            save_path=out / "pipeline_visualisation.png",
        )

        # ── Compression ratio vs Quality Factor ───────────────────────────────
        print("[→] Generating compression ratio vs Quality Factor plot …")
        plot_compression_vs_qf(
            frames,
            qf_range=range(10, 100, 10),
            gop_size=args.gop,
            save_path=out / "compression_vs_qf.png",
        )

        # ── Compression ratio vs GOP size ─────────────────────────────────────
        print("[→] Generating compression ratio vs GOP size plot …")
        plot_gop_vs_compression(
            frames,
            gop_range=range(1, min(len(frames) + 1, 16)),
            quality_factor=args.qf,
            save_path=out / "compression_vs_gop.png",
        )

        print(f"[✓] All figures saved to {out}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()