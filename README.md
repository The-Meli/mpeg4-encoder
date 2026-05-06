# mpeg4-encoder

A simplified MPEG-4-like video encoder/decoder pipeline implemented in Python from scratch — covering color space conversion, DCT-based intra coding, motion-compensated inter coding, and LZMA entropy compression.

Built as part of a Multimedia Systems university project.

---

## Pipeline Overview

```
Raw BGR Frames
      │
      ▼
┌─────────────────┐
│  Pre-processing  │  YCbCr conversion + 4:2:0 chroma subsampling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   I-frame (DCT) │  8×8 block DCT + quantisation + zig-zag scan
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ P-frame (Motion)│  16×16 macroblock matching + residual DCT
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Entropy Coding  │  Pickle serialisation + LZMA compression → .bin
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │  PSNR, compression ratio, pipeline visualisation
└─────────────────┘
```

---

## Project Structure

```
mpeg4-encoder/
│
├── src/
│   ├── extract_frames.py        # Extract consecutive frames from a video
│   ├── encoder.py               # Main CLI — encode / decode / visualise
│   ├── part1_preprocessing.py   # YCbCr conversion + chroma subsampling
│   ├── part2_intra_coding.py    # I-frame DCT + quantisation
│   ├── part3_inter_coding.py    # P-frame motion estimation + residuals
│   ├── part4_entropy_coding.py  # LZMA entropy coding + .bin format
│   └── part5_evaluation.py      # PSNR, compression ratio, visualisation
│
├── frames/                      # Sample input frame sequence
├── output/
│   └── output.bin               # Compressed output (sample)
├── figures/
│   └── pipeline_visualisation.png
│
├── report.pdf                   # Full project report (French)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/mpeg4-encoder.git
cd mpeg4-encoder
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Extract frames from a video
```bash
python src/extract_frames.py video.mp4 --n 30 --out frames/
```

### Step 2 — Encode
```bash
python src/encoder.py encode frames/ output/output.bin --qf 50 --gop 5 --sw 8
```

### Step 3 — Decode
```bash
python src/encoder.py decode output/output.bin output_frames/
```

### Step 4 — Visualise the full pipeline
```bash
python src/encoder.py visualise frames/ figures/ --qf 50 --gop 5
```

---

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--qf` | Quality factor (1–100, higher = better quality) | `50` |
| `--gop` | GOP size — frequency of I-frames | `10` |
| `--sw` | Motion search window (±pixels) | `8` |
| `--n` | Number of consecutive frames to extract | `30` |
| `--start` | Starting frame index | `0` |

---
## Key Design Choices

- **YCbCr + 4:2:0** — separates luma from chroma; human vision is less sensitive to chroma detail
- **DCT on 8×8 blocks** — concentrates energy in low-frequency coefficients; standard JPEG quantisation matrices
- **Full exhaustive motion search** — guarantees optimal motion vector per macroblock; vectorised with NumPy (~3× faster than naive loop)
- **LZMA compression** — outperforms gzip by 15–30% on structured DCT data with many repeated/zero values

---

## Report

A full written report (in French) is available in [`report.pdf`](./report.pdf), covering:
- Pipeline description
- Design choices and justifications
- Experimental analysis (compression ratio vs QF, GOP size vs compression)

---

## Requirements

```
numpy
opencv-python
scipy
matplotlib
```

---

## Authors

- AIT AHCENE MELISSA
- BELAID MERIEM
*Multimedia Systems — April 2026*
