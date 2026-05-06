import cv2
import os
import sys
import argparse


def extract_frames(video_path: str, output_dir: str = "frames", n_frames: int = 30, start_frame: int = 0):
    """
    Extrait n_frames frames CONSÉCUTIVES à partir de start_frame.
    C'est indispensable pour que l'encodeur MPEG-4 puisse exploiter
    la redondance temporelle entre frames voisines (P-frames).
    """
    if not os.path.exists(video_path):
        print(f"❌ Fichier introuvable : {video_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Vidéo     : {video_path}")
    print(f"Résolution: {w}x{h}  |  FPS: {fps:.1f}  |  Total frames: {total}")

    # Vérifications
    if start_frame >= total:
        print(f"❌ start_frame ({start_frame}) >= total frames ({total})")
        sys.exit(1)

    available = total - start_frame
    if n_frames > available:
        print(f"⚠️  Seulement {available} frames disponibles à partir de {start_frame}, ajustement.")
        n_frames = available

    print(f"Extraction: {n_frames} frames consécutives depuis la frame {start_frame} → {output_dir}/")

    # Positionner au point de départ, puis lire séquentiellement
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    saved = 0

    for i in range(n_frames):
        ret, frame = cap.read()   # lecture séquentielle : frame start+0, start+1, start+2 …
        if not ret:
            print(f"⚠️  Lecture interrompue à la frame {start_frame + i}")
            break
        cv2.imwrite(f"{output_dir}/frame_{saved:04d}.png", frame)
        saved += 1

    cap.release()

    duration_ms = (saved / fps) * 1000 if fps > 0 else 0
    print(f"\n✅ {saved} frames consécutives sauvegardées dans ./{output_dir}/")
    print(f"   Durée représentée : {duration_ms:.0f} ms  ({saved/fps:.2f} s à {fps:.1f} fps)")
    print(f"\n   Prochaine étape :")
    print(f"   python encoder.py encode ./{output_dir}/ output.bin --qf 50 --gop 10")
    print(f"   python encoder.py visualise ./{output_dir}/ ./figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraire des frames CONSÉCUTIVES d'une vidéo pour l'encodeur MPEG-4"
    )
    parser.add_argument("video",          help="Chemin vers la vidéo MP4")
    parser.add_argument("--n",     type=int, default=30,  help="Nombre de frames consécutives (défaut: 30)")
    parser.add_argument("--start", type=int, default=0,   help="Index de la frame de départ (défaut: 0)")
    parser.add_argument("--out",   type=str, default="frames", help="Dossier de sortie (défaut: frames)")
    args = parser.parse_args()

    extract_frames(args.video, args.out, args.n, args.start)