"""
app.py - Entry point for People Counter MVP

- Parses CLI args
- Initializes detector and tracker
- Opens a video source, runs detection and tracking, and displays frames
"""
import argparse
import sys
import time

import cv2
import numpy as np

try:
    from src.detect import PersonDetector
    from src.track import MultiObjectTracker
    from src.zones import LineCounter, RoiCounter
    from src.draw import Annotator
except ImportError:
    from detect import PersonDetector
    from track import MultiObjectTracker
    from zones import LineCounter, RoiCounter
    from draw import Annotator


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (0=webcam, path, url)",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detection (0.0 to 1.0)",
    )
    ap.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Intersection over Union (IoU) threshold for NMS (0.0 to 1.0)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on, e.g., 'cpu' or 'cuda:0'",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference in pixels (e.g., 640)",
    )
    return ap.parse_args()


def main() -> None:
    """
    Main function to run the people counter application.

    Initializes detector and tracker, captures video, runs prediction and tracking,
    and displays output.
    """
    args = parse_args()

    if not (0.0 <= args.conf <= 1.0):
        sys.exit("Error: --conf value must be between 0.0 and 1.0.")
    if not (0.0 <= args.iou <= 1.0):
        sys.exit("Error: --iou value must be between 0.0 and 1.0.")

    print(
        f"Configuração: conf={args.conf}, iou={args.iou}, "
        f"device={args.device}, imgsz={args.imgsz}"
    )

    detector = PersonDetector(
        conf=args.conf, iou=args.iou, device=args.device, imgsz=args.imgsz
    )
    tracker = MultiObjectTracker()
    annotator = Annotator()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = None
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ConnectionError(f"Failed to open video source: {args.source}")

        ret, frame = cap.read()
        if not ret:
            raise ConnectionError("Failed to read first frame from source.")

        h, w = frame.shape[:2]
        lc = LineCounter(point_a=(0, h // 2), point_b=(w, h // 2))
        rc = RoiCounter(
            polygon=[
                (int(0.3 * w), int(0.3 * h)),
                (int(0.7 * w), int(0.3 * h)),
                (int(0.7 * w), int(0.7 * h)),
                (int(0.3 * w), int(0.7 * h)),
            ]
        )

        last_print_time = time.time()

        while ret:
            detections = detector.predict(frame)
            tracks = tracker.update(frame, detections)
            line_counts = lc.update(tracks)
            roi_counts = rc.update(tracks)

            # --- Visualizations ---
            frame = annotator.annotate(
                frame=frame, tracks=tracks, line_counter=lc, roi_counter=rc
            )

            # --- Overlays (Metrics & Legend) ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            white_color = (255, 255, 255)
            black_color = (0, 0, 0)

            # Metrics overlay (top-left)
            metrics_text = (
                f"Pessoas: {len(detections)} | IDs: {len(tracks)} | "
                f"A->B: {line_counts['a_to_b']} | B->A: {line_counts['b_to_a']} | "
                f"ROI: {roi_counts['present']}/{roi_counts['unique_ids']}"
            )
            cv2.putText(frame, metrics_text, (11, 21), font, font_scale, black_color, font_thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, metrics_text, (10, 20), font, font_scale, white_color, font_thickness, cv2.LINE_AA)

            # Legend overlay (bottom-left)
            legend_lines = [
                "Linha vermelha tracejada (A-B): linha usada pelo LineCounter para detectar cruzamentos.",
                "Retangulo verde: ROI (regiao de interesse) onde contamos presentes e IDs unicos.",
                "Caixa azul: bounding box da pessoa detectada (com ID).",
                "Ponto azul: centro do bounding box, usado para lado da linha e ROI.",
            ]
            base_y = h - 70
            for i, line in enumerate(legend_lines):
                y = base_y + i * 18
                cv2.putText(frame, line, (11, y + 1), font, font_scale, black_color, font_thickness, cv2.LINE_AA)
                cv2.putText(frame, line, (10, y), font, font_scale, white_color, font_thickness, cv2.LINE_AA)


            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                print(
                    f"Pessoas detectadas: {len(detections)} | "
                    f"IDs ativos: {len(tracks)} | "
                    f"A->B: {line_counts['a_to_b']} | "
                    f"B->A: {line_counts['b_to_a']} | "
                    f"ROI (presente/unicos): {roi_counts['present']}/"
                    f"{roi_counts['unique_ids']}"
                )
                last_print_time = current_time

            cv2.imshow("People Counter - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            ret, frame = cap.read()

    except (ConnectionError, Exception) as e:
        sys.exit(f"Error: {e}")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
