"""
app.py - Entry point for People Counter MVP
- Parses CLI args
- Initializes detector
- Opens a video source, runs detection, and displays frames
"""
import argparse
import sys
import time
from typing import Optional

import cv2

from detect import PersonDetector


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
        type=str,  # Changed to Optional[str] in spirit, but argparse handles it
        default=None,
        help="Device to run inference on, e.g., 'cpu' or 'cuda:0'",
    )
    return ap.parse_args()


def main() -> None:
    """
    Main function to run the people counter application.

    Initializes detector, captures video, runs prediction, and displays output.
    """
    args = parse_args()

    if not (0.0 <= args.conf <= 1.0):
        sys.exit("Error: --conf value must be between 0.0 and 1.0.")
    if not (0.0 <= args.iou <= 1.0):
        sys.exit("Error: --iou value must be between 0.0 and 1.0.")

    print(f"Configuração: conf={args.conf}, iou={args.iou}, device={args.device}")

    detector = PersonDetector(
        conf=args.conf, iou=args.iou, device=args.device, imgsz=640
    )

    source = int(args.source) if args.source.isdigit() else args.source
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ConnectionError(f"Failed to open video source: {args.source}")
    except (ConnectionError, Exception) as e:
        sys.exit(f"Error: {e}")

    last_print_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        detections = detector.predict(frame)

        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            print(f"Pessoas detectadas: {len(detections)}")
            last_print_time = current_time

        cv2.imshow("People Counter - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
