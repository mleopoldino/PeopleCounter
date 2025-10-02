"""
app.py - Entry point for People Counter MVP

- Parses CLI args
- Initializes detector and tracker
- Opens a video source, runs detection and tracking, and displays frames
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from .detect import PersonDetector
from .draw import Annotator
from .metrics import SecondBasedEmitter, append_to_csv, write_summary_json
from .track import MultiObjectTracker
from .zones import LineCounter, RoiCounter


def validate_model_exists(model_name: str) -> bool:
    """
    Checks if the model file exists locally or is a known Ultralytics model.

    Args:
        model_name (str): The model name or path.

    Returns:
        bool: True if the model exists or is a valid Ultralytics model name.
    """
    # Check if it's a local file
    if Path(model_name).exists():
        return True

    # Check if it's a standard Ultralytics model name
    valid_models = [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
    ]
    if model_name in valid_models:
        return True

    return False


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
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Name of the YOLO model file (e.g., yolov8n.pt, yolov8s.pt)",
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
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional: Path to CSV file to log metrics.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional: Path to JSON file to save final summary.",
    )
    ap.add_argument(
        "--line",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional: Line coordinates for crossing counter (x1 y1 x2 y2).",
    )
    ap.add_argument(
        "--roi",
        type=float,
        nargs="+",
        default=None,
        metavar="COORD",
        help="Optional: ROI polygon coordinates (x1 y1 x2 y2 x3 y3 ...).",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (no cv2.imshow). Requires --output-video.",
    )
    ap.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional: Path to save annotated output video.",
    )
    return ap.parse_args()


def main() -> None:
    """
    Main function to run the people counter application.

    Initializes detector and tracker, captures video, runs prediction and tracking,
    and displays output.
    """
    args = parse_args()

    # Validate arguments
    if not (0.0 <= args.conf <= 1.0):
        sys.exit("Error: --conf value must be between 0.0 and 1.0.")
    if not (0.0 <= args.iou <= 1.0):
        sys.exit("Error: --iou value must be between 0.0 and 1.0.")

    if not validate_model_exists(args.model):
        sys.exit(
            f"Error: Model '{args.model}' not found. "
            "Please provide a valid model path or name (e.g., yolov8n.pt)."
        )

    if args.headless and not args.output_video:
        sys.exit("Error: --headless mode requires --output-video to be specified.")

    if args.roi and len(args.roi) < 6:
        sys.exit(
            "Error: --roi requires at least 3 points "
            "(6 coordinates: x1 y1 x2 y2 x3 y3)."
        )

    if args.roi and len(args.roi) % 2 != 0:
        sys.exit("Error: --roi requires an even number of coordinates (pairs of x y).")

    print(
        f"Configuração: model={args.model}, conf={args.conf}, iou={args.iou}, "
        f"device={args.device}, imgsz={args.imgsz}"
    )

    detector = PersonDetector(
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
    )
    tracker = MultiObjectTracker()
    annotator = Annotator()
    metrics_emitter = SecondBasedEmitter(interval_seconds=1)

    csv_path = Path(args.csv) if args.csv else None
    csv_header = [
        "timestamp",
        "line_in",
        "line_out",
        "roi_current",
        "roi_unique",
        "fps_processing",
    ]

    source = int(args.source) if args.source.isdigit() else args.source
    cap = None
    lc, rc = None, None  # Define here to be accessible in finally block
    frame_count = 0
    total_line_in = 0
    total_line_out = 0
    last_roi_counts = {"present": 0, "unique_ids": 0}
    processing_fps = 0
    fps_buffer = deque(maxlen=30)  # Buffer for averaging FPS over ~1 second

    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ConnectionError(f"Failed to open video source: {args.source}")

        ret, frame = cap.read()
        if not ret:
            raise ConnectionError("Failed to read first frame from source.")

        h, w = frame.shape[:2]

        # Validate and create line counter
        if args.line:
            x1, y1, x2, y2 = args.line
            if not (0 <= x1 <= w and 0 <= x2 <= w):
                sys.exit(f"Error: Line x coordinates must be between 0 and {w}.")
            if not (0 <= y1 <= h and 0 <= y2 <= h):
                sys.exit(f"Error: Line y coordinates must be between 0 and {h}.")
            lc = LineCounter(point_a=(int(x1), int(y1)), point_b=(int(x2), int(y2)))
        else:
            # Default: horizontal line in the middle
            lc = LineCounter(point_a=(0, h // 2), point_b=(w, h // 2))

        # Validate and create ROI counter
        if args.roi:
            roi_points = [
                (int(args.roi[i]), int(args.roi[i + 1]))
                for i in range(0, len(args.roi), 2)
            ]
            # Validate all points are within frame
            for i, (x, y) in enumerate(roi_points):
                if not (0 <= x <= w):
                    sys.exit(
                        f"Error: ROI point {i+1} x coordinate ({x}) "
                        f"must be between 0 and {w}."
                    )
                if not (0 <= y <= h):
                    sys.exit(
                        f"Error: ROI point {i+1} y coordinate ({y}) "
                        f"must be between 0 and {h}."
                    )
            rc = RoiCounter(polygon=roi_points)
        else:
            # Default: rectangle in the center
            rc = RoiCounter(
                polygon=[
                    (int(0.3 * w), int(0.3 * h)),
                    (int(0.7 * w), int(0.3 * h)),
                    (int(0.7 * w), int(0.7 * h)),
                    (int(0.3 * w), int(0.7 * h)),
                ]
            )

        # Setup video writer if output is requested
        video_writer = None
        if args.output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_out = cap.get(cv2.CAP_PROP_FPS) or 30.0
            video_writer = cv2.VideoWriter(args.output_video, fourcc, fps_out, (w, h))

        while ret:
            start_time = time.perf_counter()

            frame_count += 1
            detections = detector.predict(frame)
            tracks = tracker.update(frame, detections)

            line_counts_delta = lc.update(tracks)
            total_line_in += line_counts_delta["a_to_b"]
            total_line_out += line_counts_delta["b_to_a"]

            last_roi_counts = rc.update(tracks)

            # --- FPS Calculation ---
            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)
            fps_buffer.append(fps)
            if len(fps_buffer) == fps_buffer.maxlen:
                processing_fps = round(sum(fps_buffer) / len(fps_buffer))

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

            metrics_text = (
                f"FPS: {processing_fps} | Pessoas: {len(detections)} | "
                f"IDs: {len(tracks)} | A->B: {total_line_in} | "
                f"B->A: {total_line_out} | "
                f"ROI: {last_roi_counts['present']}/{last_roi_counts['unique_ids']}"
            )
            cv2.putText(
                frame,
                metrics_text,
                (11, 21),
                font,
                font_scale,
                black_color,
                font_thickness + 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                metrics_text,
                (10, 20),
                font,
                font_scale,
                white_color,
                font_thickness,
                cv2.LINE_AA,
            )

            legend_lines = [
                "Linha vermelha (A-B): contagem de cruzamentos.",
                "Retangulo verde: contagem de ocupação na área.",
                "Caixa azul: bounding box da pessoa detectada (com ID).",
                "Ponto azul: centro do bounding box, usado para contagem.",
            ]
            base_y = h - 70
            for i, line in enumerate(legend_lines):
                y = base_y + i * 18
                cv2.putText(
                    frame,
                    line,
                    (11, y + 1),
                    font,
                    font_scale,
                    black_color,
                    font_thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    line,
                    (10, y),
                    font,
                    font_scale,
                    white_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

            # --- Telemetry (JSON/CSV logging) ---
            def emit_metrics():
                metrics_data = {
                    "timestamp": int(time.time()),
                    "line_in": total_line_in,
                    "line_out": total_line_out,
                    "roi_current": last_roi_counts["present"],
                    "roi_unique": last_roi_counts["unique_ids"],
                    "fps_processing": processing_fps,
                }
                # Print JSON to stdout
                print(json.dumps(metrics_data))

                # Append to CSV if enabled
                if csv_path:
                    append_to_csv(csv_path, metrics_data, header_columns=csv_header)

            metrics_emitter.tick(emit_metrics)

            # Write frame to output video if requested
            if video_writer:
                video_writer.write(frame)

            # Display frame unless in headless mode
            if not args.headless:
                cv2.imshow("People Counter - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            ret, frame = cap.read()

    except (ConnectionError, Exception) as e:
        sys.exit(f"Error: {e}")
    finally:
        if cap:
            cap.release()
        if video_writer:
            video_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

        # --- Final Summary ---
        if args.out and lc and rc:
            summary_path = Path(args.out)
            summary_data = {
                "total_frames": frame_count,
                "line_in": total_line_in,
                "line_out": total_line_out,
                "roi_total_unique": len(rc._seen),
            }
            print(f"Salvando resumo final em: {summary_path}")
            write_summary_json(summary_path, summary_data)


if __name__ == "__main__":
    main()
