"""
app.py - Entry point for People Counter MVP (Starter)
- Parses CLI args
- Opens a video source
- Displays frames until 'q' is pressed
"""
import cv2
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="Video source (0=webcam, path, url)")
    return ap.parse_args()

def main():
    args = parse_args()
    source = 0 if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source {args.source}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Starter - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
