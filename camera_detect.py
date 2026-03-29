import argparse
import platform
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


def open_usb_camera(camera_index: int, width: int | None = None, height: int | None = None) -> cv2.VideoCapture:
    """Open USB webcam with fallback camera indices."""
    candidates = [camera_index] + [i for i in range(5) if i != camera_index]
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2

    for idx in candidates:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            # Fallback when backend flag is unsupported on a platform.
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Camera opened at index {idx}")
            return cap
        cap.release()

    raise RuntimeError(
        "Could not open any camera. Check if camera is connected and not used by another app."
    )


def get_primary_detection(result, min_box_area: float):
    """Return best detection as (label, conf) after area filtering."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0

    best_label = None
    best_conf = 0.0
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        area = max(0.0, (x2 - x1) * (y2 - y1))
        if area < min_box_area:
            continue
        if conf > best_conf:
            best_conf = conf
            best_label = result.names.get(cls_id, str(cls_id))

    return best_label, best_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Live camera detection with OpenCV + YOLO")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to YOLO weights file (.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="usb",
        choices=["usb"],
        help="Camera source: 'usb' for webcam",
    )
    parser.add_argument("--camera", type=int, default=0, help="Preferred camera index")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device, e.g. cpu")
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=5000.0,
        help="Ignore tiny detections smaller than this pixel area",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Number of recent frames for class smoothing",
    )
    parser.add_argument(
        "--stable-min-votes",
        type=int,
        default=4,
        help="Minimum votes in smoothing window to accept stable class",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path.resolve()}")

    model = YOLO(str(weights_path))
    cap = open_usb_camera(args.camera, args.width, args.height)

    print("Starting live detection. Press 'q' to quit.")
    label_history = deque(maxlen=max(1, args.smooth_window))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from USB camera.")
                break

            result = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
                device=args.device,
            )[0]

            annotated = result.plot()
            label, score = get_primary_detection(result, args.min_box_area)
            label_history.append(label if label is not None else "none")

            stable_label = "none"
            stable_score = 0.0
            if label_history:
                counts = Counter(label_history)
                candidate, votes = counts.most_common(1)[0]
                if candidate != "none" and votes >= args.stable_min_votes:
                    stable_label = candidate
                    stable_score = score

            cv2.putText(
                annotated,
                f"Stable: {stable_label} ({stable_score:.2f})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
