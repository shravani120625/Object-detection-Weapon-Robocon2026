# Hand Gesture Detection (YOLOv8 + OpenCV)

This project runs real-time hand gesture detection for:
- `fist`
- `palm`
- `spearhead`

It supports:
- USB webcam (`--source usb`) on Windows/Linux

## Project File

- `camera_detect.py` - live camera detection script (with smoothing to reduce palm/fist flicker)

## 1) Install Dependencies

### Windows (USB webcam)

`bash
pip install ultralytics opencv-python 

## 2) Model Weights

Default model path:

```text
runs/detect/train/weights/best.pt
```

If your model is elsewhere, pass it with `--weights`.

## 3) Run Live Detection

### USB webcam

```bash
python camera_detect.py --source usb --camera 0 --weights runs/detect/train/weights/best.pt
```

Press `q` to quit.

## 4) Recommended Stable Settings (palm/fist confusion)

```bash
python camera_detect.py --source usb --conf 0.45 --smooth-window 9 --stable-min-votes 5 --min-box-area 8000
```

Useful tuning:
- `--conf`: increases strictness (higher = fewer false positives)
- `--smooth-window`: number of frames used for class voting
- `--stable-min-votes`: votes needed before output class is accepted
- `--min-box-area`: ignores very tiny noisy detections

## 5) Push to GitHub

Run these commands from this folder:

```bash
git init
git add .
git commit -m "Add camera detection script with Raspberry Pi 5 support"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If your repo already exists with a remote, use:

```bash
git remote -v
git push
```

## Notes

- If you want to upload dataset or weights, remove rules from `.gitignore`.
- For large files (`.pt`, datasets), consider Git LFS.
