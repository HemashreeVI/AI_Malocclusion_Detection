"""Model loading, detection and image annotation."""
import os

import cv2
import numpy as np

from .config import CATEGORY_COLORS, CATEGORY_LABELS, DEFAULT_COLOR, LOW_CONFIDENCE
from .fdi import assign_tooth
from .findings import make_finding

DEFAULT_GDRIVE_ID = '1tFPVneb7mWxo5raaUoG5N2SxLcc4bR5k'
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights')


def resolve_weights():
    """Find the detector weights. Order:
    1. MODEL_PATH (default weights/best.pt) if it exists
    2. Hugging Face Hub: HF_MODEL_REPO (+ HF_MODEL_FILE, default best.pt)
    3. Google Drive: MODEL_GDRIVE_ID (defaults to the published checkpoint)
    """
    path = os.environ.get('MODEL_PATH', os.path.join(WEIGHTS_DIR, 'best.pt'))
    if os.path.exists(path):
        return path
    repo = os.environ.get('HF_MODEL_REPO')
    if repo:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, os.environ.get('HF_MODEL_FILE', 'best.pt'))
    import gdown
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gdown.download(id=os.environ.get('MODEL_GDRIVE_ID', DEFAULT_GDRIVE_ID), output=path, quiet=False)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Could not obtain model weights at {path}')
    return path


def load_models():
    """Returns ``(detector, tooth_model_or_None)``. Set TOOTH_MODEL_PATH to a
    YOLO model whose class names are FDI numbers to get real tooth numbering."""
    from ultralytics import YOLO
    detector = YOLO(resolve_weights())
    tooth_path = os.environ.get('TOOTH_MODEL_PATH')
    tooth_model = YOLO(tooth_path) if tooth_path and os.path.exists(tooth_path) else None
    return detector, tooth_model


def _tooth_boxes(tooth_model, img_bgr):
    if tooth_model is None:
        return None
    res = tooth_model(img_bgr, conf=0.4, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        name = str(tooth_model.names[int(b.cls[0])])
        if name.isdigit():
            boxes.append((int(name), [float(v) for v in b.xyxy[0].tolist()]))
    return boxes


def detect(model, img_rgb, conf_threshold=0.25, iou=0.45, tooth_model=None):
    """Run the detector. ``img_rgb`` is an RGB uint8 array."""
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)   # ultralytics treats ndarrays as BGR
    res = model(img_bgr, conf=conf_threshold, iou=iou, verbose=False)[0]
    tooth_boxes = _tooth_boxes(tooth_model, img_bgr)

    detections = []
    for b in res.boxes:
        cls = model.names[int(b.cls[0])]
        box = [int(round(v)) for v in b.xyxy[0].tolist()]
        tooth, source = assign_tooth(cls, box, w, h, tooth_boxes)
        detections.append(make_finding(cls, float(b.conf[0]), box, tooth, source))
    return detections


def finding_label(d):
    tooth = ''
    if d['tooth'] is not None:
        tooth = f"{d['tooth']}{'~' if d['tooth_source'] == 'estimated' else ''} "
    flag = '' if d['reliable'] else ' (unvalidated)'
    return f"{tooth}{d['class']} {d['conf']*100:.0f}%{flag}"


def _corner_marks(img, box, color, length, thickness):
    x1, y1, x2, y2 = box
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness)


def annotate(img_rgb, detections):
    """Draw findings. Solid box = validated class; corner marks only = low
    confidence or unvalidated class."""
    h, w = img_rgb.shape[:2]
    scale = max(0.5, min(w, h) / 1000)
    overlay = img_rgb.copy()
    for d in detections:
        if d['reliable']:
            x1, y1, x2, y2 = d['box']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), CATEGORY_COLORS.get(d['category'], DEFAULT_COLOR), -1)
    out = cv2.addWeighted(overlay, 0.18, img_rgb, 0.82, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.42 * scale
    for d in sorted(detections, key=lambda d: d['reliable']):
        color = CATEGORY_COLORS.get(d['category'], DEFAULT_COLOR) if d['reliable'] else DEFAULT_COLOR
        x1, y1, x2, y2 = d['box']
        cl = max(4, min(18, (x2 - x1) // 3, (y2 - y1) // 3))
        if d['reliable'] and d['conf'] >= LOW_CONFIDENCE:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if d['category'] == 'critical' else 2)
        else:
            _corner_marks(out, d['box'], color, cl, 2)

        label = finding_label(d)
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        ty = y1 - 6 if y1 > th + 12 else y2 + th + 10
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, ty), font, fs, (255, 255, 255) if d['reliable'] else (20, 20, 20),
                    1, cv2.LINE_AA)

    # Legend
    items = [(CATEGORY_COLORS[k], CATEGORY_LABELS[k]) for k in CATEGORY_COLORS] + [(DEFAULT_COLOR, 'Unvalidated class')]
    row = int(22 * scale)
    caption = 'Corners only = low confidence / unvalidated   ~ = estimated tooth'
    (cw, _), _ = cv2.getTextSize(caption, font, fs * 0.8, 1)
    lx, ly = 10, h - 10 - (len(items) + 1) * row
    cv2.rectangle(out, (lx - 5, ly - 5), (lx + cw + 5, h - 4), (0, 0, 0), -1)
    for i, (col, lbl) in enumerate(items):
        yp = ly + i * row + int(14 * scale)
        cv2.rectangle(out, (lx, yp - int(10 * scale)), (lx + int(16 * scale), yp + 2), col, -1)
        cv2.putText(out, lbl, (lx + int(22 * scale), yp), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, caption, (lx, ly + len(items) * row + int(12 * scale)), font, fs * 0.8,
                (190, 190, 190), 1, cv2.LINE_AA)
    return out
