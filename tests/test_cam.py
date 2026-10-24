import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('ultralytics')

from ultralytics import YOLO  # noqa: E402

from dental_ai.cam import DetectionCAM, letterbox, overlay_heatmap, to_letterbox_coords, unletterbox_map  # noqa: E402


@pytest.fixture(scope='module')
def yolo():
    torch.manual_seed(0)
    return YOLO('yolov9t.yaml')   # random weights, no download


def test_letterbox_roundtrip():
    img = np.zeros((500, 1000, 3), np.uint8)
    lb, r, pad = letterbox(img)
    assert lb.shape == (640, 640, 3)
    assert r == pytest.approx(0.64) and pad == (0, 160)
    assert to_letterbox_coords([0, 0, 1000, 500], r, pad) == pytest.approx([0, 160, 640, 480])
    cam = np.zeros((640, 640), np.float32)
    cam[160:480, :320] = 1                       # left half of the real image
    back = unletterbox_map(cam, img.shape, r, pad)
    assert back.shape == (500, 1000)
    assert back[:, :480].mean() > 0.95 and back[:, 520:].mean() < 0.05


def _fake_detection(cam, img):
    """Build a detection from the model's own top-scoring anchor."""
    preds, _, r, pad = cam._forward(img)
    p = preds[0].detach()
    nc = p.shape[0] - 4
    score, flat = p[4:].max(0)
    a = int(score.argmax())
    cls = int(flat[a])
    x, y, w, h = p[:4, a].tolist()
    box = [(x - w / 2 - pad[0]) / r, (y - h / 2 - pad[1]) / r, (x + w / 2 - pad[0]) / r, (y + h / 2 - pad[1]) / r]
    return {'class': cam.yolo.names[cls], 'conf': float(score[a]), 'box': [int(v) for v in box]}, nc


def test_cam_per_detection(yolo):
    img = (np.random.RandomState(0).rand(400, 800, 3) * 255).astype(np.uint8)
    cam = DetectionCAM(yolo)
    det, nc = _fake_detection(cam, img)
    assert nc == len(yolo.names)
    heat, note, used = cam.explain(img, [det])
    assert heat.shape == img.shape[:2]
    assert 0 <= heat.min() and heat.max() <= 1
    assert note.startswith('HiResCAM')


def test_eigencam_fallback_without_detections(yolo):
    img = (np.random.RandomState(1).rand(300, 600, 3) * 255).astype(np.uint8)
    heat, note, used = DetectionCAM(yolo).explain(img, [])
    assert heat.shape == img.shape[:2]
    assert note.startswith('EigenCAM')
    assert overlay_heatmap(img, heat, title='x').shape == img.shape


def test_works_after_predict_fuses_model(yolo):
    img = (np.random.RandomState(2).rand(400, 800, 3) * 255).astype(np.uint8)
    yolo(img, conf=0.01, verbose=False)          # fuses Conv+BN in place
    cam = DetectionCAM(yolo)
    det, _ = _fake_detection(cam, img)
    heat, note, used = cam.explain(img, [det])
    assert note.startswith('HiResCAM') and heat.shape == img.shape[:2]


def test_localises_pretrained(tmp_path, monkeypatch):
    """With a COCO-pretrained model, each detection's map should peak inside
    its own box. Skipped when the weights cannot be downloaded."""
    import os
    import cv2
    import ultralytics
    monkeypatch.chdir(tmp_path)                      # ultralytics downloads into the cwd
    try:
        model = YOLO('yolov9t.pt')
    except Exception as e:  # offline
        pytest.skip(f'pretrained weights unavailable: {e}')
    cam = DetectionCAM(model)
    assets = os.path.join(os.path.dirname(ultralytics.__file__), 'assets')
    hits = total = 0
    for name in ('bus.jpg', 'zidane.jpg'):
        bgr = cv2.imread(os.path.join(assets, name))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for b in model(bgr, conf=0.4, verbose=False)[0].boxes:
            det = {'class': model.names[int(b.cls[0])], 'conf': float(b.conf[0]),
                   'box': [int(v) for v in b.xyxy[0].tolist()]}
            heat, note, used = cam.explain(rgb, [det])
            y, x = np.unravel_index(heat.argmax(), heat.shape)
            x1, y1, x2, y2 = det['box']
            hits += x1 <= x < x2 and y1 <= y < y2
            total += 1
    assert total >= 6
    assert hits / total >= 0.75
