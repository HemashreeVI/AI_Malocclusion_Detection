"""Explanation maps for YOLO detections.

``DetectionCAM.explain`` computes a gradient-based map for individual
detections using HiResCAM (Draelos & Carin, 2020), a Grad-CAM variant that
multiplies gradients and activations element-wise instead of first averaging
the gradients per channel:

1. Letterbox the image exactly like ultralytics does (640, pad 114).
2. Forward through the raw network with hooks on the feature maps that feed
   the Detect head (one per stride: P3/P4/P5).
3. For each detection, find the anchor whose decoded box overlaps it best and
   whose score for that class is highest, and back-propagate that single class
   score.
4. On the feature level that anchor belongs to:
   ReLU(Σ_c ∂score/∂A_c ⊙ A_c), mapped back to the original image.

Why not plain Grad-CAM: on COCO-pretrained YOLOv9t (ultralytics' bus/zidane
sample images, 8 detections) plain Grad-CAM put its peak inside the detected
box for 3/8 detections, with 2.5× more heat in the box than its area share;
HiResCAM got 8/8 and 14×. See tests/test_cam.py::test_localises_pretrained.

If there are no detections (or gradients are unavailable), it falls back to
EigenCAM — the first principal component of the deepest feature map. EigenCAM
is class-agnostic and is labelled as such.
"""
import cv2
import numpy as np
import torch

IMG_SIZE = 640
PAD_VALUE = 114


def letterbox(img_rgb, size=IMG_SIZE):
    """Resize keeping aspect ratio and pad to ``size``×``size``.
    Returns ``(image, ratio, (pad_x, pad_y))``."""
    h, w = img_rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (size - nw) / 2, (size - nh) / 2
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(py - 0.1)), int(round(py + 0.1))
    left, right = int(round(px - 0.1)), int(round(px + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(PAD_VALUE,) * 3)
    return out, r, (left, top)


def to_letterbox_coords(box, ratio, pad):
    return [box[0] * ratio + pad[0], box[1] * ratio + pad[1],
            box[2] * ratio + pad[0], box[3] * ratio + pad[1]]


def unletterbox_map(cam, shape, ratio, pad):
    """``cam`` is a map over the letterboxed square; return it at ``shape``."""
    h, w = shape[:2]
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    cam = cam[pad[1]:pad[1] + nh, pad[0]:pad[0] + nw]
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)


def _box_iou(box, boxes):
    """IoU of one xyxy box against an (N, 4) tensor."""
    b = torch.as_tensor(box, dtype=boxes.dtype, device=boxes.device)
    ix = (torch.minimum(boxes[:, 2], b[2]) - torch.maximum(boxes[:, 0], b[0])).clamp(min=0)
    iy = (torch.minimum(boxes[:, 3], b[3]) - torch.maximum(boxes[:, 1], b[1])).clamp(min=0)
    inter = ix * iy
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _normalize(cam):
    cam = np.maximum(cam, 0).astype(np.float64)    # gradients can be tiny; scale-free check
    span = cam.max() - cam.min()
    return ((cam - cam.min()) / span if span > 0 else np.zeros_like(cam)).astype(np.float32)


class DetectionCAM:
    def __init__(self, yolo):
        self.yolo = yolo
        self.names_to_idx = {v: k for k, v in yolo.names.items()}

    @property
    def net(self):
        # Re-read each time: YOLO.predict() may replace/fuse the module.
        return self.yolo.model

    def _forward(self, img_rgb):
        net = self.net
        net.eval()
        detect = net.model[-1]
        feats = {}
        hooks = [net.model[i].register_forward_hook(
                     lambda m, inp, out, i=i: feats.__setitem__(i, out))
                 for i in detect.f]
        lb, ratio, pad = letterbox(img_rgb)
        device = next(net.parameters()).device
        x = torch.from_numpy(lb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        x.requires_grad_(True)            # makes every activation part of the graph
        try:
            out = net(x)
        finally:
            for h in hooks:
                h.remove()
        preds = out[0] if isinstance(out, (list, tuple)) else out
        levels = [feats[i] for i in detect.f]
        # Grad-CAM needs per-anchor scores (B, 4 + nc, A). End-to-end heads
        # return NMS-free top-k boxes instead; those fall back to EigenCAM.
        if not (torch.is_tensor(preds) and preds.dim() == 3 and preds.shape[1] == 4 + detect.nc):
            preds = None
        return preds, levels, ratio, pad

    def _cam_one(self, det, preds, levels, ratio, pad):
        """HiResCAM map (feature-grid space) for one detection, or None."""
        cls_idx = self.names_to_idx[det['class']]
        p = preds[0]                                   # (4 + nc, A)
        xywh = p[:4].T
        boxes = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
        target = to_letterbox_coords(det['box'], ratio, pad)
        iou = _box_iou(target, boxes.detach())
        scores = p[4 + cls_idx]
        cand = (iou.detach() * scores.detach())
        if float(cand.max()) <= 0:
            return None
        a = int(cand.argmax())

        sizes = [f.shape[2] * f.shape[3] for f in levels]
        level, offset = 0, 0
        while a >= offset + sizes[level]:
            offset += sizes[level]
            level += 1
        feat = levels[level]

        grads = torch.autograd.grad(scores[a], feat, retain_graph=True)[0]
        cam = torch.relu((grads * feat).sum(1))[0].detach().cpu().numpy()
        return cam if cam.max() > 0 else None

    @staticmethod
    def _eigencam(feat):
        a = feat[0].detach().float().cpu()
        c, h, w = a.shape
        m = a.reshape(c, h * w).T                       # (HW, C)
        m = m - m.mean(0)
        _, _, vh = torch.linalg.svd(m, full_matrices=False)
        proj = (m @ vh[0]).reshape(h, w).numpy()
        if proj.max() < -proj.min():                    # SVD sign is arbitrary
            proj = -proj
        return proj

    def explain(self, img_rgb, detections, max_dets=8):
        """Returns ``(heat_map in [0,1] at image size, method_note, explained_detections)``.
        Combines (pixel-wise max) the Grad-CAM maps of up to ``max_dets``
        detections, highest confidence first."""
        with torch.enable_grad():
            preds, levels, ratio, pad = self._forward(img_rgb)
            maps, used = [], []
            for d in sorted(detections, key=lambda d: -d['conf'])[:max_dets] if preds is not None else []:
                try:
                    cam = self._cam_one(d, preds, levels, ratio, pad)
                except RuntimeError:
                    cam = None
                if cam is not None:
                    maps.append(_normalize(unletterbox_map(cam, img_rgb.shape, ratio, pad)))
                    used.append(d)

        if maps:
            heat = np.max(np.stack(maps), axis=0)
            n = len(maps)
            note = (f'HiResCAM for {n} detection{"s" if n > 1 else ""}: regions whose features '
                    f'most increased the class score of the explained detection(s).')
            return heat, note, used

        cam = self._eigencam(levels[-1])
        heat = _normalize(unletterbox_map(cam, img_rgb.shape, ratio, pad))
        return heat, ('EigenCAM (class-agnostic): dominant pattern in the deepest feature map. '
                      'It does not explain any specific finding.'), []


def overlay_heatmap(img_rgb, heat, detections=(), title=''):
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    alpha = (0.15 + 0.45 * np.clip(heat, 0, 1))[..., None]          # cold regions stay readable
    out = (alpha * colored + (1 - alpha) * img_rgb).astype(np.uint8)
    for d in detections:
        x1, y1, x2, y2 = d['box']
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
    if title:
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out
