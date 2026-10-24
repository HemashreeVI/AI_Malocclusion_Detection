"""FDI tooth numbering and the tooth chart image.

Panoramic radiographs are viewed as if facing the patient, so the viewer's
left is the patient's right:

    viewer left  ── upper: 18 … 11 | 21 … 28 ── viewer right
                    lower: 48 … 41 | 31 … 38

Two ways to assign a finding to a tooth:

1. ``assign_from_tooth_boxes`` — preferred. Uses boxes from a tooth-numbering
   detector (e.g. a YOLO model whose class names are FDI numbers) and picks the
   tooth that overlaps the finding most.
2. ``estimate_tooth_number`` — fallback heuristic based on position in the
   image. It accounts for unequal tooth widths and the curved occlusal plane,
   but it is still an estimate and is labelled as such in the UI and report.
"""
from bisect import bisect_left
from itertools import accumulate

from PIL import Image, ImageDraw, ImageFont

from .config import NON_TOOTH_CLASSES, SEVERITY_ORDER, CLASS_CATEGORY

# Horizontal span of the dentition in a typical panoramic image (fraction of
# image width). Outside this range are rami/condyles.
ARCH_X_RANGE = (0.12, 0.88)
# Occlusal plane: a "smile" curve, lowest at the midline.
OCCLUSAL_CENTER_Y = 0.52
OCCLUSAL_CURVE = 0.08

# Average mesio-distal crown widths in mm, central incisor (1) → third molar (8).
UPPER_WIDTHS = [8.5, 6.5, 7.5, 7.0, 6.5, 10.0, 9.0, 8.5]
LOWER_WIDTHS = [5.0, 5.5, 7.0, 7.0, 7.0, 11.0, 10.5, 10.0]


def _cumulative_fractions(widths):
    total = sum(widths)
    return [c / total for c in accumulate(widths)]


_UPPER_CUM = _cumulative_fractions(UPPER_WIDTHS)
_LOWER_CUM = _cumulative_fractions(LOWER_WIDTHS)


def occlusal_y(x_frac):
    """Height (fraction of image height) of the occlusal plane at x_frac."""
    return OCCLUSAL_CENTER_Y - OCCLUSAL_CURVE * ((x_frac - 0.5) * 2) ** 2


def estimate_tooth_number(box, img_width, img_height):
    """Heuristic FDI number for a box ``[x1, y1, x2, y2]`` in pixel units."""
    cx = (box[0] + box[2]) / 2 / img_width
    cy = (box[1] + box[3]) / 2 / img_height
    upper = cy < occlusal_y(cx)
    viewer_left = cx < 0.5

    half_span = 0.5 - ARCH_X_RANGE[0]
    dist = min(max(abs(cx - 0.5) / half_span, 0.0), 1.0)
    position = min(bisect_left(_UPPER_CUM if upper else _LOWER_CUM, dist), 7) + 1

    if upper:
        quadrant = 1 if viewer_left else 2
    else:
        quadrant = 4 if viewer_left else 3
    return quadrant * 10 + position


def _overlap_fraction(a, b):
    """Area of a∩b divided by area of a."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area_a


def assign_from_tooth_boxes(box, tooth_boxes, min_overlap=0.3):
    """Return the FDI number of the tooth box that covers most of ``box``,
    or None. ``tooth_boxes`` is a list of ``(fdi_number, [x1, y1, x2, y2])``."""
    best, best_frac = None, min_overlap
    for fdi, tbox in tooth_boxes:
        frac = _overlap_fraction(box, tbox)
        if frac >= best_frac:
            best, best_frac = fdi, frac
    return best


def assign_tooth(cls, box, img_width, img_height, tooth_boxes=None):
    """Returns ``(tooth_number_or_None, source)`` where source is
    'detector', 'estimated' or None."""
    if cls in NON_TOOTH_CLASSES:
        return None, None
    if tooth_boxes:
        fdi = assign_from_tooth_boxes(box, tooth_boxes)
        if fdi is not None:
            return fdi, 'detector'
    return estimate_tooth_number(box, img_width, img_height), 'estimated'


# ── Chart ─────────────────────────────────────────────────────
UPPER_ROW = list(range(18, 10, -1)) + list(range(21, 29))   # 18→11, 21→28
LOWER_ROW = list(range(48, 40, -1)) + list(range(31, 39))   # 48→41, 31→38

SEVERITY_COLORS = {
    'high':     (230, 60, 60),
    'moderate': (245, 150, 20),
    'low':      (235, 205, 40),
}
EXISTING_COLOR = (60, 140, 230)
EMPTY_FILL = (40, 40, 60)
EMPTY_BORDER = (85, 85, 105)
TEXT = (235, 235, 240)
MUTED = (150, 150, 165)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def tooth_status(detections):
    """Per tooth: worst severity among reliable findings, plus flags.
    Returns ``{tooth: {'severity': str, 'existing': bool, 'unvalidated': bool}}``."""
    status = {}
    for d in detections:
        t = d.get('tooth')
        if t is None:
            continue
        s = status.setdefault(t, {'severity': 'none', 'existing': False, 'unvalidated': False})
        if not d['reliable']:
            s['unvalidated'] = True
            continue
        if CLASS_CATEGORY.get(d['class']) == 'existing':
            s['existing'] = True
        if SEVERITY_ORDER[d['severity']] > SEVERITY_ORDER[s['severity']]:
            s['severity'] = d['severity']
    return status


def generate_fdi_chart(detections):
    W, H = 860, 300
    img = Image.new('RGB', (W, H), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    f_num, f_lbl, f_small = _font(13), _font(14), _font(12)
    status = tooth_status(detections)

    left, size, gap, mid_gap = 80, 38, 6, 18
    xs = []
    for i in range(16):
        x = left + i * (size + gap) + (mid_gap if i >= 8 else 0)
        xs.append(x)
    midline_x = (xs[7] + size + xs[8]) // 2

    def draw_row(teeth, y, label):
        draw.text((10, y + 11), label, fill=(150, 200, 255), font=f_lbl)
        for x, t in zip(xs, teeth):
            s = status.get(t)
            fill, border, width = EMPTY_FILL, EMPTY_BORDER, 1
            if s:
                if s['severity'] != 'none':
                    fill, border, width = SEVERITY_COLORS[s['severity']], (255, 255, 255), 2
                elif s['existing']:
                    border, width = EXISTING_COLOR, 3
            draw.rectangle([x, y, x + size, y + size], fill=fill, outline=border, width=width)
            draw.text((x + 8, y + 11), str(t), fill=TEXT, font=f_num)
            if s and s['unvalidated']:
                draw.text((x + size - 10, y + 1), '?', fill=TEXT, font=f_small)

    draw_row(UPPER_ROW, 40, 'UPPER')
    draw_row(LOWER_ROW, 120, 'LOWER')
    draw.line([midline_x, 30, midline_x, 168], fill=(110, 110, 130), width=1)
    draw.text((xs[0], 14), "Patient's right", fill=MUTED, font=f_small)
    draw.text((xs[15] - 50, 14), "Patient's left", fill=MUTED, font=f_small)

    # Legend
    ly = 190
    items = [(SEVERITY_COLORS['high'], 'High severity', True),
             (SEVERITY_COLORS['moderate'], 'Moderate', True),
             (SEVERITY_COLORS['low'], 'Low', True),
             (EXISTING_COLOR, 'Existing work', False)]
    for i, (col, lbl, filled) in enumerate(items):
        x = 10 + i * 170
        if filled:
            draw.rectangle([x, ly, x + 18, ly + 18], fill=col)
        else:
            draw.rectangle([x, ly, x + 18, ly + 18], outline=col, width=3)
        draw.text((x + 26, ly + 2), lbl, fill=TEXT, font=f_small)
    draw.text((10, ly + 32), "? = finding from a class the model has not been validated on",
              fill=MUTED, font=f_small)
    draw.text((10, ly + 52), "Tooth numbers are estimated from position unless a tooth detector is configured.",
              fill=MUTED, font=f_small)
    draw.text((W - 230, ly + 80), 'FDI two-digit notation', fill=(110, 110, 130), font=f_small)
    return img
