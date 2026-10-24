"""Static configuration: class categories, colours, risk weights, costs and
reliability gating.

Everything that is a clinical or business assumption lives here so it can be
reviewed in one place instead of being scattered through the app.
"""
import json
import os

# ── Class categories ──────────────────────────────────────────
# Category describes *what kind of finding* a class is. It is independent of
# how confident the model is about a particular detection.
CATEGORIES = {
    'critical':    ['Cyst', 'Periapical lesion', 'Bone Loss', 'Fracture teeth', 'bone defect'],
    'orthodontic': ['Malaligned', 'impacted tooth', 'Supra Eruption', 'Missing teeth', 'attrition'],
    'restorative': ['Caries', 'Root resorption', 'Retained root', 'Root Piece'],
    'existing':    ['Crown', 'Filling', 'Implant', 'Root Canal Treatment', 'orthodontic brackets',
                    'permanent retainer', 'TAD', 'abutment', 'gingival former', 'metal band',
                    'plating', 'post - core', 'wire'],
    'anatomy':     ['Permanent Teeth', 'Primary teeth', 'Mandibular Canal', 'maxillary sinus'],
}
CLASS_CATEGORY = {cls: cat for cat, members in CATEGORIES.items() for cls in members}

CATEGORY_COLORS = {          # RGB
    'critical':    (255, 50, 50),
    'orthodontic': (255, 140, 0),
    'restorative': (255, 220, 0),
    'existing':    (50, 150, 255),
    'anatomy':     (50, 200, 100),
}
DEFAULT_COLOR = (180, 180, 180)

CATEGORY_LABELS = {
    'critical':    'Critical',
    'orthodontic': 'Orthodontic',
    'restorative': 'Restorative',
    'existing':    'Existing work',
    'anatomy':     'Normal anatomy',
}

# Classes that describe a structure rather than a single tooth; they get no
# FDI tooth number.
NON_TOOTH_CLASSES = {'Mandibular Canal', 'maxillary sinus'}

# ── Severity (clinical, NOT confidence) ───────────────────────
# How serious a finding of this class usually is. Confidence is reported
# separately; a 95%-confident filling is still not a severe finding.
CLASS_SEVERITY = {
    'Cyst': 'high', 'Periapical lesion': 'high', 'Bone Loss': 'high',
    'Fracture teeth': 'high', 'bone defect': 'high', 'impacted tooth': 'high',
    'Root resorption': 'moderate', 'Malaligned': 'moderate', 'Supra Eruption': 'moderate',
    'Caries': 'moderate', 'Missing teeth': 'moderate', 'Retained root': 'moderate',
    'Root Piece': 'moderate',
    'attrition': 'low',
}
SEVERITY_ORDER = {'none': 0, 'low': 1, 'moderate': 2, 'high': 3}

# ── Confidence bands ──────────────────────────────────────────
LOW_CONFIDENCE = 0.45
HIGH_CONFIDENCE = 0.75

# ── Risk score weights ────────────────────────────────────────
# Only pathology contributes. Existing work and normal anatomy weigh 0.
RISK_WEIGHTS = {
    'Cyst': 10, 'Periapical lesion': 9, 'Bone Loss': 9,
    'Fracture teeth': 8, 'bone defect': 8, 'Root resorption': 7,
    'Malaligned': 6, 'impacted tooth': 6, 'Supra Eruption': 5,
    'Caries': 4, 'Missing teeth': 4, 'Retained root': 4, 'Root Piece': 4,
    'attrition': 3,
}

# ── Reliability gating ────────────────────────────────────────
# Unreliable classes are shown as "unvalidated": they still appear in the
# findings list, but they do not drive the risk score, cost estimate or
# treatment discussion points.
#
# Preferred source: class_ap50.json, written by scripts/evaluate.py (per-class
# test-set AP@50). Until that exists, fall back to the number of training
# examples per class (class_stats.json) — a model cannot learn a class from a
# handful of examples.
MIN_RELIABLE_AP50 = float(os.environ.get('MIN_RELIABLE_AP50', 0.30))
MIN_TEST_INSTANCES = int(os.environ.get('MIN_TEST_INSTANCES', 10))   # AP on 1–2 boxes means nothing
MIN_TRAIN_INSTANCES = int(os.environ.get('MIN_TRAIN_INSTANCES', 100))

_HERE = os.path.dirname(__file__)


def _load(name, key):
    try:
        with open(os.path.join(_HERE, name)) as f:
            return json.load(f).get(key, {})
    except (OSError, ValueError):
        return {}


CLASS_AP50 = _load('class_ap50.json', 'ap50')
TEST_INSTANCES = _load('class_ap50.json', 'instances')
TRAIN_INSTANCES = _load('class_stats.json', 'train_instances')
GATING = 'ap50' if CLASS_AP50 else 'train_instances'


def is_reliable(cls):
    """Unknown classes are treated as unreliable."""
    if GATING == 'ap50':
        return (CLASS_AP50.get(cls, 0.0) >= MIN_RELIABLE_AP50
                and TEST_INSTANCES.get(cls, MIN_TEST_INSTANCES) >= MIN_TEST_INSTANCES)
    return TRAIN_INSTANCES.get(cls, 0) >= MIN_TRAIN_INSTANCES


def reliability_note(cls):
    """Short reason shown next to a finding, e.g. 'AP@50 12%' or '7 training examples'."""
    if GATING == 'ap50':
        if cls not in CLASS_AP50:
            return 'not in test set'
        n = TEST_INSTANCES.get(cls)
        return f"AP@50 {CLASS_AP50[cls]:.0%}" + (f" on {n} test boxes" if n is not None else '')
    return f"{TRAIN_INSTANCES.get(cls, 0):,} training examples"


def gating_rule():
    if GATING == 'ap50':
        return f'test-set AP@50 below {MIN_RELIABLE_AP50:.0%} or fewer than {MIN_TEST_INSTANCES} test boxes'
    return f'fewer than {MIN_TRAIN_INSTANCES} training examples'


# ── Treatment cost database (INR, per affected tooth) ─────────
COST_REGION_NOTE = 'Approximate ranges for India (2024); actual costs vary by city, clinic and case.'
TREATMENT_COSTS = {
    'Caries':            {'treatment': 'Composite filling',           'min': 1500,  'max': 5000,   'urgency': 'Within 1 month'},
    'Periapical lesion': {'treatment': 'Root canal + crown',          'min': 8000,  'max': 25000,  'urgency': 'Within 2 weeks'},
    'Cyst':              {'treatment': 'Surgical cyst removal',       'min': 15000, 'max': 50000,  'urgency': 'Prompt referral'},
    'Bone Loss':         {'treatment': 'Periodontal treatment',       'min': 5000,  'max': 30000,  'urgency': 'Within 2 weeks'},
    'Fracture teeth':    {'treatment': 'Crown / extraction',          'min': 5000,  'max': 20000,  'urgency': 'Prompt referral'},
    'bone defect':       {'treatment': 'Bone graft',                  'min': 20000, 'max': 80000,  'urgency': 'Within 1 month'},
    'Malaligned':        {'treatment': 'Orthodontic assessment',      'min': 25000, 'max': 300000, 'urgency': 'Within 3 months'},
    'impacted tooth':    {'treatment': 'Surgical exposure / removal', 'min': 5000,  'max': 90000,  'urgency': 'Within 1 month'},
    'Supra Eruption':    {'treatment': 'Orthodontic / restorative',   'min': 25000, 'max': 80000,  'urgency': 'Within 3 months'},
    'Missing teeth':     {'treatment': 'Implant / bridge',            'min': 25000, 'max': 200000, 'urgency': 'Within 6 months'},
    'Root resorption':   {'treatment': 'RCT or extraction',           'min': 5000,  'max': 20000,  'urgency': 'Within 2 weeks'},
    'Retained root':     {'treatment': 'Extraction',                  'min': 500,   'max': 5000,   'urgency': 'Within 1 month'},
    'Root Piece':        {'treatment': 'Surgical extraction',         'min': 2000,  'max': 8000,   'urgency': 'Within 1 month'},
    'attrition':         {'treatment': 'Night guard + restoration',   'min': 5000,  'max': 40000,  'urgency': 'Within 3 months'},
}
