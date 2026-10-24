"""Turning raw detections into findings: severity, confidence band, risk
score, cost estimate and orthodontic discussion points."""
from .config import (CLASS_CATEGORY, CLASS_SEVERITY, HIGH_CONFIDENCE, LOW_CONFIDENCE,
                     RISK_WEIGHTS, SEVERITY_ORDER, TREATMENT_COSTS, is_reliable)


def get_severity(cls):
    """Clinical severity of a class. Does not depend on model confidence."""
    return CLASS_SEVERITY.get(cls, 'none')


def confidence_band(conf):
    if conf >= HIGH_CONFIDENCE:
        return 'high'
    if conf >= LOW_CONFIDENCE:
        return 'moderate'
    return 'low'


def make_finding(cls, conf, box, tooth, tooth_source):
    return {
        'class': cls,
        'category': CLASS_CATEGORY.get(cls, 'other'),
        'conf': conf,
        'confidence': confidence_band(conf),
        'box': box,
        'tooth': tooth,
        'tooth_source': tooth_source,
        'severity': get_severity(cls),
        'reliable': is_reliable(cls),
    }


def actionable(detections):
    """Findings that may drive risk, cost and recommendations: reliable
    pathology classes only."""
    return [d for d in detections if d['reliable'] and d['class'] in RISK_WEIGHTS]


def risk_score(detections):
    """0–100. Sum of class weight × confidence over actionable findings,
    counting each (class, tooth) once at its highest confidence."""
    best = {}
    for d in actionable(detections):
        key = (d['class'], d['tooth'])
        best[key] = max(best.get(key, 0.0), d['conf'])
    return min(100.0, sum(RISK_WEIGHTS[cls] * conf for (cls, _), conf in best.items()))


def risk_status(score):
    if score >= 70:
        return 'URGENT', 'See a dentist as soon as possible'
    if score >= 40:
        return 'MODERATE', 'Schedule an appointment soon'
    if score >= 15:
        return 'LOW', 'Routine check-up recommended'
    return 'NO MAJOR FINDINGS', 'Continue regular check-ups'


def cost_items(detections):
    """One row per treatable class with the number of affected teeth.
    Returns ``(rows, total_min, total_max)``."""
    teeth_by_class = {}
    for d in actionable(detections):
        if d['class'] in TREATMENT_COSTS:
            teeth_by_class.setdefault(d['class'], set()).add(d['tooth'])

    rows, total_min, total_max = [], 0, 0
    order = sorted(teeth_by_class, key=lambda c: (-SEVERITY_ORDER[get_severity(c)], -RISK_WEIGHTS[c]))
    for cls in order:
        info = TREATMENT_COSTS[cls]
        n = len(teeth_by_class[cls])
        lo, hi = info['min'] * n, info['max'] * n
        rows.append({'class': cls, 'teeth': n, 'treatment': info['treatment'],
                     'min': lo, 'max': hi, 'urgency': info['urgency']})
        total_min += lo
        total_max += hi
    return rows, total_min, total_max


# ── Orthodontic discussion points ─────────────────────────────
# These are options a clinician may consider. A panoramic radiograph alone
# cannot establish malocclusion type or severity (that needs a clinical
# exam, cephalometrics and/or study models), so options are not gated on a
# made-up severity.
ORTHO_NOTES = {
    'Malaligned': {
        'title': 'Possible tooth malalignment',
        'note': 'Needs clinical exam and bite records to classify (e.g. Angle class, crowding, spacing).',
        'options': [
            ('Clear aligners', 'Often used for mild–moderate crowding or spacing.'),
            ('Fixed braces (metal / ceramic / self-ligating)', 'Suitable across a wide range of severities.'),
            ('Lingual braces', 'Fixed appliance placed behind the teeth.'),
        ],
    },
    'impacted tooth': {
        'title': 'Possible impacted tooth',
        'note': 'Management depends on which tooth (e.g. third molar vs canine) and its position.',
        'options': [
            ('Monitoring', 'Asymptomatic third molars are often monitored.'),
            ('Surgical removal', 'Common for problematic third molars.'),
            ('Surgical exposure + fixed braces', 'Used to guide impacted canines into the arch.'),
        ],
    },
    'Supra Eruption': {
        'title': 'Possible supra-eruption',
        'note': 'Usually secondary to a missing opposing tooth.',
        'options': [
            ('Restore the opposing space', 'Prevents further over-eruption.'),
            ('Orthodontic intrusion', 'Fixed appliances or aligners with attachments.'),
        ],
    },
    'Missing teeth': {
        'title': 'Possible missing teeth',
        'note': 'Confirm clinically; teeth may be unerupted or previously extracted.',
        'options': [
            ('Implant or bridge', 'Replaces the missing tooth.'),
            ('Orthodontic space management', 'Open or close the space before replacement.'),
        ],
    },
    'attrition': {
        'title': 'Possible tooth wear (attrition)',
        'note': 'Often related to bruxism; bite assessment needed.',
        'options': [
            ('Occlusal night guard', 'Protects against further wear.'),
            ('Restorative rehabilitation', 'For advanced wear.'),
        ],
    },
}


def ortho_discussion_points(detections):
    """Discussion points for reliable orthodontic findings, plus a list of
    orthodontic classes that were detected but are not validated."""
    points, unvalidated = [], set()
    for cls, info in ORTHO_NOTES.items():
        found = [d for d in detections if d['class'] == cls]
        if not found:
            continue
        if not found[0]['reliable']:
            unvalidated.add(cls)
            continue
        teeth = sorted({d['tooth'] for d in found if d['tooth'] is not None})
        points.append({**info, 'class': cls, 'teeth': teeth,
                       'max_conf': max(d['conf'] for d in found)})
    return points, sorted(unvalidated)
