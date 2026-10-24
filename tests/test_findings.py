from dental_ai.findings import (cost_items, get_severity, make_finding,
                                ortho_discussion_points, risk_score)


def f(cls, conf=0.9, tooth=16):
    return make_finding(cls, conf, [0, 0, 10, 10], tooth, 'estimated')


def test_severity_does_not_depend_on_confidence():
    assert get_severity('Filling') == 'none'
    assert f('Filling', 0.99)['severity'] == 'none'
    assert f('Permanent Teeth', 0.99)['severity'] == 'none'
    assert f('Cyst', 0.30)['severity'] == 'high'


def test_confidence_band_is_separate():
    assert f('Caries', 0.8)['confidence'] == 'high'
    assert f('Caries', 0.5)['confidence'] == 'moderate'
    assert f('Caries', 0.3)['confidence'] == 'low'


def test_healthy_findings_do_not_add_risk():
    healthy = [f('Permanent Teeth', 0.95, t) for t in range(11, 19)]
    healthy += [f('Filling', 0.95, t) for t in range(21, 29)]
    healthy += [f('Crown', 0.95, 36)]
    assert risk_score(healthy) == 0


def test_unvalidated_classes_do_not_add_risk():
    # 'Fracture teeth' had only 10 training examples.
    assert risk_score([f('Fracture teeth', 0.95)]) == 0


def test_risk_counts_each_class_tooth_once():
    once = risk_score([f('Caries', 0.9, 16)])
    assert risk_score([f('Caries', 0.9, 16), f('Caries', 0.6, 16)]) == once
    assert risk_score([f('Caries', 0.9, 16), f('Caries', 0.9, 26)]) == 2 * once


def test_cost_scales_with_number_of_teeth():
    rows, lo, hi = cost_items([f('Caries', 0.9, t) for t in (16, 26, 36, 46, 11)])
    assert rows[0]['class'] == 'Caries' and rows[0]['teeth'] == 5
    assert lo == 5 * 1500 and hi == 5 * 5000


def test_cost_ignores_existing_work_and_unvalidated():
    rows, lo, hi = cost_items([f('Filling'), f('Crown'), f('Supra Eruption')])
    assert rows == [] and lo == hi == 0


def test_ortho_points_only_for_validated_classes():
    points, unvalidated = ortho_discussion_points([f('impacted tooth', 0.9, 38), f('Malaligned', 0.7, 12)])
    assert [p['class'] for p in points] == ['impacted tooth']
    assert points[0]['teeth'] == [38]
    assert unvalidated == ['Malaligned']
