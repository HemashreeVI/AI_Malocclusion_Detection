from dental_ai.fdi import (LOWER_ROW, UPPER_ROW, assign_tooth, estimate_tooth_number,
                           generate_fdi_chart, tooth_status)
from dental_ai.findings import make_finding

W, H = 2000, 1000


def box_at(xf, yf, size=40):
    cx, cy = xf * W, yf * H
    return [cx - size, cy - size, cx + size, cy + size]


def test_chart_rows_follow_panoramic_orientation():
    # Viewer's left is the patient's right in both rows.
    assert UPPER_ROW == [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    assert LOWER_ROW == [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def test_quadrants():
    assert estimate_tooth_number(box_at(0.3, 0.35), W, H) // 10 == 1
    assert estimate_tooth_number(box_at(0.7, 0.35), W, H) // 10 == 2
    assert estimate_tooth_number(box_at(0.7, 0.65), W, H) // 10 == 3
    assert estimate_tooth_number(box_at(0.3, 0.65), W, H) // 10 == 4


def test_incisors_near_midline_and_molars_at_edges():
    assert estimate_tooth_number(box_at(0.49, 0.35), W, H) == 11
    assert estimate_tooth_number(box_at(0.51, 0.65), W, H) == 31
    assert estimate_tooth_number(box_at(0.13, 0.35), W, H) == 18
    assert estimate_tooth_number(box_at(0.87, 0.65), W, H) == 38


def test_position_increases_away_from_midline():
    xs = [0.5 + i * 0.03 for i in range(13)]
    nums = [estimate_tooth_number(box_at(x, 0.35), W, H) % 10 for x in xs]
    assert nums == sorted(nums)
    assert nums[0] == 1 and nums[-1] == 8


def test_curved_occlusal_plane():
    # At the far posterior the occlusal plane is higher than at the midline,
    # so y=0.49 is lower jaw near the back but upper jaw near the front.
    assert estimate_tooth_number(box_at(0.15, 0.49), W, H) // 10 == 4
    assert estimate_tooth_number(box_at(0.48, 0.49), W, H) // 10 == 1


def test_anatomy_classes_get_no_tooth():
    assert assign_tooth('maxillary sinus', box_at(0.3, 0.2), W, H) == (None, None)


def test_tooth_detector_takes_priority():
    box = box_at(0.3, 0.35)
    tooth_boxes = [(16, [box[0] - 5, box[1] - 5, box[2] + 5, box[3] + 5])]
    assert assign_tooth('Caries', box, W, H, tooth_boxes) == (16, 'detector')
    # No overlap → falls back to the estimate.
    assert assign_tooth('Caries', box, W, H, [(36, [0, 0, 10, 10])])[1] == 'estimated'


def test_tooth_status_keeps_worst_severity_and_ignores_unvalidated():
    dets = [make_finding('Caries', 0.9, [0, 0, 1, 1], 16, 'estimated'),     # moderate
            make_finding('Filling', 0.95, [0, 0, 1, 1], 16, 'estimated'),   # existing, none
            make_finding('Bone Loss', 0.8, [0, 0, 1, 1], 16, 'estimated'),  # high
            make_finding('Fracture teeth', 0.9, [0, 0, 1, 1], 21, 'estimated')]  # unvalidated
    st = tooth_status(dets)
    assert st[16]['severity'] == 'high' and st[16]['existing']
    assert st[21]['severity'] == 'none' and st[21]['unvalidated']


def test_chart_renders():
    img = generate_fdi_chart([make_finding('Caries', 0.9, [0, 0, 1, 1], 36, 'estimated')])
    assert img.size == (860, 300)
