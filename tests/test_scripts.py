import os
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

import check_leakage
import evaluate
import prepare_dataset


class FakeBox:
    """Mimics ultralytics Metric: results only for classes present in the split."""
    def __init__(self, present, results):
        self.ap_class_index = np.array(present)
        self._results = results

    def class_result(self, i):
        return self._results[i]


def test_per_class_results_follow_ap_class_index():
    names = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    # Class 1 ('B') is absent from the split: positional indexing would give
    # C's results to B and D's to C.
    metrics = SimpleNamespace(nt_per_class=np.array([5, 0, 7, 9]),
                              box=FakeBox([0, 2, 3], [(0.9, 0.8, 0.85, 0.5),
                                                      (0.1, 0.2, 0.15, 0.05),
                                                      (0.6, 0.6, 0.6, 0.3)]))
    rows = {r['class']: r for r in evaluate.per_class_results(metrics, names)}
    assert 'ap50' not in rows['B'] and rows['B']['instances'] == 0
    assert rows['C']['ap50'] == pytest.approx(0.15) and rows['C']['instances'] == 7
    assert rows['D']['ap50'] == pytest.approx(0.6)
    table = evaluate.markdown_table(list(rows.values()))
    assert '| B | 0 | — |' in table


def test_source_name_strips_roboflow_suffix():
    assert check_leakage.source_name('/x/opg12_jpg.rf.a1b2c3d4e5.jpg') == 'opg12'
    assert check_leakage.source_name('/x/OPG12.png') == 'opg12'


def _write(root, split, name, arr, labels):
    os.makedirs(f'{root}/{split}/images', exist_ok=True)
    os.makedirs(f'{root}/{split}/labels', exist_ok=True)
    Image.fromarray(arr).save(f'{root}/{split}/images/{name}.png')
    with open(f'{root}/{split}/labels/{name}.txt', 'w') as f:
        f.write(''.join(f'{c} 0.5 0.5 0.1 0.1\n' for c in labels))


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.RandomState(0)
    imgs = [(rng.rand(32, 64) * 255).astype(np.uint8) for _ in range(6)]
    for i, im in enumerate(imgs[:4]):
        _write(tmp_path, 'train', f'opg{i}_jpg.rf.{i:04x}', im, [0, 1] if i == 0 else [0])
    _write(tmp_path, 'test', 'opg1_jpg.rf.ffff', imgs[1], [0])                    # same source as train
    _write(tmp_path, 'test', 'renamed', np.clip(imgs[2].astype(int) + 5, 0, 255).astype(np.uint8), [2])  # near dup
    _write(tmp_path, 'test', 'clean', imgs[5], [2])
    data = {'path': str(tmp_path), 'train': 'train/images', 'test': 'test/images', 'names': ['Caries', 'Cyst', 'Malaligned']}
    with open(tmp_path / 'data.yaml', 'w') as f:
        yaml.safe_dump(data, f)
    return tmp_path


def test_leakage_detected_and_resplit_is_clean(dataset, capsys):
    check_leakage.main(['--data', str(dataset / 'data.yaml'), '--resplit', str(dataset / 'splits')])
    assert '2 groups span more than one split' in capsys.readouterr().out
    train = set(open(dataset / 'splits' / 'train.txt').read().split())
    test = set(open(dataset / 'splits' / 'test.txt').read().split())
    assert train and test and not train & test
    groups = check_leakage.group_images(sorted(train | test))
    for g in groups:
        assert len({p in train for p in [sorted(train | test)[i] for i in g]}) == 1


def test_prepare_dataset_merges_drops_and_oversamples(dataset, tmp_path):
    cfg = tmp_path / 'merge.yaml'
    with open(cfg, 'w') as f:
        yaml.safe_dump({'classes': {'Caries': ['Caries'], 'Lesion': ['Cyst']},
                        'oversample': {'min_instances': 3, 'max_repeat': 2}}, f)
    out = tmp_path / 'merged'
    prepare_dataset.main(['--data', str(dataset / 'data.yaml'), '--config', str(cfg), '--out', str(out)])
    data = yaml.safe_load(open(out / 'data.yaml'))
    assert data['names'] == {0: 'Caries', 1: 'Lesion'}
    labels = {p.name: p.read_text().split('\n')[:-1] for p in (out / 'labels' / 'test').iterdir()}
    assert all(l == [] for n, l in labels.items() if 'clean' in n or 'renamed' in n)   # Malaligned dropped
    train_list = open(out / 'train.txt').read().split()
    assert len(train_list) == 5                     # image with the rare 'Lesion' class repeated twice
    assert os.path.islink(train_list[0])


def test_merge_plan_rejects_unknown_and_duplicate_classes():
    names = {0: 'A', 1: 'B'}
    with pytest.raises(SystemExit):
        prepare_dataset.build_mapping(names, {'X': ['Z']})
    with pytest.raises(SystemExit):
        prepare_dataset.build_mapping(names, {'X': ['A'], 'Y': ['A']})
