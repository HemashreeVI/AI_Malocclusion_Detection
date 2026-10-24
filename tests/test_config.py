from dental_ai import config


def test_falls_back_to_training_counts_without_evaluation(monkeypatch):
    monkeypatch.setattr(config, 'GATING', 'train_instances')
    assert config.is_reliable('Caries')                 # 7,357 examples
    assert not config.is_reliable('Malaligned')         # 7 examples
    assert not config.is_reliable('unknown class')
    assert config.reliability_note('Malaligned') == '7 training examples'


def test_uses_per_class_ap_when_evaluated(monkeypatch):
    monkeypatch.setattr(config, 'GATING', 'ap50')
    monkeypatch.setattr(config, 'CLASS_AP50', {'Caries': 0.12, 'Malaligned': 0.6, 'Cyst': 1.0})
    monkeypatch.setattr(config, 'TEST_INSTANCES', {'Caries': 900, 'Malaligned': 40, 'Cyst': 1})
    assert not config.is_reliable('Caries')
    assert config.is_reliable('Malaligned')
    assert not config.is_reliable('Cyst')               # perfect AP on a single box proves nothing
    assert not config.is_reliable('TAD')                # absent from the test set
    assert config.reliability_note('Caries') == 'AP@50 12% on 900 test boxes'
    assert config.reliability_note('TAD') == 'not in test set'
    assert 'AP@50' in config.gating_rule()


def test_every_category_class_has_training_count():
    classes = {c for members in config.CATEGORIES.values() for c in members}
    assert classes == set(config.TRAIN_INSTANCES)
