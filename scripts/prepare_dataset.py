"""Build a merged / rebalanced copy of a YOLO dataset.

    python scripts/prepare_dataset.py --data dental.yaml --config configs/class_merge.yaml --out data/merged

* Remaps class indices according to the merge plan; unlisted classes are dropped.
* Images are symlinked (no copies); label files are rewritten.
* Writes ``train.txt`` in which images containing under-represented classes
  are repeated (oversampling), plus ``val.txt``/``test.txt`` and ``data.yaml``.
* If --splits points at the output of check_leakage.py --resplit, those
  leakage-free lists are used instead of the YAML's splits.
"""
import argparse
import math
import os
from collections import Counter

import yaml

from dataset_utils import list_images, load_data_yaml, read_labels


def build_mapping(names, plan):
    """old index → new index (or None to drop); returns (mapping, new_names)."""
    by_name = {v: k for k, v in names.items()}
    new_names, mapping = [], {k: None for k in names}
    for new_idx, (new_name, olds) in enumerate(plan.items()):
        new_names.append(new_name)
        for old in olds:
            if old not in by_name:
                raise SystemExit(f'Unknown class in merge plan: {old!r}')
            if mapping[by_name[old]] is not None:
                raise SystemExit(f'Class {old!r} is listed twice in the merge plan')
            mapping[by_name[old]] = new_idx
    return mapping, new_names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--config', default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'class_merge.yaml'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--splits', help='directory with train/val/test .txt lists from check_leakage.py --resplit')
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    _, names, splits = load_data_yaml(args.data)
    mapping, new_names = build_mapping(names, cfg['classes'])
    dropped = [names[k] for k, v in mapping.items() if v is None]
    print(f'{len(names)} → {len(new_names)} classes. Dropped: {", ".join(dropped) or "none"}')

    if args.splits:
        splits = {s: os.path.join(args.splits, f'{s}.txt') for s in ('train', 'val', 'test')
                  if os.path.exists(os.path.join(args.splits, f'{s}.txt'))}

    out = os.path.abspath(args.out)
    per_split_images, train_counts, train_classes_of = {}, Counter(), {}
    for split, sp in splits.items():
        written = []
        for i, img in enumerate(list_images(sp)):
            name = f'{i:06d}_{os.path.basename(img)}'
            dst_img = os.path.join(out, 'images', split, name)
            dst_lbl = os.path.join(out, 'labels', split, os.path.splitext(name)[0] + '.txt')
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
            if not os.path.lexists(dst_img):
                os.symlink(os.path.abspath(img), dst_img)
            lines, classes = [], set()
            for parts in read_labels(img):
                new = mapping.get(int(parts[0]))
                if new is not None:
                    lines.append(' '.join([str(new)] + parts[1:]))
                    classes.add(new)
                    if split == 'train':
                        train_counts[new] += 1
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            written.append(dst_img)
            if split == 'train':
                train_classes_of[dst_img] = classes
        per_split_images[split] = written

    # Oversampling
    os_cfg = cfg.get('oversample') or {}
    min_inst, max_rep = os_cfg.get('min_instances', 0), os_cfg.get('max_repeat', 1)
    repeat = {c: min(max_rep, math.ceil(min_inst / n)) if 0 < n < min_inst else 1
              for c, n in train_counts.items()}
    train_list = []
    for img in per_split_images.get('train', []):
        train_list += [img] * max([repeat.get(c, 1) for c in train_classes_of[img]] + [1])

    for split, items in per_split_images.items():
        items = train_list if split == 'train' else items
        with open(os.path.join(out, f'{split}.txt'), 'w') as f:
            f.write('\n'.join(items) + '\n')
    data = {'path': out, 'names': dict(enumerate(new_names))}
    data.update({s: f'{s}.txt' for s in per_split_images})
    with open(os.path.join(out, 'data.yaml'), 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print('\nTraining boxes per class (before oversampling) and repeat factor:')
    for c, n in enumerate(new_names):
        print(f'  {n:<24} {train_counts[c]:>7}   ×{repeat.get(c, 1)}')
    print(f'\nTrain list: {len(per_split_images.get("train", []))} images → {len(train_list)} entries')
    print(f'Dataset YAML: {os.path.join(out, "data.yaml")}')


if __name__ == '__main__':
    main()
