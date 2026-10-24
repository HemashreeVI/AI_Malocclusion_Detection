"""Count labelled instances per class and split, and flag rare classes.

    python scripts/class_stats.py --data dental.yaml [--min-instances 100] [--update-gating]

With --update-gating, writes training counts to dental_ai/class_stats.json
(used by the app when no per-class evaluation is available).
"""
import argparse
import json
import os
from collections import Counter

from dataset_utils import list_images, load_data_yaml, read_labels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count(split_path):
    counts, images = Counter(), Counter()
    for img in list_images(split_path):
        classes = [int(l[0]) for l in read_labels(img)]
        counts.update(classes)
        images.update(set(classes))
    return counts, images


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--min-instances', type=int, default=100)
    ap.add_argument('--update-gating', action='store_true')
    args = ap.parse_args(argv)

    _, names, splits = load_data_yaml(args.data)
    stats = {s: count(p) for s, p in splits.items()}

    header = '| Class | ' + ' | '.join(f'{s} boxes (images)' for s in stats) + ' | |'
    print(header)
    print('|---|' + '---:|' * len(stats) + '---|')
    rare = []
    for c, name in sorted(names.items(), key=lambda kv: -stats.get('train', (Counter(),))[0][kv[0]]):
        cells = [f'{stats[s][0][c]} ({stats[s][1][c]})' for s in stats]
        flag = ''
        if 'train' in stats and stats['train'][0][c] < args.min_instances:
            flag = 'rare'
            rare.append(name)
        print(f'| {name} | ' + ' | '.join(cells) + f' | {flag} |')
    print(f'\n{len(rare)} classes with < {args.min_instances} training boxes: {", ".join(rare)}')

    if args.update_gating and 'train' in stats:
        path = os.path.join(ROOT, 'dental_ai', 'class_stats.json')
        with open(path, 'w') as f:
            json.dump({'_source': f'scripts/class_stats.py --data {args.data}',
                       'train_instances': {n: stats['train'][0][c] for c, n in names.items()}}, f, indent=2)
        print(f'Written: {path}')


if __name__ == '__main__':
    main()
