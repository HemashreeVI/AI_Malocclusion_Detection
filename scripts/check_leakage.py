"""Find the same radiograph appearing in more than one split, and optionally
re-split the dataset so that each source image lives in exactly one split.

    python scripts/check_leakage.py --data dental.yaml
    python scripts/check_leakage.py --data dental.yaml --resplit splits/   # writes train/val/test .txt lists

Two images are grouped when either
* they share a source name (Roboflow exports augmented copies as
  ``<name>_jpg.rf.<hash>.jpg``), or
* their 64-bit difference hashes are within --max-distance bits (near
  duplicates: re-exports, crops, brightness changes).

Without patient IDs this cannot catch two *different* radiographs of the same
patient; if you have patient IDs, group by them instead.
"""
import argparse
import os
import random
import re
from collections import defaultdict

from PIL import Image

from dataset_utils import list_images, load_data_yaml

RF_SUFFIX = re.compile(r'\.rf\.[0-9a-f]+$', re.I)
EXT_TAG = re.compile(r'_(jpe?g|png|bmp|tiff?)$', re.I)


def source_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return EXT_TAG.sub('', RF_SUFFIX.sub('', stem)).lower()


def dhash(path, size=8):
    img = Image.open(path).convert('L').resize((size + 1, size), Image.BILINEAR)
    px = img.tobytes()                       # one byte per pixel in mode L
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def group_images(paths, max_distance=6, use_hash=True):
    uf = UnionFind(len(paths))
    by_source = defaultdict(list)
    for i, p in enumerate(paths):
        by_source[source_name(p)].append(i)
    for idx in by_source.values():
        for j in idx[1:]:
            uf.union(idx[0], j)

    if use_hash:
        hashes = [dhash(p) for p in paths]
        # LSH: two hashes within 6 bits share at least one of 8 exact 8-bit bands.
        buckets = defaultdict(list)
        for i, h in enumerate(hashes):
            for band in range(8):
                buckets[(band, (h >> (band * 8)) & 0xFF)].append(i)
        for members in buckets.values():
            if len(members) > 2000:          # degenerate bucket (e.g. blank images); skip
                continue
            for a_pos, a in enumerate(members):
                for b in members[a_pos + 1:]:
                    if bin(hashes[a] ^ hashes[b]).count('1') <= max_distance:
                        uf.union(a, b)

    groups = defaultdict(list)
    for i in range(len(paths)):
        groups[uf.find(i)].append(i)
    return list(groups.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--max-distance', type=int, default=6)
    ap.add_argument('--no-hash', action='store_true', help='only group by source file name')
    ap.add_argument('--resplit', metavar='DIR', help='write leakage-free train/val/test lists here')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    _, _, splits = load_data_yaml(args.data)
    paths, split_of = [], []
    for split, sp in splits.items():
        imgs = list_images(sp)
        paths += imgs
        split_of += [split] * len(imgs)
    print(f'{len(paths)} images in splits: ' + ', '.join(f'{s}={split_of.count(s)}' for s in splits))

    groups = group_images(paths, args.max_distance, not args.no_hash)
    leaking = [g for g in groups if len({split_of[i] for i in g}) > 1]
    n_leak_imgs = sum(len(g) for g in leaking)
    print(f'{len(groups)} source groups; {len(leaking)} groups span more than one split '
          f'({n_leak_imgs} images, {n_leak_imgs / max(1, len(paths)):.1%} of the dataset)')
    for g in leaking[:10]:
        print('  ' + ' | '.join(f'{split_of[i]}:{os.path.basename(paths[i])}' for i in g[:4]))

    if args.resplit:
        sizes = {s: split_of.count(s) / len(paths) for s in splits}
        rng = random.Random(args.seed)
        rng.shuffle(groups)
        targets = {s: sizes[s] * len(paths) for s in splits}
        assigned = {s: [] for s in splits}
        for g in sorted(groups, key=len, reverse=True):          # big groups first, then fill
            split = max(targets, key=lambda s: targets[s] - len(assigned[s]))
            assigned[split] += [paths[i] for i in g]
        os.makedirs(args.resplit, exist_ok=True)
        for s, items in assigned.items():
            with open(os.path.join(args.resplit, f'{s}.txt'), 'w') as f:
                f.write('\n'.join(items) + '\n')
            print(f'{s}: {len(items)} images → {os.path.join(args.resplit, s + ".txt")}')
        print('Point train/val/test in your data YAML at these .txt files.')


if __name__ == '__main__':
    main()
