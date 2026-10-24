"""Shared helpers for the dataset scripts (YOLO-format datasets)."""
import glob
import os

import yaml

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def load_data_yaml(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    root = data.get('path') or os.path.dirname(os.path.abspath(path))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(path)), root)
    names = data['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    splits = {}
    for split in ('train', 'val', 'test'):
        if data.get(split):
            entry = data[split]
            splits[split] = entry if os.path.isabs(entry) else os.path.join(root, entry)
    return root, {int(k): v for k, v in names.items()}, splits


def list_images(split_path):
    """Images of a split given as a directory or a .txt list."""
    if split_path.endswith('.txt'):
        with open(split_path) as f:
            return [l.strip() for l in f if l.strip()]
    return sorted(p for p in glob.glob(os.path.join(split_path, '**', '*'), recursive=True)
                  if p.lower().endswith(IMG_EXTS))


def label_path(img_path):
    """ultralytics convention: …/images/… → …/labels/…, extension → .txt"""
    head, tail = img_path.rsplit(f'{os.sep}images{os.sep}', 1)
    return os.path.join(head, 'labels', os.path.splitext(tail)[0] + '.txt')


def read_labels(img_path):
    path = label_path(img_path)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [l.split() for l in f if l.strip()]
