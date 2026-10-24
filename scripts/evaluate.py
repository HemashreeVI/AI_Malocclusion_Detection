"""Evaluate the detector on a dataset split and write honest metrics.

    python scripts/evaluate.py --weights weights/best.pt --data dental.yaml --split test --update-gating

Writes to --out (default results/):
  metrics.json       overall + per-class precision/recall/AP with instance counts
  per_class.md       the same as a Markdown table (paste into the README)
  ap50_per_class.png per-class AP@50, annotated with test instance counts
and, with --update-gating, dental_ai/class_ap50.json, which the app uses to
decide which classes are "validated".

Differences from the old notebook evaluation:
* conf defaults to 0.001 (ultralytics' default for mAP). conf=0.25 truncates
  the precision-recall curve and under-reports mAP.
* Per-class results are mapped through ``box.ap_class_index``. ultralytics
  only returns entries for classes present in the split; indexing them by
  position attaches values to the wrong class names.
* No "accuracy" from the confusion matrix. Dropping the background row/column
  ignores every missed and every spurious detection, so it is not a detection
  metric. No confusion matrix is ever synthesised from recall values.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def per_class_results(metrics, names):
    """Rows for every class, including those absent from the split."""
    nt = metrics.nt_per_class if metrics.nt_per_class is not None else [0] * len(names)
    present = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    rows = []
    for c, name in names.items():
        row = {'class': name, 'instances': int(nt[c])}
        if c in present:
            p, r, ap50, ap = metrics.box.class_result(present[c])
            row.update(precision=float(p), recall=float(r), ap50=float(ap50), ap50_95=float(ap))
        rows.append(row)
    return rows


def markdown_table(rows):
    lines = ['| Class | Test instances | Precision | Recall | AP@50 | AP@50-95 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda r: -r.get('ap50', -1)):
        if 'ap50' in r:
            lines.append(f"| {r['class']} | {r['instances']} | {r['precision']:.1%} | {r['recall']:.1%} | "
                         f"{r['ap50']:.1%} | {r['ap50_95']:.1%} |")
        else:
            lines.append(f"| {r['class']} | 0 | — | — | — | — |")
    return '\n'.join(lines)


def plot_ap50(rows, map50, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = sorted([r for r in rows if 'ap50' in r], key=lambda r: r['ap50'])
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 1.5))
    ax.barh([r['class'] for r in rows], [r['ap50'] for r in rows], color='#3b6fb6')
    for i, r in enumerate(rows):
        ax.text(r['ap50'] + 0.01, i, f"{r['ap50']:.0%}  (n={r['instances']})", va='center', fontsize=8)
    ax.axvline(map50, color='#c0392b', linestyle='--', linewidth=1, label=f'mAP@50 {map50:.1%}')
    ax.set_xlim(0, 1.25)
    ax.set_xlabel('AP@50 (n = test instances)')
    ax.set_title('Per-class AP@50 (classes present in the split)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default=os.path.join(ROOT, 'weights', 'best.pt'))
    ap.add_argument('--data', required=True, help='dataset YAML')
    ap.add_argument('--split', default='test')
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--iou', type=float, default=0.7, help='NMS IoU (ultralytics default)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--update-gating', action='store_true',
                    help='write dental_ai/class_ap50.json so the app gates on these results')
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)
    metrics = model.val(data=args.data, split=args.split, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                        device=args.device, plots=True, project=args.out, name='val', exist_ok=True)

    names = model.names
    rows = per_class_results(metrics, names)
    absent = [r['class'] for r in rows if 'ap50' not in r]
    summary = {
        'weights': os.path.relpath(args.weights, ROOT) if args.weights.startswith(ROOT) else args.weights,
        'split': args.split, 'conf': args.conf, 'iou': args.iou, 'imgsz': args.imgsz,
        'classes_evaluated': len(rows) - len(absent),
        'classes_absent_from_split': absent,
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'mean_precision': float(metrics.box.mp),
        'mean_recall': float(metrics.box.mr),
        'speed_ms_per_image': {k: round(v, 2) for k, v in metrics.speed.items()},
        'per_class': rows,
    }
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, 'per_class.md'), 'w') as f:
        f.write(markdown_table(rows) + '\n')
    plot_ap50(rows, summary['mAP50'], os.path.join(args.out, 'ap50_per_class.png'))

    print(f"\nmAP@50 {summary['mAP50']:.1%}   mAP@50-95 {summary['mAP50_95']:.1%}   "
          f"P {summary['mean_precision']:.1%}   R {summary['mean_recall']:.1%}")
    print(f"{summary['classes_evaluated']} classes evaluated; absent from '{args.split}': {', '.join(absent) or 'none'}")
    print(markdown_table(rows))

    if args.update_gating:
        gating = {
            '_source': f"scripts/evaluate.py on split '{args.split}' (conf={args.conf}, iou={args.iou}). "
                       f"Classes absent from the split are omitted and treated as unvalidated.",
            'mAP50': summary['mAP50'],
            'ap50': {r['class']: round(r['ap50'], 4) for r in rows if 'ap50' in r},
            'instances': {r['class']: r['instances'] for r in rows if 'ap50' in r},
        }
        path = os.path.join(ROOT, 'dental_ai', 'class_ap50.json')
        with open(path, 'w') as f:
            json.dump(gating, f, indent=2)
        print(f'Gating file written: {path}')
    return summary


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
