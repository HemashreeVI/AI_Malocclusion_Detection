"""Train the detector, then evaluate it honestly on the test split.

    python scripts/train.py --data data/merged/data.yaml [--model yolov9c.pt] [--imgsz 1024]

Recommended pipeline:
    python scripts/check_leakage.py   --data dental.yaml --resplit splits/
    python scripts/prepare_dataset.py --data dental.yaml --splits splits/ --out data/merged
    python scripts/train.py           --data data/merged/data.yaml
"""
import argparse
import os
import shutil

import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default='yolov9c.pt', help='pretrained checkpoint or model YAML')
    ap.add_argument('--imgsz', type=int, default=1024,
                    help='panoramics are wide and lesions small; 1024 keeps more detail than 640')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=-1, help='-1 = auto batch size for available GPU memory')
    ap.add_argument('--device', default=None)
    ap.add_argument('--project', default=os.path.join(ROOT, 'runs'))
    ap.add_argument('--name', default='dental')
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
        project=args.project, name=args.name, exist_ok=True, seed=0, deterministic=True,
        patience=30, cos_lr=True, close_mosaic=15,
        # Radiographs are grayscale: colour jitter only adds noise.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        # Mild geometric augmentation; horizontal flips are fine for these
        # (side-agnostic) classes, vertical flips are not anatomically plausible.
        degrees=3.0, translate=0.1, scale=0.3, fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.0,
    )

    best = os.path.join(args.project, args.name, 'weights', 'best.pt')
    print(f'\nBest weights: {best}')
    os.makedirs(os.path.join(ROOT, 'weights'), exist_ok=True)
    shutil.copy(best, os.path.join(ROOT, 'weights', 'best.pt'))
    evaluate.main(['--weights', best, '--data', args.data, '--split', 'test', '--imgsz', str(args.imgsz),
                   '--update-gating'])


if __name__ == '__main__':
    main()
