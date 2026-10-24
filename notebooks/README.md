# Notebooks

| Notebook | Purpose |
|---|---|
| `colab_quickstart.ipynb` | Clone the repo in Colab and run `app.py` / the evaluation scripts. **Start here.** |
| `colab_v2_app_and_evaluation.ipynb` | The original v2 notebook (previously `accuracy_97.ipynb`), kept unchanged for reference: its saved outputs are where the published test-set numbers come from. |

## Known problems in the v2 notebook

These are fixed in the `dental_ai` package and `scripts/`; the notebook itself is left as it was.

- **"Accuracy 97%" is not a detection metric.** It is computed from the confusion matrix after dropping the
  background row and column, which removes every missed and every spurious detection. The notebook's own
  detection metrics on the test split are **mAP@50 38.7%, precision 40.3%, recall 37.5%**.
- **Per-class results are attached to the wrong class names.** `box.ap50`, `box.p` and `box.r` only contain
  classes present in the split (in `box.ap_class_index` order); the notebook pads them to 31 and indexes by
  position. That is why the per-class chart shows e.g. *TAD* (1 training box) at 94% AP and *impacted tooth*
  (19,573 training boxes) at 0%. Use `scripts/evaluate.py`.
- **Evaluation used `conf=0.25`**, which truncates the precision–recall curve and under-reports mAP.
  ultralytics' default for mAP is `conf=0.001`.
- **The confusion-matrix fallback fabricates data** from recall values when the real matrix is unavailable.
- **RGB arrays were passed to `model()`,** which treats NumPy arrays as BGR.
- **The "GradCAM" was a class-agnostic activation map** (mean/max of one layer, no gradients), with a Sobel
  edge-map fallback that was labelled as model attention.
